#!/usr/bin/env python
# gridsearch.py
import os,sys
sys.path.append(os.getcwd() + '/mlmm/qmmlpack/python')
try:
    import qmmlpack as qmml
    qmmlpackexists = True
except:
    print ("Unable to import qmmlpack!! MBTR calculation will fail! Continuing..")
    qmmlpackexists = False
    
import numpy as np                 # numeric routines for arrays
import hashlib                     # hashing functions
import itertools                   # helper routines for iteration constructs
import zipfile                     # reading and writing packed files in ZIP format
import math
from .machine import *
from .representations import StructureList,EACTrajectory, VelocityData, CompositeGrid

def cached(max_entries=sys.maxsize):
    """Memoization decorator.
    
    Allows caching of function values for representation and kernel functions.
    
    A maximum number of results to be stored can be specified via max_entries.
    
    functools.lru_cache does not accept mutable objects such as NumPy's ndarray.
    
    Caching the functions directly (as opposed to automatic caching within code for experiments) 
    allows caching over multiple experiments."""
    return lambda f: _cached(f, max_entries=max_entries)

class _cached:
    """Implementation of memoization decorator."""
    
    def __init__(self, f, max_entries=sys.maxsize):
        """Wraps f in a caching wrapper."""
        self.f = f  # cached function
        self.max_entries = max_entries  # maximum number of results to store
        self.cache = {}  # stores pairs of hash keys and function results 
        self.lruc = 0  # least recently used counter
        self.hashf = None  # hashing function, initialized by __call__
        self._intbuf = np.empty((1,), dtype=np.int)  # buffer for temporary storage of hash() values
        self.hits, self.misses = 0, 0  # number of times function values were retrieved from cache/had to be computed 
        self.total_hits, self.total_misses = 0, 0  # statistics over lifetime of cached object
        
    def _hash(self, arg):
        """Hashes an argument."""
        try:
            self.hashf.update(arg)
        except:
            self._intbuf[0] = hash(arg)
            self.hashf.update(self._intbuf)
        
    def __call__(self, *args, **kwargs):
        """Calls to cached function."""
        
        # compute hash key of arguments
        self.hashf = hashlib.md5()  # non-cryptographic fast hash function
        for i in args: self._hash(i)
        for i in kwargs: self._hash(kwargs[i])
        key = self.hashf.digest()
        
        # return stored or computed value
        if key in self.cache:
            self.total_hits, self.hits = self.total_hits + 1, self.hits + 1
            item = self.cache[key]
            item[0] = self.lruc; self.lruc += 1
            return item[1]
        else:
            self.total_misses, self.misses = self.total_misses + 1, self.misses + 1
            value = self.f(*args, **kwargs)
            self.cache[key] = [self.lruc, value]; self.lruc += 1

            # remove another entry if new entry exceeded the maximum
            if len(self.cache) > self.max_entries:
                c, k = self.lruc, key
                for k_,v_ in self.cache.items():
                    if v_[0] < c: c, k = v_[0], k_
                del self.cache[k]
            
            return value
        
    def clear_cache(self):
        """Empties cache, releasing memory."""
        self.cache.clear()
        self.hits, self.misses = 0, 0


class EvaluateModelPerformance:
    """Performance of a kernel ridge regression model.
    
    Returns loss of kernel regression model on validation set using given hyperparameters.
    Implemented as a stateful functor."""
    
    def __init__(self, z, r, y, indtrain, indvalid, reprf, kernelf, lossf, paramf, basis=None, centering=False, maxcache=False):
        """Sets up model performance evaluator.
        
        z, r, y, basis - atomic numbers, atom coordinates, property values and basis vectors
        indtrain, indvalid - indices into z, r, y, basis of training and validation sets
        reprf - function f(z, r, basis, theta) returning n x d representation matrix, where 
                n is number of data points and d is dimensionality of representation; can be a list of functions
        kernelf - kernel function
        lossf - loss function f(true, pred, state, z) returning a single summary statistic of the performance
        paramf - function accepting one parameter vector theta, which it splits into 
            individual parameter vectors for representation, kernel and method.
        centering - whether to center the kernelf
        maxcache - if True, reprf and kernelf will always be called for the whole input, 
                   maximixing cache exploitation across large series of experiments such as learning curves.
        
        If reprf is a list, representations are cached separately and concatenated.
        
        The employed representation and kernel functions should cache (memoize) their values.
        This caching is best done outside of this class by the actual functions to maximize speed gains across
        different instances of EvaluateModelPerformance."""

        # parameters
        self.z, self.r, self.y, self.indtrain, self.indvalid, self.reprf, self.kernelf, self.lossf, self.paramf,             self.b, self.centering, self.maxcache = z, r, y, indtrain, indvalid, reprf, kernelf, lossf, paramf,             basis, centering, maxcache
        if not qmml.is_sequence(self.reprf): self.reprf = (self.reprf,)  # always a sequence
        
    def reset(self, z=None, r=None, y=None, indtrain=None, indvalid=None, reprf=None, kernelf=None, lossf=None,               paramf=None, basis=None, centering=None, maxcache=None):
        """Modifies parametrization."""

        if z is not None: self.z = z
        if r is not None: self.r = r
        if y is not None: self.y = y
        if indtrain is not None: self.indtrain = indtrain
        if indvalid is not None: self.indvalid = indvalid
        if reprf is not None: self.reprf = reprf  # sequence will be ensured by __init__ below
        if kernelf is not None: self.kernelf = kernelf
        if lossf is not None: self.lossf = lossf
        if paramf is not None: self.paramf = paramf
        if basis is not None: self.b = basis
        if centering is not None: self.centering = centering
        if maxcache is not None: self.maxcache = maxcache

        self.__init__(self.z, self.r, self.y, self.indtrain, self.indvalid, self.reprf, self.kernelf, self.lossf,                       self.paramf, basis=self.b, centering=self.centering, maxcache=self.maxcache)

    def __call__(self, *theta):
        """Evaluates model performance for given hyperparameters theta."""

        # parameter vectors
        (theta_repr, theta_kernel, theta_method) = self.paramf(*theta)
        if len(theta_repr) == 0 or not qmml.is_sequence(theta_repr[0]): theta_repr = [theta_repr]
        theta_repr = tuple(tuple(theta) for theta in theta_repr)
        theta_kernel, theta_method = tuple(theta_kernel), tuple(theta_method)
        
        # representation and kernel matrices
        if self.maxcache:  # call reprf and kernelf for all inputs
            # representation
            repr_ = [rf(self.z, self.r, basis=self.b, theta=rt) for (rf,rt) in zip(self.reprf, theta_repr)]
            repr_ = np.concatenate(repr_, axis=1)
            # kernel matrix
            M = self.kernelf(repr_, theta=theta_kernel)
            # reduce to training and test sets
            reprtrain, reprvalid = repr_[self.indtrain], repr_[self.indvalid]
            K, L = M[np.ix_(self.indtrain, self.indtrain)], M[np.ix_(self.indtrain, self.indvalid)]
        else:  # call reprf and kernelf only for training and validation sets
            # representation
            (btrain, bvalid) = (None, None) if self.b is None else (self.b[self.indtrain], self.b[self.indvalid])
            reprtrain = [rf(self.z[self.indtrain], self.r[self.indtrain], basis=btrain, theta=rt) for (rf,rt) in zip(self.reprf, theta_repr)]
            reprvalid = [rf(self.z[self.indvalid], self.r[self.indvalid], basis=bvalid, theta=rt) for (rf,rt) in zip(self.reprf, theta_repr)]
            reprtrain, reprvalid = np.concatenate(reprtrain, axis=1), np.concatenate(reprvalid, axis=1)
            # kernel matrices
            K = self.kernelf(reprtrain, theta=theta_kernel)
            L = self.kernelf(reprtrain, reprvalid, theta=theta_kernel)
   
        # model and predictions
        krr = qmml.KernelRidgeRegression(K, self.y[self.indtrain], theta_method, centering=self.centering)
        pred = krr(L)

        # loss
        self.loss = lambda: None  # dummy object for storing additional data computed by lossf
        loss = self.lossf(self.y[self.indvalid], pred, state=self.loss)
        return loss



def paramf(aa, bb, cc):
    """Parameter splitting function dividing parameters into three blocks of sizes a, b, c.
    
    a, b, c are numbers of parameters.
    a can also be a list of numbers of parameters.
    
    paramf([1,2], 0, 1)(['a', 'b', 'c', 'd']) -> ([('a',), ('b', 'c')], tuple(), ('d',))"""
    def f(a, b, c, *theta):
        if qmml.is_sequence(a):
            # multiple representation functions
            # as we have only need for cases 2 and 3, no need to generalize
            assert len(a) == 2 or len(a) == 3
            resa = [ tuple(theta[:a[0]]), tuple(theta[a[0]:a[0]+a[1]]) ]
            if len(a) == 3: resa.append( tuple(theta[a[0]+a[1]:a[0]+a[1]+a[2]]) )
            a = np.sum(a)
        else:
            # single representation function
            resa = tuple(theta[:a])
        return ( resa, tuple(theta[a:a+b]), tuple(theta[a+b:a+b+c]) )
    return lambda *theta: f(aa, bb, cc, *theta)


def lossf(true, pred, state=None, return_='root_mean_squared_error'):
    """Augmented loss function.
    
    Parameters:
      true - reference labels
      pred - predicted labels
      state - if passed, reference values, predictions, residuals and loss function values will be
              stored as true, pred, residuals, loss. 
      return_ - name of summary statistic to return."""

    residuals = np.asfarray(pred - true)
    loss = qmml.loss(true, pred)
    
    # if state is available, store statistics
    if state is not None:
        state.true, state.pred, state.residuals, state.loss = true, pred, residuals, loss

    # quantity to return
    return loss[return_]

@cached(max_entries=30)
def reprf2(z, r, basis=None, theta=None, flatten=True):
    """MBTR, k=2, inverse distances, quadratic weighting"""
    (dsigma2,) = theta
    return qmml.many_body_tensor(
        z, r, (0.1, 1.1/100, 100),  # [1/dmax, 1/dmin] plus margin
        (2, '1/distance', 'identity^2', ('normal', (dsigma2,)), 'identity', 'noreversals', 'noreversals'),
        basis=basis, acc=0.001, elems=(1,6,7,8,14,16,17,34), flatten=flatten)

@cached(max_entries=30)
def reprf3(z, r, basis=None, theta=None, flatten=True):
    """MBTR, k=3, angles"""
    (dsigma3,) = theta
    return qmml.many_body_tensor(
        z, r, (-0.15, 3.45575/100, 100),  # [0,Pi] plus margin
        (3, 'angle', '1/dotdotdot', ('normal', (dsigma3,)), 'identity', 'noreversals', 'noreversals'),
        basis=basis, acc=0.001, elems=(1,6,7,8,14,16,17,34), flatten=flatten)
def printf(trialv, trialf, bestv, bestf, state):
    print('{:+#4.1f} {:+#4.1f} {:+#4.1f} {:+#4.1f} -> {:+#6.2f}'.format(*trialv, trialf))


#### DO NOT EDIT ABOVE THIS########

#conf = StructureList('adf_methane_out.xyz','adf_methane_E',frames=[0,10000])
#filename = 'gdb7-13.zip'  # the file as downloaded from the website
#with zipfile.ZipFile(filename, 'r') as zf:
#    zfcnt = zf.read('dsgdb7njp.xyz').decode('ascii')  # read packed file content into memory
#    raw = qmml.import_extxyz(zfcnt, additional_properties=True);  # parse XYZ format

#z = np.asarray([m.an for m in raw]);
#r = np.asarray([m.xyz for m in raw]);  # in units of angstrom
#e = np.asfarray([m.mp[0] for m in raw]);  # in units of kcal/mol

### Above this is not used####
#conf = StructureList('adf_pentane_out_5k.xyz','adf_pentane_E_5k',frames=[0,5000])
#conf.getRep (rep='mbtr',mbtr_f=2.**-4)
#z = np.asarray(conf.mbtr_z);
#r = np.asarray(conf.mbtr_r);
#e = np.asarray(conf.y);
#indvalid = np.random.choice(range(len(z)), size=1000, replace=False);  # 1000 draws without repetition
#indtrain = np.setdiff1d(range(len(z)), indvalid, assume_unique=True);  # remaining indices are training
#assert np.intersect1d(indtrain, indvalid).size == 0 and np.union1d(indvalid, indtrain).size == len(z)

#indvalid2 = np.random.choice(indtrain, size=100, replace=False);
#indtrain2 = np.setdiff1d(indtrain, indvalid2);
#assert np.intersect1d(indvalid2, indtrain2).size == 0

#kernelf_linear    = cached(max_entries=55)(qmml.kernel_linear)
#kernelf_gaussian  = cached(max_entries=55)(qmml.kernel_gaussian)



#variables = (
#    { 'value': -4.0, 'priority': 2, 'stepsize': 0.5 },  # 2-body MBTR normal distribution width
#    { 'value': -3.0, 'priority': 1, 'stepsize': 0.5 },  # 3-body MBTR normal distribution width
#    { 'value': +0.0, 'priority': 3, 'stepsize': 0.5 },  # Gaussian kernel sigma
#    { 'value': -20., 'priority': 4, 'direction': -1, 'minimum': -20 }  # regularization strength
#)


#f = EvaluateModelPerformance(z, r, e, indtrain2, indvalid2, reprf=(reprf2, reprf3), 
#    kernelf=kernelf_gaussian, lossf=lossf, paramf=paramf([1,1],1,1))
#optvars = qmml.local_grid_search(f, variables, resolution=0.001, evalmonitor=printf)[0]
#print('Done. Solution:', optvars)

#f.reset(indtrain=indtrain,indvalid=indvalid)
#f(*(2.**optvars))



class QuantumMachine(object):
    def __init__(self,*args,**kwargs):
        kw = {}
        kw.update(kwargs)
        self.xdata = kw.get('xdata',None)
        self.ydata = kw.get('ydata',None)
        assert self.xdata is not None            
        self.frames = kw.get('frames',[0,5000])
        self.rep = kw.get('rep','cm')
        #if self.rep not in ['1norm','2norm']:
        #if "ColVel" not in self.rep:
        #    assert self.ydata is not None
        self.reg = kw.get('reg',KernelRidge(kernel='rbf'))
        self.thetas = kw.get('thetas',[2.**-4,None])
        self.zeta = 0.0
        self.tralen = kw.get('tralen',200)
        self.nuc = kw.get('nuc',['Ar','Pt'])
        self.tramin = kw.get('tramin',25)
    def getRep(self):
        if self.rep in ['wvcm','wcm','wvel']:
            self.conf = EACTrajectory(self.xdata,self.ydata,traj_length=self.tralen,frames=self.frames)
            self.conf.getRep(rep=self.rep,nuc=self.nuc,tramin=self.tramin)
        elif "ColVel" in self.rep: # in ['1norm','2norm']:
            self.conf = VelocityData (self.xdata,frames=self.frames)
            self.conf.getRep (rep=self.rep,nuc=self.nuc)
        elif self.rep in ["fbd","fnbd","son",'dist','spring','cust','all']:
            self.conf = CompositeGrid(self.xdata, frames=self.frames)
            self.conf.getRep (rep=self.rep)
        else:
            self.conf = StructureList(self.xdata,self.ydata,frames=self.frames)
            self.conf.getRep(rep=self.rep,dsigma2=self.thetas[0],dsigma3=self.thetas[1],zeta=self.zeta)
    def pcmGridSearch(self):
        zeta = np.arange(0,0.2,0.01)
        zeta_error = {}
        for z in zeta:
            self.zeta = z
            print ("searching grid for zeta:" + str(self.zeta))
            self.getRep()
            X_train, X_test, y_train, y_test = train_test_split(
                self.conf.X, self.conf.y, test_size=0.5, random_state=0)
            param_space = [{'alpha': np.logspace(-10,-5,6),
                            'gamma': np.logspace(0,-5,6)}]
            reg = GridSearchCV(self.reg,cv=5,param_grid=param_space,scoring="neg_mean_absolute_error",verbose=2)
            reg.fit(X_train,y_train)        
            print("Grid scores on development set:")
            means = reg.cv_results_['mean_test_score']
            stds = reg.cv_results_['std_test_score']
            for mean, std, params in zip (means,stds, reg.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      %(mean,std*2,params))
            y_true,y_pred = y_test,reg.predict(X_test)
            print("Best parameters set found on development set:")
            print(reg.best_params_)
            mae = mean_absolute_error(y_true,y_pred)
            print ("MAE:%f"% mae)
            alpha,gamma = reg.best_params_.values()
            zeta_error[float(mae)] = [self.zeta,alpha,gamma]
        print ("MAE:[zeta, best_alpha, best_gamma]")
        print (zeta_error)
        return (zeta_error[min(zeta_error)])
    def mbtrGridSearch(self):
        assert self.rep == 'mbtr3' and self.reg.__class__.__name__ == 'QmmlMBTR'
        print ("Grid search for MBTR3 started")
        z = np.asarray(self.conf.mbtr_z);
        r = np.asarray(self.conf.mbtr_r);
        e = np.asarray(self.conf.y);
        indvalid = np.random.choice(range(len(z)), size=1000, replace=False);  # 1000 draws without repetition
        indtrain = np.setdiff1d(range(len(z)), indvalid, assume_unique=True);  # remaining indices are training
        assert np.intersect1d(indtrain, indvalid).size == 0 and np.union1d(indvalid, indtrain).size == len(z)
        indvalid2 = np.random.choice(indtrain, size=100, replace=False);
        indtrain2 = np.setdiff1d(indtrain, indvalid2);
        assert np.intersect1d(indvalid2, indtrain2).size == 0
        kernelf_linear    = cached(max_entries=55)(qmml.kernel_linear)
        kernelf_gaussian  = cached(max_entries=55)(qmml.kernel_gaussian)
        variables = (
            { 'value': -4.0, 'priority': 2, 'stepsize': 0.5 },  # 2-body MBTR normal distribution width
            { 'value': -3.0, 'priority': 1, 'stepsize': 0.5 },  # 3-body MBTR normal distribution width
            { 'value': +0.0, 'priority': 3, 'stepsize': 0.5 },  # Gaussian kernel sigma
            { 'value': -20., 'priority': 4, 'direction': -1, 'minimum': -20 }  # regularization strength
        )
        f = EvaluateModelPerformance(z, r, e, indtrain2, indvalid2, reprf=(reprf2, reprf3),
                                     kernelf=kernelf_gaussian, lossf=lossf, paramf=paramf([1,1],1,1))
        optvars = qmml.local_grid_search(f, variables, resolution=0.001, evalmonitor=printf)[0]
        #print('Done. Solution:', optvars)
        print ('Optimum MBTR3 parameters:')
        print ('dsigma2: %g\ndsigma3: %g\nsig:%e\nalpha:%e'%(2.**optvars[0],2.**optvars[1],2.**optvars[2],2.**optvars[3]))
        f.reset(indtrain=indtrain,indvalid=indvalid)
        print('MAE for optimum parameters:%f'% f(*(2.**optvars)))
        return (2.**optvars[0],2.**optvars[1],2.**optvars[2],2.**optvars[3])
    def regGridSearch(self,paramspace=None):
        X_train, X_test, y_train, y_test = train_test_split(
            self.conf.X, self.conf.y, test_size=0.5, random_state=0)
        if paramspace is None:
            param_space = [{'alpha': np.logspace(-10,-5,6),
                       'gamma': np.logspace(-5,-15,11)}]
        else: param_space = paramspace
        reg = GridSearchCV(self.reg,cv=5,param_grid=param_space,scoring="neg_mean_absolute_error",verbose=2)
        reg.fit(X_train,y_train)        
        print("Grid scores on development set:")
        means = reg.cv_results_['mean_test_score']
        stds = reg.cv_results_['std_test_score']
        for mean, std, params in zip (means,stds, reg.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  %(mean,std*2,params))
        y_true,y_pred = y_test,reg.predict(X_test)
        print ("Param search space:",param_space)
        print("Best parameters set found on development set:")
        print(reg.best_params_)
        print ("MAE:%f"% mean_absolute_error(y_true,y_pred))
        return (reg.best_params_.values())
 
