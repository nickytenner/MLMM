import os,sys
sys.path.append(os.getcwd() + '/mlmm/qmmlpack/python')
try:
   import qmmlpack as qmml
   qmmlpackexists = True
except:
   qmmlpackexists = False
   print("WARNING: Unable to import qmmlpack!! continuing")

import numpy as np
import math
import qml
import copy
import matplotlib
import platform
if platform.system() == 'Darwin':
   os.environ['KMP_DUPLICATE_LIB_OK']='True'
matplotlib.use('Agg')
import pyprind, time
import matplotlib.pyplot as plt
import numpy.random as rand
from qml.representations import *
from qml.fchl import generate_representation,get_local_kernels,get_local_symmetric_kernels
from sklearn.kernel_ridge import KernelRidge
from periodictable import elements as el
from qml.kernels import gaussian_kernel
from qml.math import cho_solve
from matplotlib.ticker import FormatStrFormatter
from collections import Counter
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer
rn = rand.random

class fileinfo (object):
    def __init__(self,*args,**kwargs):
        kw = {}
        kw.update(kwargs)
        self.reg = kw.get('reg',None)
        self.xdata = kw.get('xdata',None)
        self.ydata = kw.get('ydata',None)
        self.conf = kw.get('conf',None)
        assert self.reg is not None and self.xdata is not None and self.ydata is not None and self.conf is not None           
        self.reg_name = self.reg.__class__.__name__
        self.reg_dict_keys1 = ','.join([str(elem) for elem in list(self.reg.__dict__.keys())[:4]])
        #dict_vals = [elem for elem in list(self.reg.__dict__.values())[:4]]
        vals = []
        for elem in [elem for elem in list(self.reg.__dict__.values())[:4]]: # dict_vals:
           if isinstance(elem,str) or isinstance(elem,int) or isinstance(elem,float) or isinstance(elem,bool) or isinstance(elem,list):
              if isinstance(elem,list):
                 vals.append('model'+':'.join(str(a) for a in elem))
              else:
                 vals.append(str(elem))
           else:
              vals.append(str(elem.__class__.__name__))
        xdata_head,xdata_tail=os.path.split(self.xdata)
        ydata_head,ydata_tail=os.path.split(self.ydata)
        #self.reg_dict_vals = '_'.join([str(elem) for elem in list(self.reg.__dict__.values())[:4]])
        #self.reg_dict_vals1 = ','.join([str(elem) for elem in list(self.reg.__dict__.values())[:4]])
        self.reg_dict_vals = '_'.join(vals)
        self.reg_dict_vals1 = ','.join(vals)
        #print (self.reg_dict_vals)
        self.data_name = xdata_tail[:-4]+'_'+ydata_tail+'_'+self.conf.rep
        self.data_reg_name = self.reg_name+'_'+str(self.conf.frames[0])+'_'+str(self.conf.frames[1]) #+'_'+self.data_name
        #print (self.data_name)
        self.directory = os.path.join('./',xdata_head,self.data_name)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        print ('directory:%s\n data_reg_name: %s\n reg_dict_vals: %s\n'%(self.directory,self.data_reg_name,self.reg_dict_vals))
        self.lc_fname = self.directory+'/'+'lc_'+self.data_reg_name+'_'+self.reg_dict_vals+'.png'
        self.res_name = self.directory+'/'+'pred_'+self.data_reg_name+'_'+self.reg_dict_vals+'.txt'
        self.io_name = self.directory+'/'+'io_'+self.data_reg_name+'_'+self.reg_dict_vals+'.txt'
        self.xyz_start = self.directory+'/'+'xyz_'+self.data_reg_name+'_'+self.reg_dict_vals+'_start.xyz'
        self.xyz_end = self.directory+'/'+'xyz_'+self.data_reg_name+'_'+self.reg_dict_vals+'_end.xyz'
        self.mc_log = self.directory+'/'+'mclog_'+self.data_reg_name+'_'+self.reg_dict_vals+'.log'
        self.data_fname = self.directory+'/'+self.reg_name+'_fit_results.txt'
        print('REGRESSOR:'+self.reg_name)
        print('SETTINGES:')
        print(self.reg.__dict__)
   
def plot_learn_curve(estimator,X,y,ylim=None,n_fold=10,train_sizes=np.linspace(0.1,1,10),fname='learning_curve.png',etype='MAE'):
    lenX= len(X)
    ftxt = fname[:-3]+'txt'
    sam_size = []
    in_sam_err = []
    out_sam_err = []
    for elem in train_sizes:
        X_lc = X[:int(elem*lenX)]
        y_lc = y[:int(elem*lenX)]
        if estimator.__class__.__name__ == 'tf_net':
           estimator.build_model(input_shape=np.shape(X_lc)[1])
        plt_y_test, plt_y_pred, loss_out, loss_out_std, loss_in,loss_in_std, r_squaredAll,r_squared_std = n_fold_cv(n_fold,X_lc,y_lc,estimator,etype)
        print('Sample size:',len(X_lc),' In sample ',str(etype), ':',loss_in,' Out sample ', str(etype),':', loss_out)
        sam_size.append(len(X_lc))
        in_sam_err.append(loss_in)
        out_sam_err.append(loss_out)
        X_lc = None
        y_lc = None
    plt.figure()
    fig,ax=plt.subplots()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    #plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid()
    plt.xlabel("Training examples")
    plt.ylabel(etype)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.plot(sam_size, in_sam_err, 'o-', color="r",
             label="In sample")
    plt.plot(sam_size, out_sam_err, 'o-', color="g",
             label="Out of sample")
    with open(ftxt,'w') as fp:
        fp.write("sample_size in_sam_err out_sam_err\n")
        for a,b,c in zip(sam_size,in_sam_err,out_sam_err):
            fp.write("%d %f %f\n"%(a,b,c))
    plt.legend(loc="best")
    plt.savefig(fname)
    plt.figure()
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid()
    plt.xlabel("Training examples")
    plt.ylabel(etype)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.loglog(sam_size, in_sam_err, 'o-', color="r",
             label="In sample")
    plt.loglog(sam_size, out_sam_err, 'o-', color="g",
             label="Out of sample")
    plt.legend(loc="best")
    fname_lst = list(fname)
    fname_lst[-4:] = '_ll.png'
    fname = "".join(fname_lst)
    plt.savefig(fname)
    return

def n_fold_cv (n,X,y,reg,er,info=None):
    if len(X.shape) != 2:
       print("ERROR: X data is not in a block shape. Double check your representation!")
       quit()
    r_squared = []
    out_error = []
    in_error = []
    plt_y_test = []
    plt_y_pred = []
    plt_x_test = []
    for i in range(1):
        kf = KFold(n_splits=n,shuffle=True)
        for train, test in kf.split(X):
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
            if reg.__class__.__name__ == 'tf_net':
               reg.fit(X_train,y_train)
            else:
               reg.fit(X_train, y_train)
            #if "ColVel" in info.conf.rep: # in ['1norm','2norm']:
            #   plt_x_test.extend(X_test)
            y_pred = reg.predict(X_test)
            r2_val = r2_score (y_test,y_pred)
            plt_y_test.extend(y_test)
            plt_y_pred.extend(y_pred)
            if r2_val > 0.0:
                r_squared.append(r2_val)
                if er == 'MSE':
                    loss = mean_squared_error(y_test,y_pred)
                    in_error.append(mean_squared_error(y_train,reg.predict(X_train)))
                elif er == 'MAE':
                    loss = mean_absolute_error(y_test,y_pred)
                    in_error.append(mean_absolute_error(y_train,reg.predict(X_train)))
                else:
                    print("ERROR: Unknown error metric:"+er)
                    quit()
                out_error.append(loss)                
                print((time.ctime() + ' ' +  er +'   :'+str(loss)+'  R-square   :'+str(r2_val)))
            else:
                print("Too poor fit to report. continuing...")
    if info is not None:
        with open (info.res_name,'w') as f:
            f.write("train_data    test_data\n")
            for train,pred in zip(plt_y_test,plt_y_pred):
                f.write("%f   %f\n"%(train,pred))
        if "ColVel" in info.conf.rep: # in ['1norm','2norm']:
           with open(info.io_name,'w') as f:
              f.write("vel_in vel_out_act vel_out_pred\n")
              for vin,vout1,vout2 in zip(plt_x_test,plt_y_test, plt_y_pred):
                 f.write("%s %s %s\n"%(str(vin[0]),str(vout1),str(vout2)))
        print(('Kfold '+ str(er)+ ':'+str(np.mean(out_error))+' rSquaredEr :'+str(np.mean(r_squared))))
        if not os.path.isfile(info.data_fname):
            with open(info.data_fname,'w') as fp:
                fp.write ("%s, %s ,%sstd, R^2, R^2_std, Learning curve, Prediction\n" % (info.reg_dict_keys1,er,(er+'std')))
        with open(info.data_fname, 'a') as fp:
                fp.write("%s, %g, %g, %g, %g, %s, %s\n" % (info.reg_dict_vals1 , np.mean(out_error),np.std(out_error),np.mean(r_squared),np.std(r_squared), info.lc_fname, info.res_name))  
    return plt_y_test, plt_y_pred,np.mean(out_error),np.std(out_error),np.mean(in_error),np.std(in_error),np.mean(r_squared),np.std(r_squared)

    
