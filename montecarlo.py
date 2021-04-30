from .machine import *
from .representations import count_frame

class mctrajectory(object):
    def __init__(self,*args,**kwargs):
        kw = {}
        kw.update(kwargs)
        self.infile = kw.get('fname',None)
        self.box = kw.get('box',[[0,1000],[0,1000],[0,1000]])
        self.frame=kw.get('frame',1)
        self.natom = 0
        self.coord = []
        self.new_coord = []
        self.ene = 0.0
        self.new_ene = 0.0
        self.update()
        return
    def update(self):
        nframe = count_frame(self.infile)[0]
        if nframe < self.frame:
            print ("ERROR: requested frame %d is more than the total available frame %d" % (self.nframe,nframe))
            quit()
        else:
            with open(self.infile) as fc:
                for i in range(self.frame):
                    self.coord = []
                    self.natom = int(fc.readline())
                    fc.readline()
                    for j in range(self.natom):
                        cline = fc.readline()
                        axyz = cline.split()
                        self.coord.append([axyz[0],float(axyz[1]),float(axyz[2]),float(axyz[3])])
        for i in range(len(self.coord)):
            for j in range(3):
                if self.coord[i][j+1] < self.box[j][0] or self.coord[i][j+1] > self.box[j][1]:
                    print (self.coord[i], self.box)
                    print ("ERROR: Found atoms outside the box. Correct the input geometry")
                    quit()
                self.coord[i][j+1] -= self.box[j][0]
        for j in range(3):
            self.box[j][1] = self.box[j][1]-self.box[j][0]
            self.box[j][0] = 0.0     
    def energy(self, coord, repr = None, regr=None):
        #print (coord)
        repr.getLocRep(coord)
        #print (repr.lX)
        if repr.rep == 'mbtr3':
            return regr.predict(repr.lX)[0]
        else:
            return regr.predict([repr.lX])[0]
     
    #def calc_pair_energy(self,conf,reg,d2=False,mollen='all',cut_off=15.0):
    #    if mollen == 'all':
    #        mollen = len(conf)
    #    mol_list = []
    #    for i in range(conf/mollen):
    #        mol = []
    #        for j in range(mollen):
    #            mol.append(conf[i*mollen+j])
    #        mol_list.append(mol)
    #    if len(mol_list) == 1:
    #        pair_list = mol_list[0]
    #    else:
    #        pair_list = []
    #        per_ind = per_img_ind(3)
    #        for i in range(len(mol_list)):
    #            for j in range(i):
    #                for image in per_ind:
    #                    mol1 = mol_list[i]
    #                    mol2 = []
    #                    for elem in mol_list[j]:
    #                        mol2.append([elem[0]]+[elem[k+1] + self.box[k][1]*image[k] for k in range(3)])
    #                    com1 = calc_com(mol1)
    #                    com2 = calc_com(mol2)
    #                    com_dist = distance(com1,com2)
    #                    if com_dist < cut_off:
    #                        pair_list.append(mol_list[i]+mol_list[j])
    #    toten = 0;
    #    for elem in pair_list:
    #        toten += self.energy(elem,regr=reg,d2=d2)
    #    return            
    def new_move(self,dr):
        new_xyz = []
        for elem in self.coord:
            new_xyz.append([elem[0]]+[ele+dr*(2*rn()-1.0) for ele in elem[1:]])
        self.new_coord = new_xyz
    def accept(self):
        for i in range(len(self.new_coord)):
            for j in range(3):
                x = self.new_coord[i][j+1]
                x_size = self.box[j][1]
                self.new_coord[i][j+1] = x - math.floor(x/x_size)*x_size
        self.coord = self.new_coord
    def dump(self,fname='dump.xyz',mode='w',data='axyz'):
        with open (fname,mode) as f:
            f.write("%d\n\n" % self.natom)
            for elem in self.coord:
                if data == 'axyz':
                    f.write('%5s %10.4f %10.4f %10.4f\n' %(elem[0],elem[1],elem[2],elem[3]))
    def gen_feat(self,coord,d2=False):
        dist = []
        dist2 = []
        for i in range(len(coord)):
            for j in range(i):
                x = coord[i]
                y = coord[j]
                d = ((x[1]-y[1])**2+(x[2]-y[2])**2+(x[3]-y[3])**2)**0.5
                dist.append(d)
                dist2.append(d**2)
        if d2:
            return [dist+dist2]
        else:
            return [dist]
        
