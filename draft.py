import os
import sys
import math

# from shapely.geometry import Polygon as Pol2

import dmsh
import meshio
import optimesh
import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
import scipy
import torch
import random

import sys
from sklearn.cluster import KMeans
from scipy.interpolate import Rbf
from scipy.optimize import minimize
from scipy.stats import qmc
import pandas as pd



from geometry import Polygon, Annulus
from utils import extract_path_from_dir, save_eps, plot_figures, grf, spread_points, closest, wn_PnPoly
from packages.my_packages import Gauss_zeidel, interpolation_2D

from constants import Constants
from packages.my_packages import *
from two_d_data_set import create_loader 
import time

def loss(a,*args):
        basis,f, x,y=args
        assert len(a)==len(basis)
        return np.linalg.norm(np.sum(np.array([a[i]*func(np.array([x, y]).T) for i,func in enumerate(basis)]),axis=0)-f)**2


def create_data(domain):
    x=domain['interior_points'][:,0]
    y=domain['interior_points'][:,1]
    M=domain['M']
    angle_fourier=domain['angle_fourier']
    T=domain['translation']
    A = (-M - Constants.k* scipy.sparse.identity(M.shape[0]))
    test_functions=domain['radial_basis']
    V=[np.array(func(x,y)) for func in test_functions]
    F=[v for v in V]
    U=[scipy.sparse.linalg.spsolve(A,b) for b in F]

    


    return x,y,F, U, angle_fourier, T

def expand_function(f,domain):
            
    # ev,V=scipy.sparse.linalg.eigs(-domain['M'],k=10, return_eigenvectors=True,which="SR")



    f=np.array(f)
    a=f[domain['hot_indices']]
    # if len(a)<77:
    #      a=np.hstack((a,np.zeros(2,)))
    return a
   
def expand_domain(domain):
    return domain['hot_points']



def generate_domains(S,T,n1,n2):

    # x1=[3/4,3/8,1/4,3/8]
    # y1=[1/2,5/8,1/2,3/8] 
    x1=[0,1,1,0]
    y1=[0,0,1,1] 
    X=[]
    Y=[]
    for j in range(len(x1)):
        
        p=T@np.array([x1[j],y1[j]])
        # new_p=S@(p-np.array([0.5,0.5]))+np.array([0.5,0.5])+T
        new_p=S@p
        X.append(new_p[0])
        Y.append(new_p[1])
    domain=Polygon(np.vstack((np.array(X),np.array(Y))).T, S)
    # domain=Annulus(np.vstack((np.array(X),np.array(Y))).T, T)
    # domain.plot(domain.generators)
                # plt.gca().set_aspect('equal', adjustable='box')  
             
                # plt.xlim([0,1])
                # plt.ylim([0,1])

    domain.create_mesh(1/10)     
   
    domain.save(Constants.path+'polygons/10'+str(n1)+str(n2)+'.pt')
    # domain.save(Constants.path+'hints_polygons/10_1150'+str(n1)+str(n2)+'.pt')

                # domain.create_mesh(0.05)
                # domain.save(Constants.path+'hints_polygons/005_1150'+str(n1)+str(n2)+'.pt')

                # # domain.plot_geo(domain.X, domain.cells, domain.geo)
                
                # domain.plot_geo(domain.X, domain.cells, domain.geo)
                # domain.save(Constants.path+'hints_polygons/80_tta_2_1150'+str(n1)+str(n2)+'.pt')
                # print('sucess')
                
            # except:
                # print('failed')    

def create_lshape():
    domain=Polygon(np.array([[0,0],[1,0],[1,1/2],[1/2,1/2],[1/2,3/2],[0,3/2]]))
    domain.create_mesh(0.16)
    
    # ind=np.lexsort((domain.interior_points[:,0],domain.interior_points[:,1]))
    domain.hot_points=domain.interior_points
    # domain.hot_indices=ind
    domain.save(Constants.path+'hints_polygons/lshape.pt')
   

    # dmsh.show(domain['X'], domain['cells'], domain['geo'])
    # plt.scatter(domain['interior_points'][:,0], domain['interior_points'][:,1],color='black')
    # plt.scatter(domain['interior_points'][:,0], domain['interior_points'][:,1],color='red')
    # plt.show()

def create_base_domains(p,name):
    # p=np.array([[0,0],[1,0],[1,1],[0,1]])
    # # create_base_domains(p,'rect')
    domain=Polygon(p)
    domain.create_mesh(0.1)
    domain.save(Constants.path+'base_polygons/'+name+'.pt')


def create_sub_domain(A, base_domain,name):
    domain=Polygon((A@base_domain['generators'].T).T, A, base_domain)
    domain.create_mesh(0.15)
    domain.save(Constants.path+'polygons/'+name+'.pt')
    
# p=np.array([[0,0],[1,0],[1,1],[0,1]])
# create_base_domains(p,'rect')

# p=np.array([[0,0],[1,0],[1,0.5],[0.5,0.5],[0.5,1.5],[0,1.5]])
# create_base_domains(p,'l_shape')

# if __name__=='__main__':
#     for m,i in enumerate(np.linspace(0.5,1.5,5)):
#         for n,j in enumerate(np.linspace(0.5,1.5,5)):
#             base_domain=torch.load(Constants.path+'base_polygons/rect.pt')
#             A=np.array([[i,0],[0,j]])
#             create_sub_domain(A, base_domain,'ex'+str(m)+str(n))



# domain=torch.load(Constants.path+'polygons/ex04.pt')
# # domain=torch.load(Constants.path+'base_polygons/l_shape.pt')
# domain=torch.load(Constants.path+'base_polygons/rect.pt')
# A=domain['angle_function'].reshape((20, 20))
# plt.imshow(A, cmap="gray")
# plt.show()
# sample=grf(domain['interior_points'][:,0], 1, seed=0, sigma=0.1, mu=0 )
# func=interpolation_2D(domain['interior_points'][:,0],domain['interior_points'][:,1],sample[0])

# print(func(np.array([0]),np.array([1])))


        # np.array([wn_PnPoly(g,poly) for g in grid ])
        # # return np.array([wn_PnPoly(g,poly) for g in grid ]).reshape((N, N))
        # return np.array([wn_PnPoly(g,poly) for g in grid ])

# print(f_grid(domain, sample[0]))
# Polygon.plot_geo(domain)


# Polygon.plot_angle_function(domain)



