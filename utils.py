

import pandas as pd
from sklearn.metrics import pairwise_distances
import random
from tqdm import tqdm
import datetime
import pickle
import math
import random
from scipy.stats import gaussian_kde
import cmath
import os
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np
import scipy
from typing import List, Tuple
from packages.my_packages import *

import torch


from constants import Constants

# plt.scatter(d['interior_points'][:,0], d['interior_points'][:,1],color='black')
# points=d['hot_points']
# for i in range(len(points)):
#     x = points[i][0]
#     y = points[i][1]
#     plt.plot(x, y, 'ro')
#     plt.text(x * (1 ), y * (1 ) , i, fontsize=10)
# plt.show()


def calc_angle(u,v=[1,0]):
    v=np.array(v)
    u=np.array(u)
    if u[1]>=0:
      return np.arccos(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)))
    else:
      return 2*math.pi-np.arccos(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)))
class norms:
    def __init__(self): 
        pass
    @classmethod
    def relative_L2(cls,x,y):
        return torch.linalg.norm(x-y)/(torch.linalg.norm(y)+1e-10)
    @classmethod
    def relative_L1(cls,x,y):
        return torch.nn.L1Loss()(x,y)/(torch.nn.L1Loss(y,y*0)+1e-10)
    
def grf(domain, n, seed=0, mu=0, sigma=math.sqrt(0.1)):
    np.random.seed(seed)
    A=np.array([np.random.normal(mu, sigma,n) for i in range(len(domain)) ]).T

    # [plt.plot(domain, np.sqrt(2)*A[i,:]) for i in range(n)]
    # plt.show(block=False)
    # torch.save(A, Constants.outputs_path+'grf.pt')
    return np.sqrt(2)*A

def plot_polygon(ax, poly, **kwargs):
    path = Path.make_compound_path(
        Path(np.asarray(poly.exterior.coords)[:, :2]),
        *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])

    patch = PathPatch(path, **kwargs)
    collection = PatchCollection([patch], **kwargs)
    
    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection

def clip(value, lower, upper):
    """
    Given an interval, values outside the interval are clipped to the interval
    edges.
    """
    return min(upper, max(value, lower))



def count_trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def polygon_centre_area(vertices):
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def polygon_centroid(vertices):
    A = polygon_centre_area(vertices)
    x = vertices[:, 0]
    y = vertices[:, 1]
    Cx = np.sum((x[:-1] + x[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1])) / 6 / A
    Cy = np.sum((y[:-1] + y[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1])) / 6 / A
    return Cx, Cy


def map_right(p1, p2, p3):
    B = np.array([[p1[0]], [p1[1]]])
    A = np.array([[p2[0] - B[0], p3[0] - B[0]], [p2[1] - B[1], p3[1] - B[1]]])

    return np.squeeze(A), B


def is_between(p1, p2, point):
    crossproduct = (point[1] - p1[1]) * (p2[0] - p1[0]) - (point[0] - p1[0]) * (
        p2[1] - p1[1]
    )

    # compare versus epsilon for floating point values, or != 0 if using integers
    if abs(crossproduct) > 1e-10:
        return False

    dotproduct = (point[0] - p1[0]) * (p2[0] - p1[0]) + (point[1] - p1[1]) * (
        p2[1] - p1[1]
    )
    if dotproduct < 0:
        return False

    squaredlengthba = (p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (
        p2[1] - p1[1]
    )
    if dotproduct > squaredlengthba:
        return False

    return True


def on_boundary(point, geo):
    for i in range(len(geo.__dict__["paths"])):
        p1 = geo.__dict__["paths"][i].__dict__["x0"]
        p2 = geo.__dict__["paths"][i].__dict__["x1"]
        if is_between(p1, p2, point):
            return True
    return False





def spread_points(subset_num,X):
    
    x=X[:,0]
    y=X[:,1]
    total_num = x.shape[0]
    xy = np.vstack([x, y])
    dens = gaussian_kde(xy)(xy)

    # Try playing around with this weight. Compare 1/dens,  1-dens, and (1-dens)**2
    weight = 1 / dens
    weight /= weight.sum()

    # Draw a sample using np.random.choice with the specified probabilities.
    # We'll need to view things as an object array because np.random.choice
    # expects a 1D array.
    dat = xy.T.ravel().view([('x', float), ('y', float)])
    # subset = np.random.choice(dat, subset_num, p=weight)
    subset = np.random.choice(dat, subset_num)
    return np.vstack((subset['x'], subset['y'])).T
    



def np_to_torch(x):
    return torch.tensor(x, dtype=Constants.dtype)


def save_file(f, dir, name):

    isExist = os.path.exists(dir)
    if not isExist:
        os.makedirs(dir)
    torch.save(f, dir + name + ".pt")
    return dir + name + ".pt"



def save_plots(train_loss, valid_loss, test_loss, metric_type: str, dir_path):

    # accuracy plots
    fig, ax = plt.subplots(1, 2)
    # plt.figure(figsize=(10, 7))
    ax[0].plot(train_loss[1:], color="orange", linestyle="-", label="train")
    ax[0].plot(valid_loss[1:], color="red", linestyle="-", label="validataion")
    ax[0].set(xlabel='Epochs', ylabel=metric_type)
    ax[0].legend(loc="upper right")

    ax[1].plot(test_loss, color="blue", linestyle="-", label="test")
    ax[1].set(xlabel='Epochs', ylabel=metric_type)
    ax[1].legend(loc="upper right")

    fig.suptitle("metric type: "+metric_type)
    isExist = os.path.exists(dir_path+'figures')
    if not isExist:
        os.makedirs(dir_path+'figures')

    plt.savefig(dir_path + "figures/" + metric_type+".png")
    plt.show(block=False)







def calc_min_angle(geo):
    seg1 = []
    for i in range(len(geo.__dict__["paths"])):
        p1 = geo.__dict__["paths"][i].__dict__["x0"]
        p2 = geo.__dict__["paths"][i].__dict__["x1"]
        seg1.append(p1)

    angle = []
    for i in range(len(seg1)):
        p1 = seg1[i % len(seg1)]
        p2 = seg1[(i - 1) % len(seg1)]
        p3 = seg1[(i + 1) % len(seg1)]
        angle.append(
            np.dot(p2 - p1, p3 - p1)
            / (np.linalg.norm(p2 - p1) * np.linalg.norm(p3 - p1))
        )
 
    return np.arccos(angle)




def solve_helmholtz(M, interior_indices, f):
    A = -M[interior_indices][:, interior_indices] - Constants.k * scipy.sparse.identity(
        len(interior_indices)
    )
    #    x,y,e=Gauss_zeidel(A,f[interior_indices])
    #    print(e)
    return scipy.sparse.linalg.spsolve(A, f[interior_indices])


# solve_helmholtz(M, interior_indices, f)


def extract_path_from_dir(dir):
    raw_names = next(os.walk(dir), (None, None, []))[2]
    return [dir + n for n in raw_names if n.endswith(".pt")]








def complex_version(v):
    assert v.size == 2
    r = np.sqrt(v[0] ** 2 + v[1] ** 2)
    theta = np.arctan2(v[1], v[0])
    return r*cmath.exp(1j*theta)




def save_figure(X, Y, titles, names, colors):

    # accuracy plots
    fig, ax = plt.subplots(1, len(X))
    for j in range(len(X)):
        ax[j].scatter(X[j],Y[j])

    plt.savefig(Constants.fig_path + "figures/" + ".eps",format='eps',bbox_inches='tight')
    plt.show(block=False)

 



def step_fourier(L,Theta):
    # fourier expansion of simple function in [0,1]
    # L is segments lengths
    # Theta is the angle function's values on the segments
    N=100
    x=[0]+[np.sum(L[:k+1]) for k in range(len(L))]
    a0=np.sum([l*theta for l,theta in zip(L,Theta)])
    a1=[math.sqrt(2)*np.sum([Theta[i]*(np.sin(2*math.pi*n*x[i+1])-np.sin(2*math.pi*n*x[i]))/(2*math.pi*n) 
                  for i in range(len(L))]) for n in range(1,N)]
    a2=[math.sqrt(2)*np.sum([Theta[i]*(-np.cos(2*math.pi*n*x[i+1])+np.cos(2*math.pi*n*x[i]))/(2*math.pi*n)
                   for i in range(len(L))]) for n in range(1,N)]
    coeff=[a0]
    for i in range(N-1):
        coeff.append(a1[i])
        coeff.append(a2[i])

    return np.array(coeff)

def save_uniqe(file, path):
    uniq_filename = (
            str(datetime.datetime.now().date())
            + "_"
            + str(datetime.datetime.now().time()).replace(":", ".")
        )
    torch.save(file, path+uniq_filename+'.pt') 

def save_eps(name):
        plt.savefig(Constants.tex_fig_path+name, format='eps',bbox_inches='tight')
        plt.show(block=False)


def plot_figures(ax,y, **kwargs):
    d=kwargs
    try:
        ax.plot(y, color=d['color'],  label=d['label'])
        ax.legend()
    except:
        ax.plot(y, color=d['color'])    
    try:
        ax.set_title(d['title'])
    except:
        pass    
    ax.set_xlabel(d['xlabel'])
    ax.set_ylabel(d['ylabel']) 
    try:
        ax.text(320, d['text_hight'], f'err={y[-1]:.2e}', c=d['color'])
    except:
        pass    
    
      
       
def closest(set1,p):
    temp=np.argmin(np.array([np.linalg.norm(x-p) for x in set1]))
    return set1[temp], temp


def fig1():
   
    fig,ax=plt.subplots(1)
    # fig.gca().set_aspect('equal', adjustable='box')   
    all_names=extract_path_from_dir(Constants.path+'polygons/')
    # test_names=[Constants.path+'hints_polygons/lshape.pt']
    test_names=[]
    train_names=all_names
  
    rect_names=[Constants.path+'polygons/00.pt']
    rect_names=[]

    # test_names=extract_path_from_dir(Constants.path+'hints_polygons/')
    x,y= poly(((0,0),(1,0),(1,1),(0,1))).exterior.xy
    # ax.plot(x,y,color='red', label='rect', alpha=0.4)

    [Plot_Polygon(torch.load(name)['generators'],ax=ax, color='black',alpha=0.7,linewidth=0.5, label='train') for name in train_names]
    [Plot_Polygon(torch.load(name)['generators'],ax=ax, color='blue',linewidth=1, linestyle='dashed', label='test') for name in test_names]
    
    # [ax[0].scatter(torch.load(name)['generators'][0,0], torch.load(name)['generators'][0,1], color='black',alpha=0.7,linewidth=0.5) for name in train_names[::]]
    # [ax[0].scatter(torch.load(name)['generators'][1,0], torch.load(name)['generators'][1,1], color='red',alpha=0.7,linewidth=0.5) for name in train_names[::]]
    ax.legend(loc='upper right', shadow=True)
    # [ax[0].scatter(list(range(len(torch.load(name)['angle_fourier']))),torch.load(name)['angle_fourier'], color='black') for name in train_names]
    

    fig1,ax1=plt.subplots(1)
    [ax1.plot(torch.load(name)['angle_fourier'][:20], color='black',alpha=0.7,linewidth=0.5, label='train') for name in train_names[::]]
    [ax1.plot(torch.load(name)['angle_fourier'][:20], color='blue',linewidth=0.5, linestyle='dashed', label='test') for name in test_names]
    # [ax1.plot(torch.load(name)['angle_fourier'][:10], color='red',linewidth=0.5, linestyle='dashed', label='rect') for name in rect_names]

    # [ax[0].scatter(torch.load(name)['generators'][0,0], torch.load(name)['generators'][0,1], color='black',alpha=0.7,linewidth=0.5) for name in train_names[::]]
    # [ax[0].scatter(torch.load(name)['generators'][1,0], torch.load(name)['generators'][1,1], color='red',alpha=0.7,linewidth=0.5) for name in train_names[::]]
    ax1.legend(loc='upper right', shadow=True)

     

    # # fig.savefig(Constants.eps_fig_path+'train_test.eps', format='eps', bbox_inches='tight')
    # plt.show(block=False)   


def fig2(test_names=[]):
   
    fig,ax=plt.subplots(2)
    train_names=extract_path_from_dir(Constants.path+'polygons/')

    [ax[0].plot(torch.load(name)['angle_fourier'], color='black',alpha=0.7,linewidth=0.5) for name in train_names[::]]
    # [ax[1].plot(torch.load(name)['angles']) for name in train_names[::]]
    # fig.gca().set_aspect('equal', adjustable='box')    

    fig.savefig(Constants.eps_fig_path+'train_test.eps', format='eps', bbox_inches='tight')
    plt.show(block=False)   

if __name__=='__main__':
    fig1()
    # print(np.arctan2(1/4,1/4)+math.pi)
    plt.show()


class Piecewise:
    def __init__(self, knts,vals):
        assert (len(knts)-len(vals))==1
        self.knts=knts
        self.vals=vals
    def __call__(self,x):
        cond=[((self.knts[i]<=x) & (x<self.knts[i+1])) for i in range(len(self.knts[:-2]))]
        cond.append(((self.knts[-2]<=x) & (x<=self.knts[-1])))
        return np.piecewise(x, cond, self.vals)


def shape_function(p1):

    x1=p1[:,0]
    y1=p1[:,1]
    dx1=[np.linalg.norm(np.array([y1[(k+1)%y1.shape[0]]-y1[k],x1[(k+1)%x1.shape[0]]-x1[k]])) for k in range(x1.shape[0])]
    theta1=[calc_angle([x1[(k+1)%x1.shape[0]]-x1[k],y1[(k+1)%y1.shape[0]]-y1[k]]) for k in range(x1.shape[0])]
    l1=[h/np.sum(dx1) for h in dx1]
    knts=[0]+[np.sum(l1[:k+1]) for k in range(len(l1))]
    f=Piecewise(knts, theta1)
    return f

def picewise(x,l,theta):
    if x<l[0]:
        return theta[0]
    if x>l[-1]:
        return theta[-1]
    if x<l[1]:
        return theta[1]
    return theta[2]
    
def sort_points(source,target):
    l=[]
    for t in iter(target):
        l.append(closest(source,t)[1])
    ind=np.argsort(l)    
    return  target[ind], ind
    

def plotter(generators, color='red'):
    x1=generators[:,0]
    y1=generators[:,1]
    dx=[np.linalg.norm(np.array([y1[(k+1)%y1.shape[0]]-y1[k],x1[(k+1)%x1.shape[0]]-x1[k]])) for k in range(x1.shape[0])]
    # theta=[np.arctan2(y1[(k+1)%y1.shape[0]]-y1[k],x1[(k+1)%x1.shape[0]]-x1[k]) for k in range(x1.shape[0])]
    theta=[calc_angle([x1[(k+1)%x1.shape[0]]-x1[k],y1[(k+1)%y1.shape[0]]-y1[k]]) for k in range(x1.shape[0])]  
    l=[h/np.sum(dx) for h in dx]
    
    intervals=np.array(l)
    angles=np.array(theta)

    knts=[0]+[np.sum(intervals[:k+1]) for k in range(len(intervals))]
    f=Piecewise(knts, angles)
    x=np.linspace(0,1,100)
    plt.plot(x,f(x),color=color)
    plt.show()
  
def isLeft(P0, P1, P2):
  return ((P1[0] - P0[0]) * (P2[1] - P0[1]) - (P2[0] - P0[0]) * (P1[1] - P0[1]))  

def wn_PnPoly(P, V):
  # P is tuple
  # V list of tuples 
  n = len(V)-1
  wn = 0;    # the  winding number counter
  i = 0

  # loop through all edges of the polygon
  while i<n:   # edge from V[i] to  V[i+1]
    if V[i][1] <= P[1]:         # start y <= P.y
      if V[i+1][1]  > P[1]:      # an upward crossing
        if isLeft(V[i], V[i+1], P) > 0:  # P left of  edge
          wn += 1            # have  a valid up intersect
    else:                        # start y > P.y (no test needed)
      if V[i+1][1] <= P[1]:     # a downward crossing
        if isLeft(V[i], V[i+1], P) < 0:  # P right of  edge
          wn -= 1            # have  a valid down intersect

    i += 1
  
  #print str(wn)
  return wn

def f_grid(domain,f):
        grid=[]
        N=20
        x=np.linspace(0,2,N)
        y=np.linspace(0,2,N)
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                grid.append(np.array([x[i],y[j]]))
        poly=[(point[0],point[1]) for point in domain['generators']]
        poly=poly+[poly[0]]
        func=interpolation_2D(domain['interior_points'][:,0],domain['interior_points'][:,1],f)
        
        s=np.array([ np.array(func(np.array([g[0]]),np.array([g[1]]))) for g in grid ]).squeeze()
        w=np.array([wn_PnPoly(g,poly) for g in grid ])
        return w*s
# p=np.array([[0,0], [0.8,0], [0.8,1/0.8], [0,1/0.8]])

# g=shape_function(p)
# x=np.linspace(0,1,100)

# plt.plot(x,(f(x)-g(x)))
# # plt.plot(x,g(x))
# plt.show()
# print(np.linalg.norm(f(x)-g(x)))

# count=0
# err=[]
# Lp=np.array([[0,0],[1,0],[1,1/2],[1/2,1/2],[1/2,3/2],[0,3/2]])
# f=shape_function(Lp)
# for i,a in enumerate(np.linspace(0.1,3,200)):
#     b=1/a

#     p=np.array([[0,0], [a,0], [a,b], [0,b]])
#     for j,theta in enumerate(iter(np.linspace(0,2*math.pi,205))):
#         S=np.array([[math.cos(theta), -math.sin(theta)],[math.sin(theta), math.cos(theta)]])
#         new_p=np.array([S@pt for pt in iter(p)])
#         g=shape_function(new_p)
#         x=np.linspace(0,1,100)
#         err.append(np.trapz((f(x)-g(x))**2,x))
#         count+=1

# print(np.min(err))
#
#  i=int(np.argmin(err)/100)
# j=np.argmin(err)%100
# # print(np.argmin(err))
# print(i)
# print(j)
# p=np.array([[0,0],[1,0],[1,0.1],[0.1,0.1],[0.1,1.1],[0,1.1]])
# domain=torch.load(Constants.path+'base_polygon/0.pt')
           