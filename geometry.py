import datetime
import os
import sys
import time

from shapely.geometry import Polygon as Pol2
from pylab import figure
import numpy as np
import matplotlib.pyplot as plt
import scipy
import dmsh
import meshio
import optimesh
from pathlib import Path
import torch
from sklearn.cluster import KMeans
from scipy.interpolate import Rbf
from scipy.spatial.distance import euclidean, cityblock
import cmath

from utils import *
from constants import Constants
# from coords import Map_circle_to_polygon
from pydec.dec import simplicial_complex
from functions.functions import Test_function

from packages.my_packages import *







    

    
class mesh:
    ''''
        points=np.random.rand(5,2)
        points=[points[i] for i in range(5)]
        mesh(points)
    '''
    def __init__(self, points):
        #  points (50,2) as list of arrays[[],[],[]..]

        self.points=points.copy()
        self.p=[]
        for i in range(len(self.points)):
            l=self.points.copy()
            point=l[i].reshape(1,2)
            l.pop(i)
            nbhd=np.array(l)
            #  50,2
            X=np.vstack((point,nbhd))
            values=np.hstack((1,np.zeros(nbhd.shape[0])))
            self.p.append(interpolation_2D(X[:,0],X[:,1,], values))
            # self.p.append(scipy.interpolate.RBFInterpolator(X,values, function='gaussian'))
            self.T=1000

class Polygon:
    def __init__(self, generators, S=None, base_polygon=None):
        self.T=-100
        self.generators = generators
        self.grid=[]
        x=np.linspace(np.min([self.generators[:,0]]),np.max([self.generators[:,0]]),40)
        y=np.linspace(np.min([self.generators[:,1]]),np.max([self.generators[:,1]]),40)
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                self.grid.append(np.array([x[i],y[j]]))
        self.n=self.generators.shape[0]
        self.geo = dmsh.Polygon(self.generators)
        self.fourier_coeff, self.lengths, self.angles = self.fourier()
        self.S=S
        self.base_polygon=base_polygon

    def create_mesh(self, h):

        # if np.min(calc_min_angle(self.geo)) > (math.pi / 20):
        start=time.time()

        try:
            self.X=np.array([self.S@p for p in iter(self.base_polygon['X'])])
            self.cells=self.base_polygon['cells']
            self.interior_indices=self.base_polygon['interior_indices']
            self.interior_points=np.array([self.S@p for p in iter(self.base_polygon['interior_points'])])
            self.hot_indices=self.base_polygon['hot_indices']
            self.hot_points=np.array([self.S@p for p in iter(self.base_polygon['hot_points'])])            



        except:
            X, cells = dmsh.generate(self.geo, h)

            self.X, self.cells = optimesh.optimize_points_cells(
                X, cells, "CVT (full)", 1.0e-8, 200
            )

            print(f'triangulation generated with time= {time.time()-start}')  


            self.interior_points=[]
            self.interior_indices=[]
            for j,x in enumerate(self.X):
                if (not on_boundary(x,self.geo)):
                    self.interior_points.append(x)
                    self.interior_indices.append(j)
            self.interior_points=np.array(self.interior_points)

        self.hot_points=sort_points(self.grid,self.interior_points)[0][:75]
        self.hot_indices=sort_points(self.grid,self.interior_points)[1][:75]
        
        # ind=np.lexsort((self.interior_points[:,0],self.interior_points[:,1]))
        # self.hot_points=self.interior_points[ind]
        # # self.hot_points=self.interior_points
        # self.hot_indices=ind




        self.sc = simplicial_complex(self.X, self.cells)
        self.M = (
            (self.sc[0].star_inv)
            @ (-(self.sc[0].d).T)
            @ (self.sc[1].star)
            @ self.sc[0].d
        )


        print(f'geometry generated with time= {time.time()-start}')
        # self.radial_functions=self.radial_basis()
        self.radial_functions=None
     
    
    def laplacian(self):
        ev,V=scipy.sparse.linalg.eigs(-self.M[self.interior_indices][:, self.interior_indices],k=10,
            return_eigenvectors=True,
            which="SR",
        )
        return ev,V
    # def angle_function(self):
    #     x1=self.generators[:,0]
    #     y1=self.generators[:,1]
    #     dx=[np.linalg.norm(np.array([y1[(k+1)%y1.shape[0]]-y1[k],x1[(k+1)%x1.shape[0]]-x1[k]])) for k in range(x1.shape[0])]
    #     # theta=[np.arctan2(y1[(k+1)%y1.shape[0]]-y1[k],x1[(k+1)%x1.shape[0]]-x1[k]) for k in range(x1.shape[0])]
    #     theta=[calc_angle([x1[(k+1)%x1.shape[0]]-x1[k],y1[(k+1)%y1.shape[0]]-y1[k]]) for k in range(x1.shape[0])]  
    #     l=[h/np.sum(dx) for h in dx]
        
    #     intervals=np.array(l)
    #     angles=np.array(theta)

    #     knts=[0]+[np.sum(intervals[:k+1]) for k in range(len(intervals))]
    #     f=Piecewise(knts, angles)
    #     x=np.linspace(0,1,100)
    #     return f(x)
    def angle_function(self):
        grid=[]
        N=20
        x=np.linspace(0,2,N)
        y=np.linspace(0,2,N)
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                grid.append(np.array([x[i],y[j]]))
        poly=[(point[0],point[1]) for point in self.generators]
        poly=poly+[poly[0]]
        # return np.array([wn_PnPoly(g,poly) for g in grid ]).reshape((N, N))
        return np.array([wn_PnPoly(g,poly) for g in grid ])
    
    def find_hot_points(self):
        try:
            pts= np.array([self.S@p for p in self.base_polygon['hot_points']])
            pts=pts[np.lexsort((pts[:,0],pts[:,1]))]
        except:
            x,y=np.meshgrid(np.linspace(np.min(self.X[:,0]),np.max(self.X[:,0]),9)[1:-1], np.linspace(np.min(self.X[:,1]),np.max(self.X[:,1]),9)[1:-1])
            pts=np.array([[i,j] for i,j in zip(x.ravel(),y.ravel())])
            pts=pts[np.lexsort((pts[:,0],pts[:,1]))]
        return np.array([closest(self.interior_points,pts[i])[0] for i in range(pts.shape[0]) ])
        
    def find_hot_indices(self):

        try:
            pts= np.array([self.S@p for p in self.base_polygon['hot_points']])
            pts=pts[np.lexsort((pts[:,0],pts[:,1]))]
        except:
            x,y=np.meshgrid(np.linspace(np.min(self.X[:,0]),np.max(self.X[:,0]),9)[1:-1], np.linspace(np.min(self.X[:,1]),np.max(self.X[:,1]),9)[1:-1])
            pts=np.array([[i,j] for i,j in zip(x.ravel(),y.ravel())])
            pts=pts[np.lexsort((pts[:,0],pts[:,1]))]
        return np.array([closest(self.interior_points,pts[i])[1] for i in range(pts.shape[0]) ])
        
    def is_legit(self):
        if np.min(abs(self.sc[1].star.diagonal())) > 0:
            return True
        else:
            return False

    def save(self, path):
        # try:    
            assert self.is_legit()
            data = {
             
               
                "interior_points": self.interior_points,
                "interior_indices": self.interior_indices,
                "hot_points": self.hot_points,
                "hot_indices":self.hot_indices,
                # "hot_points": self.hot_points[np.lexsort((self.hot_points[:,1], self.hot_points[:,0]))],
                "generators": self.generators,
                "M": self.M[self.interior_indices][:, self.interior_indices],
                'radial_basis':self.radial_functions,
                'angle_fourier':self.fourier_coeff,
                'angle_function':self.angle_function(),
                'angles':self.angles,
                'translation':self.T,
                'cells':self.cells,
                'X':self.X,
                'geo':self.geo,
                "legit": True,
                'type': 'polygon'
            }
            torch.save(data, path)
        # except:
        #     data={"generators":self.generators,
        #            'angle_fourier':self.fourier_coeff,
        #            'lengths':self.lengths,
        #            'angles':self.angles}
        #     torch.save(data, path)

    def plot2(self):
        plt.scatter(self.interior_points[:, 0],
                    self.interior_points[:, 1], color='black')
        plt.show()

    def radial_basis(self):
        m=mesh([self.vertices[i] for i in range(self.vertices.shape[0])])
        return [m.p[i] for i in self.interior_indices]
    
    
    def fourier(self):
        x1=self.generators[:,0]
        y1=self.generators[:,1]
        dx=[np.linalg.norm(np.array([y1[(k+1)%y1.shape[0]]-y1[k],x1[(k+1)%x1.shape[0]]-x1[k]])) for k in range(x1.shape[0])]

        # theta=[np.arctan2(y1[(k+1)%y1.shape[0]]-y1[k],x1[(k+1)%x1.shape[0]]-x1[k]) for k in range(x1.shape[0])]
        theta=[calc_angle([x1[(k+1)%x1.shape[0]]-x1[k],y1[(k+1)%y1.shape[0]]-y1[k]],[1,0]) for k in range(x1.shape[0])]

        l=[h/np.sum(dx) for h in dx]
 
        coeff=step_fourier(l,theta)
        return coeff, l, theta
    
    @classmethod
    def plot(cls,generators, ax=None, title='no title was given'):
        assert generators.shape[1]==2
        x1=generators[:,0]
        y1=generators[:,1]
        polygon = Pol2(shell=[[x1[k],y1[k]] for k in range(x1.shape[0])],holes=None)
        try:        
            ax.set_title(title)
            plot_polygon(ax, polygon, facecolor='white', edgecolor='red')
        except:
            fig, ax = plt.subplots()    
            plot_polygon(ax, polygon, facecolor='white', edgecolor='red')
        


        
    @classmethod
    def plot_geo(cls,domain):
        
        fig, ax = plt.subplots()
        for i,p in enumerate(iter(domain['hot_points'])):
            ax.scatter(p[0],p[1])
            ax.annotate(str(i), (p[0], p[1]))
        
        # dmsh.show(domain['X'], domain['cells'], domain['geo'])
        plt.show()




        


# geo= dmsh.Rectangle(-1, +1, -1, +1)- dmsh.Rectangle(-0.5, 0.5, -0.5, 0.5)

# X, cells = dmsh.generate(geo, 0.2)
# boundary=[]
# ind=[]
# for i,x in enumerate(X):
#     if on_boundary(x,geo):
#         boundary.append(x)
#         ind.append(i)
# plt.scatter(X[ind,0], X[ind,1]);plt.show()
# dmsh.show(X, cells, geo)


class Annulus(Polygon):
    def __init__(self, generators,T):
        self.generators = generators
        self.n=self.generators.shape[0]
        self.geo =dmsh.Rectangle(0, 1, 0, 1)- dmsh.Polygon(self.generators)
        self.fourier_coeff, self.lengths, self.angles = self.fourier()
        self.T=T












 