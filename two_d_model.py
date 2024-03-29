import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


import numpy as np
import os
import sys




class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=8,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(8, 16, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(16 * 8 * 8, 50)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output    # return x for visualization
# layer=CNN()
# y=torch.rand(15,1,32,32)
# layer(y)

class fc(torch.nn.Module):
    def __init__(self, input_shape, output_shape, num_layers, activation_func=torch.nn.Tanh(), activation_last=True):
        super().__init__()
        self.activation_last=activation_last
        self.input_shape = input_shape
        self.output_shape = output_shape

        layer_size = max(input_shape,80)

        # self.activation = torch.nn.SELU()
        self.activation = activation_func

        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(in_features=self.input_shape, out_features= layer_size, bias=True)])
        output_shape =  layer_size

        for j in range(num_layers):
            layer = torch.nn.Linear(
                in_features=output_shape, out_features= layer_size, bias=True)
            # initializer(layer.weight)
            output_shape =  layer_size
            self.layers.append(layer)

        self.layers.append(torch.nn.Linear(
            in_features=output_shape, out_features=self.output_shape, bias=True))

    def forward(self, y):
        s=y
        for layer in self.layers:
            s = layer(self.activation(s))
        if self .activation_last:
            return self.activation(s)
        else:
            return s


class Deeponet(nn.Module):
    # good parameters: n_layers in deeponet=4,n_layers in geo_deeponet=10, infcn=100, ,n=5*p, p=100

    def __init__(self, dim, f_dim):
        super().__init__()
        n_layers = 4
        branch_width=60
        trunk_width=60
        self.alpha = nn.Parameter(torch.tensor(0.))
        self.branch1 = fc(f_dim, branch_width, n_layers, activation_func=torch.nn.Tanh(),activation_last=False)
        self.trunk1 = fc(dim, trunk_width,  n_layers, activation_func=torch.nn.Tanh(), activation_last=True)
        self.c_layer = fc( branch_width, trunk_width, n_layers, activation_func=torch.nn.Tanh(), activation_last=False)
        self.c2_layer =fc( f_dim+dim, 1, n_layers, activation_func=torch.nn.Tanh(), activation_last=False) 


    def forward(self, X):
        y,f,translation,m_y, angle=X

        branch = self.c_layer(self.branch1(f))
        trunk = self.trunk1(y)
        alpha = torch.squeeze(self.c2_layer(torch.cat((f,y),dim=1)))
        return torch.sum(branch*trunk, dim=-1, keepdim=False)+alpha
        

class geo_deeponet2(nn.Module):
    # good parameters: n_layers in deeponet=4,n_layers in geo_deeponet=10, infcn=100, ,n=5*p, p=100

    def __init__(self, dim, f_dim, angle_dim):
        super().__init__()
        n_layers = 4
        branch_width=100
        trunk_width=100
        # self.n = p
        self.alpha = nn.Parameter(torch.tensor(0.))
        self.branch1 = fc(f_dim, branch_width, n_layers,activation_last=False)
        self.branch2= fc(angle_dim, branch_width, n_layers,activation_last=False)
        self.trunk1 = fc(dim, trunk_width,  n_layers, activation_last=True)
        self.cnn=CNN()
        self.mha=torch.nn.MultiheadAttention(embed_dim=branch_width, num_heads=10, bias=True, batch_first=True)

        self.c_layer = fc( 2*branch_width, trunk_width, n_layers, activation_last=False)
        # self.c2_layer =fc( angle_dim+translation_dim, 1, n_layers, False) 
        self.c2_layer =fc( f_dim+dim+angle_dim, 1, n_layers, activation_last=False) 



    def forward(self, X):
        y,f, angle=X
        angle=angle/(2*math.pi)

        # branch_temp=self.mha(self.branch1(f), self.branch2(angle), self.branch3(translation))[0]
        branch = self.c_layer( torch.cat((self.branch1(f), self.branch2(angle)), dim=1))
        # branch=self.c_layer(branch_temp)
        trunk = self.trunk1(y)
        # alpha=torch.squeeze(self.c2_layer(torch.cat((translation,angle),dim=1)),dim=1)
        alpha = torch.squeeze(self.c2_layer(torch.cat((f,y, angle),dim=1)))
        return torch.sum(branch*trunk, dim=-1, keepdim=False)+alpha
        

# torch.manual_seed(0)
# x=torch.rand(1,4)
# y=torch.rand(1,4)
# z=torch.rand(1,4)
# y=torch.rand(1,4)
# # # b=torch.cat((x,x), dim=0)
# # b=x
# L=torch.nn.MultiheadAttention(embed_dim=4, num_heads=2, bias=True)
# # print(x)
# print(L(x,y,z)[0].shape)













