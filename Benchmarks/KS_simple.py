# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 15:36:46 2016

@author: osumageed
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys

def KS_doubleprime(uL,uR,u,dx):
    
    u1 = (uL + uR - 2.0*u)/(dx**2)
    
    return u1
    
def KS_disc(uL,uR,u,uRxx,uLxx,uxx,dx,dt):
    
    cv = (uR**2 - uL**2)/(4.0*dx)
    
    diffs = ((uL+uLxx) + (uR+uRxx) - 2.0*(u+uxx))/(dx**2)
    ui = u - dt * (diffs + cv)
    
    return ui

n = 2048
dx = .5
dt = .005

x = np.arange(0,dx*float(n),dx)

u = 2.0*np.cos(19.0*x*np.pi/128.0) 

uxx = -361.0*np.pi/4096.0*np.cos(19.0*x*np.pi/128.0) 

u1 = u[0]
u2 = u[-1]

u = np.append(u,u1)

u = np.insert(u,0,u2)

uxx1 = uxx[0]
uxx2 = uxx[-1]

uxx = np.append(uxx,uxx1)

uxx = np.insert(uxx,0,uxx2)
    
plt.subplot(211)
plt.plot(x,u[1:-1])
plt.title('U')
plt.hold(True)  
plt.subplot(212)
plt.plot(x,uxx[1:-1])
plt.title('Uxx')
plt.hold(True)
plt.show()

tend = 2048
tn = np.arange(0,tend*dt,dt)

uF = np.copy(u)

for b in range(1,tend):
    ut = np.copy(u)
    
    for k in range(1,n+1):
        
        u[k] = KS_disc(ut[k-1],ut[k+1],ut[k],uxx[k-1],uxx[k+1],uxx[k],dx,dt)
        
    u[-1] = u[1]
    u[0] = u[-2]
    uF = np.vstack([uF,u])
        
    for k in range(1,n+1):  
        uxx[k] = KS_doubleprime(u[k-1],u[k+1],u[k],dx)
        

    uxx[-1] = uxx[1]
    uxx[0] = uxx[-2]    
    
    
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
xmn,tmn = np.meshgrid(x,tn)
ax.plot_surface(xmn,tmn,uF[:,1:-1],linewidth=0)
plt.show()



