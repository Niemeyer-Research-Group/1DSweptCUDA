# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 15:36:46 2016

@author: osumageed
"""

from math import *
import matplotlib.pyplot as plt
import numpy as np

def KS_disc(uL,uR,u,dx,dt):
    
    u0 = u[0]
    
    u[1] = (uL[0] + uR[0] - 2.0*u[0])/(dx**2)
    cv = (uL[0]**2 - uR[0]**2)/(4.0*dx)
    diffs = (sum(uL)+sum(uR) - 2.0*sum(u))/(dx**2)
    print cv, diffs
    u[0] = u[0] - 0.5 * dt * (diffs + cv)
    
    
    
    u[1] = (uL[0] + uR[0] - 2.0*u[0])/(dx**2)
    cv = (uL[0]**2 - uR[0]**2)/(4.0*dx)
    diffs = (sum(uL)+sum(uR) - 2.0*sum(u))/(dx**2)
    u[0] = u0 - 0.5 * dt * (diffs + cv)
    
    return u

n= 2048
k = np.arange(n)
lx = 50.0

x = lx/float(n)*k-lx/2.0 

u1 = 2.0*np.cos(19.0*x*pi/128.0) 
u2 = -361.0/8192.0*np.cos(19.0*x[k]*pi/128.0)

ulist = np.empty([2,n])
for k in range(n):
    ulist[0,k] = u1[k]
    ulist[1,k] = u2[k]
    
plt.subplot(211)
plt.plot(x,ulist[0,:])
plt.hold()  
plt.subplot(212)
plt.plot(x,ulist[1,:])
plt.hold()
plt.show()

dx = x[1]-x[0]
dt = .005
ut = ulist

for k in range(20):
    
    ut[:,0] = KS_disc(ulist[:,n-1],ulist[:,1],ulist[:,0],dx,dt)

    ut[:,1:n-2] = KS_disc(ulist[:,0:n-3],ulist[:,2:n-1],ulist[:,1:n-2],dx,dt)
        
    ut[:,n-1] = KS_disc(ulist[:,n-2],ulist[:,0],ulist[:,n-1],dx,dt)
    
    ulist = ut

    plt.subplot(211)
    plt.plot(x,ulist[0,:])
    plt.subplot(212)
    plt.plot(x,ulist[0,:])
    plt.draw()
    g = int(raw_input("Enter 0 to break: "))
    if g == 0:
        break



