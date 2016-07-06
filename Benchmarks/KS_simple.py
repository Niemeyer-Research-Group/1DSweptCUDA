# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 15:36:46 2016

@author: osumageed
"""
import matplotlib.pyplot as plt
import numpy as np

def KS_doubleprime(uL,uR,u,dx,dt):
    
    u1 = (uL[0] + uR[0] - 2.0*u[0])/(dx**2)
    
    return u1
    
def KS_disc(uL,uR,u,dx,dt):
    cv = (uL[0]**2 - uR[0]**2)/(4.0*dx)
    diffs = (np.sum(uL)+np.sum(uR) - 2.0*np.sum(u))/(dx**2)
    ui = u[0] - dt * (diffs + cv)
    
    return ui

n= 2048
k = np.arange(n)
lx = 128.0/19.0*2.0

x = np.linspace(-lx/2.0,lx/2.0,n)

x = x[:-1]
n = n - 1

u1 = 2.0*np.cos(19.0*x*np.pi/128.0) 

ulist = np.empty([2,n])

ulist[0,:] = u1
ulist[1,:] = np.zeros([1,n])
    
plt.subplot(211)
plt.plot(x,ulist[0,:])
plt.hold()  
plt.subplot(212)
plt.plot(x,ulist[1,:])
plt.hold()
plt.show()

dx = x[1]-x[0]
dt = .005
ut = np.copy(ulist)

for k in range(20):
    
    ut[1,0] = KS_doubleprime(ulist[:,n-1],ulist[:,1],ulist[:,0],dx,dt)

    ut[1,1:n-2] = KS_doubleprime(ulist[:,0:n-3],ulist[:,2:n-1],ulist[:,1:n-2],dx,dt)
        
    ut[1,n-1] = KS_doubleprime(ulist[:,n-2],ulist[:,0],ulist[:,n-1],dx,dt)
    
    ut[0,0] = KS_disc(ulist[:,n-1],ulist[:,1],ulist[:,0],dx,dt)

    ut[0,1:n-2] = KS_disc(ulist[:,0:n-3],ulist[:,2:n-1],ulist[:,1:n-2],dx,dt)
        
    ut[0,n-1] = KS_disc(ulist[:,n-2],ulist[:,0],ulist[:,n-1],dx,dt)
    
    ulist = np.copy(ut)

    plt.subplot(211)
    plt.title(str(dt*(k+1)))
    plt.plot(x,ulist[0,:])
    plt.subplot(212)
    plt.plot(x,ulist[1,:])
    plt.show()
    g = int(raw_input("Enter 0 to break: "))
    if g == 0:
        break



