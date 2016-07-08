# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 15:36:46 2016

@author: osumageed
"""
import matplotlib.pyplot as plt
import numpy as np
import sys

def KS_doubleprime(uL,uR,u,dx):
    
    u1 = (uL + uR - 2.0*u)/(dx**2)
    
    return u1
    
def KS_disc(uL,uR,u,uRxx,uLxx,uxx,dx,dt):
    cv = (uL**2 - uR**2)/(4.0*dx)
    diffs = ((uL+uLxx)+(uR+uRxx) - 2.0*(u+uxx))/(dx**2)
    ui = u - dt * (diffs + cv)
    
    return ui

n= 8196
k = np.arange(n+4)
lx = 128.0/19.0*12.0

x = np.linspace(-lx/2.0,lx/2.0,n)

u = 2.0*np.cos(19.0*x*np.pi/128.0) 

u1 = u[0]
u2 = u[-1]

u = np.append(u,u1)

u = np.insert(u,0,u2)

uxx = np.empty(n+2)
dx = x[1]-x[0]
dt = .005

for k in range(1,n+1):
    uxx[k] = KS_doubleprime(u[k-1],u[k+1],u[k],dx)

uxx[-1] = uxx[1]
uxx[0] = uxx[-2]


    
plt.subplot(211)
plt.plot(x,u[1:-1])
plt.title('U')
plt.hold()  
plt.subplot(212)
plt.plot(x,uxx[1:-1])
plt.title('Uxx')
plt.hold()
plt.show()

g = int(raw_input("Enter 0 to break: "))
if g == 0:
   sys.exit(0)


for b in range(20):
    ut = np.copy(u)
    
    for k in range(1,n+1):
        
        u[k] = KS_disc(ut[k-1],ut[k+1],ut[k],uxx[k-1],uxx[k+1],uxx[k],dx,dt)\
        
    u[-1] = u[1]
    u[0] = u[-2]
        
    for k in range(1,n+1):  
        uxx[k] = KS_doubleprime(u[k-1],u[k+1],u[k],dx)
        


    uxx[-1] = uxx[1]
    uxx[0] = uxx[-2]
    

    plt.subplot(211)
    plt.plot(x,u[1:-1])
    plt.title('U')
    plt.hold()  
    plt.subplot(212)
    plt.plot(x,uxx[1:-1])
    plt.title('Uxx')
    plt.hold()
    plt.show()
    g = int(raw_input("Enter 0 to break: "))
    if g == 0:
        break



