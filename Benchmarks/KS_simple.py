# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 15:36:46 2016

@author: osumageed
"""

from math import *
import matplotlib.pyplot as plt

def KS_disc(uL,uR,u,dx,dt):
    
    u0 = u[0]
    
    u[1] = (uL[0] + uR[0] - 2.0*u[0])/(dx**2)
    cv = (uL[0]**2 - uR[0]**2)/(4.0*dx)
    diffs = (sum(uL)+sum(uR) - 2*sum(u))/(dx**2)
    u[0] = u[0] - 0.5 * dt * (diffs + cv)
    
    
    u[1] = (uL[0] + uR[0] - 2.0*u[0])/(dx**2)
    cv = (uL[0]**2 - uR[0]**2)/(4.0*dx)
    diffs = (sum(uL)+sum(uR) - 2*sum(u))/(dx**2)
    u[0] = u0 - 0.5 * dt * (diffs + cv)
    
    return u

n = 2048
lx = 50.0
x = [lx/float(n)*float(k)-lx/2.0 for k in range(n)]
u1 = [2.0*cos(19.0*x[k]*pi/128.0) for k in range(n)]
u2 = [-361.0/8192.0*cos(19.0*x[k]*pi/128.0) for k in range(n)]

ulist = list()
for k in range(n):
    ulist.append([u1[k],u2[k]])
    
plt.subplot(211)
plt.plot(x,u1)
plt.hold()  
plt.subplot(212)
plt.plot(x,u2)
plt.hold()

dx = x[1]
dt = .005
ut = ulist
for k in range(20):
    ut[0][:] = KS_disc(ulist[n-1],ulist[1],ulist[0],dx,dt)
    for nm in range(1,n-1):
        ut[nm] = KS_disc(ulist[nm-1],ulist[nm+1],ulist[nm],dx,dt)
        
    ut[n-1][:] = KS_disc(ulist[n-2],ulist[0],ulist[n-1],dx,dt)
    
    ulist = ut
    plt.subplot(211)
    plt.plot(x,ulist[:][0])
    plt.subplot(212)
    plt.plot(x,ulist[:][1])
    g = int(raw_input("Enter 0 to break: "))
    if g == 0:
        break



