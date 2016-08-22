
import numpy as np
import subprocess as sp
import shlex
import os
import matplotlib.pyplot as plt
from sympy import *
##from exactpack.solvers.riemann import Sod
##
##def euler_exact():
##    r = np.linspace(0.0,1.0,1025)
##    print r[1]-r[0]
##    t = .25
##
##    solver = Sod()
##    soln = solver(r,t)
##
##    soln.plot('density')
##    plt.show()

#All numeric first shot.
def heat_exact(x,t,L,n,c0,cn):
    rt = 1e-4
    at = 1e-5
    alpha = 8.418e-5;    
    Lhs1 = cn[0]*np.exp(-(np.pi*alpha/L)**2*t)*np.cos(x*np.pi/L)

    for k in range(2,50):
        cnt = float(k)
        
        Lhs1 +=  cn[k]*np.exp(-(cnt*np.pi*alpha/L)**2*t)*np.cos(cnt*x*np.pi/L)

    Tf1 = c0 + Lhs1

    return Tf1

if __name__ == '__main__':
    nm = 1024
    Lm = .5
    tm = np.linspace(2.0,500.0,6)
    xm = np.linspace(0.0,Lm,nm)
    
    lbl = []
    x = symbols('x')
    #L = symbols('L', positive = True)
    n = symbols('n', positive = True, integer = True)    
    fx = 100*x
    fourier = cos(n*x*pi/Lm)
    c0i = 1.0/Lm * integrate(fx,(x,0,Lm))
    cNi = 2.0/Lm * integrate(fx*fourier,(x,0,Lm))
    print c0i, cNi
    cn = []
    c0 = c0i.evalf()
    for k in range(1,51):
        cn.append(cNi.evalf(subs = {n: k}))

    for k in tm:
        ar = []
        for g in xm:
            ar.append(heat_exact(g,k,Lm,nm,c0,cn))

        plt.plot(xm,ar)
        plt.hold(True)
        lbl.append(str(k))

plt.legend(lbl)
plt.show()
