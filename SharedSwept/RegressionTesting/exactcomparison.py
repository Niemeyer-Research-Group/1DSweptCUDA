
import numpy as np
import subprocess as sp
import shlex
import os
import matplotlib.pyplot as plt
import sys
from sympy import *

from exactpack.solvers.riemann import Sod

def euler_exact():
   r = np.linspace(0.0,1.0,1025)
   print r[1]-r[0]
   t = .25

   solver = Sod()
   soln = solver(r,t)

   soln.plot('density')
   plt.show()


##500.f*expf((-ds*(REAL)xnode)/lx ) + 50.f*sinf(-ds*128.f*(REAL)xnode/lx);
#
def heat_exact(x,t,L,n,c0,cn):
    rt = 1e-4
    at = 1e-5
    alpha = 8.418e-5;
    Lhs1 = cn[0]*np.exp(-(np.pi/L)**2*(alpha*t))*np.cos(x*np.pi/L)

    for k in range(2,50):
        cnt = float(k)

        Lhs1 +=  cn[k]*np.exp(-(cnt*np.pi/L)**2*(alpha*t))*np.cos(cnt*x*np.pi/L)

    Tf1 = c0 + Lhs1

    return Tf1

def KS_exact(x,x0,t,k,c):
    u = c + (5.0/19.0)*np.sqrt(11.0/19.0)*(11.0*np.tanh(k*(x-c*t-x0))**3 - 9.0*np.tanh(k*(x-c*t-x0)))
    return u

if __name__ == '__main__':

    print len(sys.argv)

    if len(sys.argv) < 2:
        sys.exit(-1)

    print sys.argv[1]
    #HEAT Exact
    if int(sys.argv[1]) == 0:
        nm = 2048
        Lm = 16.0
        tm = np.linspace(2.0,500.0,6)
        xm = np.linspace(0.0,Lm,nm)
        ds = xm[1]-xm[0]

        lbl = []
        x = symbols('x')
        #L = symbols('L', positive = True)
        n = symbols('n', positive = True, integer = True)
        fx = 500.0 - 250.0*exp(-x/Lm)
        fourier = cos(n*x*pi/Lm)
        c0i = 1.0/Lm * integrate(fx,(x,0,Lm))
        cNi = 2.0/Lm * integrate(fx*fourier,(x,0,Lm))
        print c0i, cNi
        cn = []
        c0 = c0i.evalf()
        for k in range(1,51):
            cn.append(cNi.evalf(subs = {n: k}))

        init = [fx.evalf(subs = {x : tmp}) for tmp in xm]
        plt.plot(xm,init)
        plt.hold(True)
        lbl = ["Initial"]
        for k in tm:
            ar = []
            for g in xm:
                ar.append(heat_exact(g,k,Lm,nm,c0,cn))

            plt.plot(xm,ar)
            plt.hold(True)
            lbl.append(str(k))

        plt.legend(lbl)
        plt.show()

    #KS exact
    else:
        c0 = 0.1
        k0 = 1.0/4.0*np.sqrt(11.0/19.0)
        x0 = -30.0
        nm = 3
        Lm = 1.0
        tm = np.linspace(0.0,1.0,3)
        xm = np.linspace(0.0,Lm,nm)
        lbl = []
        for t in tm:
            ar = []
            for g in xm:
                ar.append(KS_exact(g,x0,t,k0,c0))

            plt.plot(xm,ar)
            plt.hold(True)
            lbl.append(str(t))
            print t, ar

        plt.legend(lbl)
        plt.show()
