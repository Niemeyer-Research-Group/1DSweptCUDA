
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


##50*(x-L/2) + 50
def heats(n,L,x,t):
    alpha = 8.418e-5
    return 1.0/n**2 * np.exp(-alpha*t*(n*np.pi/L)**2) * np.cos(n*x*np.pi/L)


def heat_exact(t,L,divs):

    xm = np.linspace(0,L,divs)
    c0 = 50.0*L/3.0
    cout = 400.0*L/(np.pi**2)
    Tf1 = []
    for xr in xm:
        c  = 2
        ser = heats(c,L,xr,t)
        h = np.copy(ser)
        for k in range(100):
            c += 2
            ser = heats(c,L,xr,t)
            h += ser

        Tf1.append(c0 - cout * h)

    return Tf1

def KS_exact(x,x0,t,k,c):
    u = c + (5.0/19.0)*np.sqrt(11.0/19.0)*(11.0*np.tanh(k*(x-c*t-x0))**3 - 9.0*np.tanh(k*(x-c*t-x0)))
    return u

if __name__ == '__main__':

    if len(sys.argv) < 2:
        sys.exit(-1)

    #HEAT Exact
    if int(sys.argv[1]) == 0:
        nm = 1024
        Lm = 1.0
        tm = np.linspace(2.0,500.0,6)
        xm = np.linspace(0.0,Lm,nm)
        ds = xm[1]-xm[0]

        lbl = []
        x = symbols('x')
        #L = symbols('L', positive = True)
        n = symbols('n', positive = True, integer = True)
        fx = 50.0*sin(np.pi*x*4.0)
        fourier = cos(n*x*pi/Lm)
        c0i = 1.0/Lm * integrate(fx,(x,0,Lm))
        cNi = 2.0/Lm * integrate(fx*fourier,(x,0,Lm))
        print c0i, cNi
        cn = []
        c0 = c0i.evalf()
        for k in range(100):
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
            if k == tm[1]:
                gr = open('Temp.txt','a+')
                for a in ar:
                    gr.write(str(a) + "  ")

                gr.close()
                break

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


        plt.legend(lbl)
        plt.show()
