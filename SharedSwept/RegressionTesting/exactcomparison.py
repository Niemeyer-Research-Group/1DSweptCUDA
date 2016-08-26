#This should be stand alone
import numpy as np
import subprocess as sp
import shlex
import os
import matplotlib.pyplot as plt
import sys
from sympy import *
import pandas as pd

from exactpack.solvers.riemann import Sod

def euler_exact(t):
   r = np.linspace(0.0,1.0,513)
   print r[1]-r[0]

   solver = Sod()
   soln = solver(r,t)

   print soln.density.size

   soln.plot('density')
   plt.show()


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

def rmse(exact,sim):
    return np.sqrt(np.mean((exact-sim)**2))

if __name__ == '__main__':

    exactpath = os.path.abspath(os.path.dirname(__file__))
    mainpath = os.path.dirname(exactpath)

    proc = sp.Popen("make", cwd = mainpath)
    sp.Popen.wait(proc)

    rsltpath = os.path.join(mainpath,'Results')
    binpath = os.path.join(mainpath,'bin')

    #Heat first
    Fname = ["Heat","Euler"]
    Varfile = os.path.join(rsltpath, Fname[0] + "1D_Result.dat")
    execut = os.path.join(binpath, Fname[0] + "Out")

    alpha = 8.418e-5
    div = 2048.0
    dx = 1/div
    bks = 256
    L = 1.0
    tf = 200
    freq = 30
    cpu = 0
    swept = 0
    dt = [.0015,.001,.0005,.0001]
    Fo = [alpha*k/dx**2 for k in dt]
    err = np.empty()

    for i,dts in enumerate(dt):
        execstr = execut +  ' {0} {1} {2} {3} {4} {5} {6} {7}'.format(div,bks,dts,tf,freq,swept,cpu,Varfile)
        exeStr = shlex.split(execstr)
        proc = sp.Popen(exeStr)
        sp.Popen.wait(proc)

        # Now read in the files and plot, need better matplotlib strategy.
        fin = open(Varfile)
        data = []
        time = []
        h = []
        xax = np.linspace(0,L,div)

        for i,line in enumerate(fin):
            if i>1:

                ar = [float(n) for n in line.split()]
                data.append(ar[1:])
                time.append(ar[0])
                h = heat_exact(ar[0],L,div)

        for k in range(len(h)):
            err[i,k] = rmse(h[k],data[k])



    euler_exact()
