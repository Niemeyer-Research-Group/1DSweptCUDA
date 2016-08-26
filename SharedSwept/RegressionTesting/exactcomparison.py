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

def euler_exact(t,dv):
   r = np.linspace(0.0,1.0,dv)

   solver = Sod()
   soln = solver(r,t)

   return soln.density


def heats(n,L,x,t):
    alpha = 8.418e-5
    return 1.0/n**2 * np.exp(-alpha*t*(n*np.pi/L)**2) * np.cos(n*x*np.pi/L)


def heat_exact(t,L,divs):

    xm = np.linspace(0,L,divs)
    c0 = 50.0*L/3.0
    cout = 400.0*L/(np.pi**2)
    Tf1 = np.empty(int(div))
    for i,xr in enumerate(xm):
        c  = 2
        ser = heats(c,L,xr,t)
        h = np.copy(ser)
        for k in range(1):
            c += 2
            ser = heats(c,L,xr,t)
            h += ser

        Tf1[i] = (c0 - cout * h)

    return Tf1

def rmse(exact,sim):
    return np.sqrt(np.mean((np.array(exact)-np.array(sim))**2))

if __name__ == '__main__':

    w = int(sys.argv[1])

    exactpath = os.path.abspath(os.path.dirname(__file__))
    mainpath = os.path.dirname(exactpath)

    proc = sp.Popen("make", cwd = mainpath)
    sp.Popen.wait(proc)

    rsltpath = os.path.join(mainpath,'Results')
    binpath = os.path.join(mainpath,'bin')

    #Heat first
    Fname = ["Heat","Euler"]
    Varfile = os.path.join(rsltpath, Fname[w] + "1D_Result.dat")
    execut = os.path.join(binpath, Fname[w] + "Out")

    SCHEME = [
        "Classic",
        "SweptGPU",
        "SweptCPUshare"
    ]

    alpha = 8.418e-5
    div = 1024.0
    bks = 64
    dx = .001

    if w == 0:

        L = dx * (div-1)
        print L
        tf = 200
        freq = 60
        cpu = 0
        swept = 0
        dt = [.0015,.0010,.0005,.0001]
        Fo = [alpha*k/dx**2 for k in dt]
        err = np.empty([4,4])
        lbl = []
        lbl2 = []
        tm = np.copy(err)

        if swept and cpu:
            sch = SCHEME[2]
            timestr = Fname[w] + " " + sch
        elif swept:
            sch = SCHEME[1]
            timestr = Fname[w] + " " + sch
        else:
            sch = SCHEME[0]
            timestr = Fname[w] + " " + sch

        for i,dts in enumerate(dt):
            execstr = execut +  ' {0} {1} {2} {3} {4} {5} {6} {7}'.format(div,bks,dts,tf,freq,swept,cpu,Varfile)
            exeStr = shlex.split(execstr)
            proc = sp.Popen(exeStr)
            sp.Popen.wait(proc)
            lbl.append(str(dts) + " " + str(Fo[i]))

            # Now read in the files and plot, need better matplotlib strategy.
            fin = open(Varfile)
            data = []
            time = []
            h = []
            xax = np.linspace(0,L,div)

            for p,line in enumerate(fin):

                if p>1:

                    ar = [float(n) for n in line.split()]
                    data.append(ar[1:])
                    time.append(ar[0])
                    h.append(heat_exact(ar[0],L,div))

            print len(h), len(h[0])
            for k in range(len(h)):
                err[i,k] = rmse(h[k],data[k])
                tm[i,k] = time[k]


        for k in range(4):
            plt.subplot(121)
            plt.plot(tm[k,:],err[k,:],'-o')
            plt.hold(True)
            plt.subplot(122)
            plt.plot(xax,data[k])
            lbl2.append(str(time[k]))

        plt.subplot(121)
        leg = plt.legend(lbl,title = "dt ------ Fo")
        plt.xlabel('Simulation time (s)')
        plt.ylabel('RMS Error')
        plt.title('Error in ' + timestr + ' ' + str(div) + ' spatial points')
        plt.subplot(122)
        plt.legend(lbl2)
        plt.xlabel('Spatial point')
        plt.ylabel('Temperature')
        plt.title(timestr + ' result ')
        leg.draggable()
        plt.show()

    else:

        L = 1.0
        dx = L/div
        tf = .5
        freq = .15
        cpu = 0
        swept = 0
        dt = [.00001*k for k in range(1,5)]
        dt_dx = [k/dx for k in dt]
        err = np.empty([4,4])
        lbl = []
        lbl2 = []
        tm = np.copy(err)

        if swept and cpu:
            sch = SCHEME[2]
            timestr = Fname[w] + " " + sch
        elif swept:
            sch = SCHEME[1]
            timestr = Fname[w] + " " + sch
        else:
            sch = SCHEME[0]
            timestr = Fname[w] + " " + sch

        for i,dts in enumerate(dt):
            execstr = execut +  ' {0} {1} {2} {3} {4} {5} {6} {7}'.format(div,bks,dts,tf,freq,swept,cpu,Varfile)
            exeStr = shlex.split(execstr)
            proc = sp.Popen(exeStr)
            sp.Popen.wait(proc)
            lbl.append(str(dts) + " -- " + str(dt_dx[i]))

            # Now read in the files and plot, need better matplotlib strategy.
            fin = open(Varfile)
            data = []
            time = []
            h = []
            xax = np.linspace(0,L,div)

            for p,line in enumerate(fin):

                if p>1:

                    ar = [float(n) for n in line.split()]
                    data.append(ar[1:])
                    time.append(ar[0])
                    h.append(euler_exact(ar[0],div))

            for k in range(len(h)):
                err[i,k] = rmse(h[k],data[k])
                tm[i,k] = time[k]


        for k in range(4):
            plt.subplot(121)
            plt.plot(tm[k,:],err[k,:],'-o')
            plt.hold(True)
            plt.subplot(122)
            plt.plot(xax,data[k])
            lbl2.append(str(time[k]))

        plt.plot(xax,h[0])
        plt.subplot(121)
        leg = plt.legend(lbl,title = "dt ------ dt/dx")
        plt.xlabel('Simulation time (s)')
        plt.ylabel('RMS Error')
        plt.title('Error in ' + timestr + ' ' + str(div) + ' spatial points')
        leg.draggable()
        plt.subplot(122)
        lbl2.append("exact")
        leg2 = plt.legend(lbl2)
        plt.xlabel('Spatial point')
        plt.ylabel('Temperature')
        plt.title(timestr + ' result ')
        leg2.draggable()
        plt.show()
