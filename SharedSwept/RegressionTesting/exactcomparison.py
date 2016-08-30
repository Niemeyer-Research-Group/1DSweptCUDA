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
        for k in range(200):
            c += 2
            ser = heats(c,L,xr,t)
            h += ser

        Tf1[i] = (c0 - cout * h)

    return Tf1

def rmse(exact,sim):
    return np.sqrt(np.mean((np.array(exact)-np.array(sim))**2))

if __name__ == '__main__':


    w = int(sys.argv[1])
    sch = int(sys.argv[2])
    cpu = sch/2
    swept = int(bool(sch))

    Fname = ["Heat","Euler"]
    prec = ["Single","Double"]

    SCHEME = [
        "Classic",
        "SweptGPU",
        "SweptCPUshare"
    ]

    timestr = Fname[w] + " " + SCHEME[sch]
    exactpath = os.path.abspath(os.path.dirname(__file__))
    mainpath = os.path.dirname(exactpath)
    gitpath = os.path.dirname(mainpath)
    plotpath = os.path.join(os.path.join(gitpath,'ResultPlots'),'ExactTesting')

    proc = sp.Popen("make", cwd = mainpath)
    sp.Popen.wait(proc)

    rsltpath = os.path.join(mainpath,'Results')
    binpath = os.path.join(mainpath,'bin')
    myplotpath = os.path.join(plotpath,Fname[w])

    div = 1024.0
    bks = 64

    head = np.array(['Variable','time'])
    color = ['r','b','k','g']

    #Heat

    if not w:
        alpha = 8.418e-5
        dx = .001
        L = dx * (div-1)
        tf = 200
        freq = 60
        dt = [.0015,.0010,.0005,.0001]
        Fo = [alpha*k/dx**2 for k in dt]
        err = np.empty([4,4])

        tm = np.copy(err)

        for pr in prec:
            plotstr = timestr + " " + pr
            lbl = []
            lbl2 = []

            myplot = os.path.join(myplotpath, plotstr + ".pdf")
            Varfile = os.path.join(rsltpath, Fname[w] + pr + "_1D_Result.dat")
            execut = os.path.join(binpath, Fname[w] + pr + "Out")

            for i,dts in enumerate(dt):
                execstr = execut +  ' {0} {1} {2} {3} {4} {5} {6} {7}'.format(div,bks,dts,tf,freq,swept,cpu,Varfile)
                exeStr = shlex.split(execstr)
                proc = sp.Popen(exeStr)
                sp.Popen.wait(proc)
                lbl.append("{:.4f}      {:.4f}".format(dts,Fo[i]))

                # Now read in the files and plot, need better matplotlib strategy.
                fin = tuple(open(Varfile))
                ar = [float(n) for n in fin[0].split()]
                xax = np.linspace(0,ar[0],ar[1])
                h = []
                data= []
                time = []
                # YOU IDIOT.  YOU DON"T NEED TO MAKE XAX THE HEADER ROW.  ITLL DEFAULT TO NONE AND
                #THEN YOU DON"T NEED TO NAME THE TEMPERATURE AND TIME COLS.

                for p in range(2,len(fin)):

                    ar = [float(n) for n in fin[p].split()[1:]]
                    data.append(ar[1:])
                    time.append(ar[0])
                    h.append(heat_exact(ar[0],L,div))

                print len(h)

                pts = range(0,len(h[0]),20)

                for k in range(len(h)):
                    err[i,k] = rmse(h[k],data[k])
                    tm[i,k] = time[k]

            fig, (ax1, ax2) = plt.subplots(figsize = (10.,5.),ncols = 2)
            ax1.hold(True)
            ax2.hold(True)
            for k in range(4):
                ax1.plot(tm[k,:],err[k,:],'-o')
                ax2.plot(xax,data[k],color[k])
                ax2.plot(xax[pts], h[k][pts],'s'+color[k])
                lbl2.append("{:2.0f} Simulated ".format(time[k]) )
                lbl2.append("Exact")

            ax1.legend(lbl,title = "dt ------  Fo")
            ax1.set_xlabel('Simulation time (s)')
            ax1.set_ylabel('RMS Error')
            ax1.set_title(plotstr + ' Error {0} spatial points'.format(div))
            ax1.grid(alpha = 0.5)

            ax2.legend(lbl2, loc=8, ncol=2)
            ax2.set_xlabel('Spatial point')
            ax2.set_ylabel('Temperature')
            ax2.set_title(plotstr + ' result ')
            ax2.set_ylim([0,30])
            ax2.grid(alpha = 0.5)
            plt.savefig(myplot, dpi=2000)
            plt.show()
            
   #Euler territory
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
        for pr in prec:
            plotstr = timestr + " " + pr
            lbl = []
            lbl2 = []
            #I should be able to initialize a panel here and add to it.
            iDict = dict()

            myplot = os.path.join(myplotpath, plotstr + ".pdf")
            Varfile = os.path.join(rsltpath, Fname[w] + pr + "_1D_Result.dat")
            execut = os.path.join(binpath, Fname[w] + pr + "Out")
            for i,dts in enumerate(dt):
                execstr = execut +  ' {0} {1} {2} {3} {4} {5} {6} {7}'.format(div,bks,dts,tf,freq,swept,cpu,Varfile)
                exeStr = shlex.split(execstr)
                proc = sp.Popen(exeStr)
                sp.Popen.wait(proc)
                lbl.append("{0}      {1}".format(dts,dt_dx[i]))

                # Now read in the files and plot, need better matplotlib strategy.
                fin = tuple(open(Varfile))
                ar = [float(n) for n in fin[0].split()]
                xax = np.linspace(0,ar[0],ar[1])
                h_intit = np.concatenate((head,xax))
                h = []
                tbl_rslt = pd.read_table(Varfile, delim_whitespace = True, names = h_init, skip_rows = [0])
                tbl_rslt.set_index_names(head)
                tbl_real = tbl_rslt.transpose()
                iDict[dts] = tbl_real

                #Still gotta do something for this it's just doing my head in.
                time = set(tbl_real.get_level(1)) #Takes the level 1 row
                for tm in time:
                   h.append(euler_exact(tm,div))
               
               

            fig, (ax1, ax2) = plt.subplot(figsize = (12.,6.)ncols = 2)
            ax1.hold(True)
            ax2.hold(True)
            for k in range(4):
                ax1.plot(tm[k,:],err[k,:],'-o')
                ax2.plot(xax,data[k])
                lbl2.append(str(time[k]) + " Simulated " )
                lbl2.append("Exact")

            ax1.legend(lbl,title = "dt ------  Fo")
            ax1.set_xlabel('Simulation time (s)')
            ax1.set_ylabel('RMS Error')
            ax1.set_title('Error in ' + plotstr + ' ' + str(div) + ' spatial points')
            ax1.grid(alpha = 0.5)

            ax2.legend(lbl2, loc=8)
            ax2.set_xlabel('Spatial point')
            ax2.set_ylabel('Temperature')
            ax2.set_title(plotstr + ' result ')
            ax2.grid(alpha = 0.5)
            plt.savefig(myplot)
            plt.show()
