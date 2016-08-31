#This should be stand alone
import numpy as np
import subprocess as sp
import shlex
import os
import matplotlib.pyplot as plt
import sys
import pandas as pd
import time

from exactpack.solvers.riemann import Sod
import warnings

def euler_exact(t,dv,thing):
    warnings.filterwarnings("ignore")
    r = np.linspace(0.0,1.0,dv)
    if thing == 'velocity':
        thing = "velocity_x"

    solver = Sod()
    soln = solver(r,t)

    return getattr(soln,thing)


def heats(n,L,x,t):
    alpha = 8.418e-5
    return 1.0/n**2 * np.exp(-alpha*t*(n*np.pi/L)**2) * np.cos(n*x*np.pi/L)


def heat_exact(t,L,divs):

    xm = np.linspace(0,L,divs)
    c0 = 50.0*L/3.0
    cout = 400.0*L/(np.pi**2)
    Tf1 = np.empty(int(divs))
    for i,xr in enumerate(xm):
        c  = 2
        ser = heats(c,L,xr,t)
        h = np.copy(ser)
        for k in range(500):
            c += 2
            ser = heats(c,L,xr,t)
            h += ser

        Tf1[i] = (c0 - cout * h)

    return Tf1

def Fo(dx,dt):
    alpha = 8.418e-5
    return alpha*dt/dx**2


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
    color = ['r','b','k','g']
    pts = range(0,int(div),50)

    #Heat

    if not w:

        dx = .001
        L = dx * (div-1)
        tf = 200
        freq = 60
        dt = [.0015,.0010,.0005,.0001]
        # dt, var, tf

        for pr in prec:
            plotstr = timestr + " " + pr
            lbl = []
            lbl2 = []

            myplot = os.path.join(myplotpath, plotstr + ".pdf")
            Varfile = os.path.join(rsltpath, Fname[w] + pr + "_1D_Result.dat")
            execut = os.path.join(binpath, Fname[w] + pr + "Out")
            dMain = dict()
            err = dict()
            exMain = dict()

            #Main loop
            for i,dts in enumerate(dt):
                execstr = execut +  ' {0} {1} {2} {3} {4} {5} {6} {7}'.format(div,bks,dts,tf,freq,swept,cpu,Varfile)
                exeStr = shlex.split(execstr)
                proc = sp.Popen(exeStr)
                sp.Popen.wait(proc)

                if not dts in dMain.keys():
                    dMain[dts] = dict()
                if not dts in exMain.keys():
                    exMain[dts] = dict()


                fin = tuple(open(Varfile))
                ar = [float(n) for n in fin[0].split()]
                xax = np.linspace(0,ar[0],ar[1])

                for p in range(2,len(fin)):

                    ar = fin[p].split()

                    tm = float(ar[1])

                    dMain[dts][tm] = [float(n) for n in ar[2:]]
                    exMain[dts][tm] = heat_exact(tm, L, div)

            for keydt in dMain.keys():
                err[keydt] = dict()
                for keytf in dMain[keydt].keys():
                    err[keydt][keytf] = rmse(exMain[keydt][keytf], dMain[keydt][keytf])

            fig, (ax1, ax2) = plt.subplots(figsize=(14.,8.), ncols = 2)
            ax1.hold(True)
            ax2.hold(True)
            tmp = err.keys()
            tmp.sort()
            for key_one in tmp:
                x = []
                y = []
                tmp2 = err[key_one].keys()
                tmp2.sort()
                sv = key_one
                for key_two in tmp2:
                    x.append(key_two)
                    y.append(err[key_one][key_two])

                lbl.append("{:.4f}     {:.4f}".format(key_one, Fo(dx,key_one)))

                ax1.plot(x,y,'-o')

            ax1.legend(lbl,title = "dt  ------------  Fo", fontsize = "medium", loc=0)
            ax1.set_xlabel('Simulation time (s)')
            ax1.set_ylabel('RMS Error')
            ax1.set_title(plotstr + ' | {0} spatial points'.format(div), fontsize = "medium")
            ax1.grid(alpha = 0.5)

            tmp = exMain[sv].keys()
            tmp.sort()

            for k,key_f in enumerate(tmp):
                sim = dMain[sv][key_f]
                ex = exMain[sv][key_f]
                ax2.plot(xax,sim,color[k])
                ax2.plot(xax[pts], ex[pts],'s'+color[k])
                lbl2.append( "{:2.0f} (s) Simulated".format(key_f) )
                lbl2.append("Exact")

            ax2.legend(lbl2, loc=8, ncol=2, fontsize = "medium")
            ax2.set_xlabel('Spatial point')
            ax2.set_ylabel('Temperature')
            ax2.set_title(plotstr + ' | dt = {:.4f}'.format(sv), fontsize = "medium")
            ax2.set_ylim([0,30])
            ax2.grid(alpha=0.5)
            plt.savefig(myplot, dpi=1000, bbox_inches="tight")
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

            myplot = os.path.join(myplotpath, plotstr + ".pdf")
            Varfile = os.path.join(rsltpath, Fname[w] + pr + "_1D_Result.dat")
            execut = os.path.join(binpath, Fname[w] + pr + "Out")

            dMain = []
            err = []
            exMain = []

            #Main loop
            for i,dts in enumerate(dt):
                execstr = execut +  ' {0} {1} {2} {3} {4} {5} {6} {7}'.format(div,bks,dts,tf,freq,swept,cpu,Varfile)
                exeStr = shlex.split(execstr)
                proc = sp.Popen(exeStr)
                sp.Popen.wait(proc)
                lbl.append("{0}      {1}".format(dts,dt_dx[i]))


                fin = tuple(open(Varfile))
                ar = [float(n) for n in fin[0].split()]
                xax = np.linspace(0,ar[0],ar[1])

                for p in range(2,len(fin)):

                    ar = fin[p].split()
                    nameV = ar[0].lower()
                    tm = float(ar[1])
                    if tm>0:
                        dMain.append([dts,nameV,tm,[float(n) for n in ar[2:]]])

                        exMain.append([dts,nameV,tm] + list(euler_exact(tm, div, nameV)))

                for k in range(len(exMain)):
                    err.append([dMain[k][0],dMain[k][1],dMain[k][2],float(rmse(exMain[k][3:], dMain[k][3:]))])

            #And now you need to parse it.
            err = pd.DataFrame(err)
            print err.set_index([0,1,2])

            fig, (ax1, ax2) = plt.subplot(figsize=(12.,6.), ncol=2)
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
            ax1.grid(alpha=0.5)

            ax2.legend(lbl2, loc=8)
            ax2.set_xlabel('Spatial point')
            ax2.set_ylabel('Temperature')
            ax2.set_title(plotstr + ' result ')
            ax2.grid(alpha=0.5)
            plt.savefig(myplot)
            plt.show()
