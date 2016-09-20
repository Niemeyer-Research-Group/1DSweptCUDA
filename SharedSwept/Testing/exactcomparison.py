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

def euler_exact(dx,t,dv,thing):
    warnings.filterwarnings("ignore")
    r = np.arange(0.5*dx,1,dx)
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
    print cpu, swept

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

    div = 2048.0
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
            Varfile = os.path.join(rsltpath, Fname[w] + "_Result.dat")
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

                f = open(Varfile)
                fin = tuple(f)
                ar = [float(n) for n in fin[0].split()]
                xax = np.linspace(0,ar[0],ar[1])

                for p in range(2,len(fin)):
                    ar = fin[p].split()
                    tm = float(ar[1])
                    if tm>0:
                        dMain.append([dts, tm] + [float(n) for n in ar[2:]])
                        exMain.append([dts, tm] + list(heat_exact(tm, L, div)))

                f.close()

            for k in range(len(exMain)):
                err.append([dMain[k][0], dMain[k][1], float(rmse(exMain[k][2:], dMain[k][2:]))])

            fig, (ax1, ax2) = plt.subplots(figsize=(14.,8.), ncols = 2)
            ax1.hold(True)
            ax2.hold(True)

            err = pd.DataFrame(err)
            head = ['dt','tf','Error']
            err.columns = head
            err = err.set_index(head[0])
            dfi = err.index.get_level_values(0).unique()

            for dfidx in dfi:
                dn = err.xs(dfidx)
                ax1.plot(dn['tf'], dn['Error'],'-o', label="{:.2e}       {:.5f}".format(dfidx,dfidx/dx))

            hand, lbl = ax1.get_legend_handles_labels()
            ax1.legend(hand, lbl, title = "dt  ------------  Fo", fontsize="medium", loc=0)
            ax1.set_xlabel('Simulation time (s)')
            ax1.set_ylabel('RMS Error')
            ax1.set_title(plotstr + ' | {0} spatial points     '.format(int(div)), fontsize="medium")
            ax1.grid(alpha = 0.5)

            simF = pd.DataFrame(dMain)
            exF = pd.DataFrame(exMain)
            simF = simF.set_index(0)
            exF = exF.set_index(0)
            typ = simF.index.get_level_values(0).unique()
            rt = typ[0]
            simF = simF.xs( rt )
            exF = exF.xs( rt )
            simF = simF.set_index(1)
            exF = exF.set_index(1)
            simF.columns = xax
            exF.columns = xax
            df_sim = simF.transpose()
            df_exact = exF.transpose()
            cl = df_sim.columns.values.tolist()
            df_exact.reset_index(inplace=True)
            df_sim.reset_index(inplace=True)

            for k, tfs in enumerate(cl):
                ax2.plot(df_sim['index'], df_sim[tfs], color[k], label="{:.2f} (s) Simulated".format(tfs))
                ax2.plot(df_exact.loc[pts,'index'], df_exact.loc[pts,tfs], 's'+color[k], label='Exact')

            hand, lbl = ax2.get_legend_handles_labels()
            ax2.legend(hand, lbl, loc=8, ncol=2, fontsize="medium")
            ax2.set_xlabel('Spatial point')
            ax2.set_ylabel('Temperature')
            ax2.set_title(plotstr + ' | dt = {:.4f} | {:.0f} spatial points   '.format(rt,div), fontsize="medium")
            ax2.set_ylim([0,30])
            ax2.grid(alpha=0.5)
            ax2.set_xlim([0,xax[-1]])

            plt.tight_layout()
            plt.savefig(myplot, dpi=1000, bbox_inches="tight")
            #plt.show()

   #Euler territory
    else:

        L = 1.0
        dx = L/div
        tf = .5
        freq = .15
        dt = [.00001*k for k in range(1,5)]
        dt_dx = [k/dx for k in dt]
        err = np.empty([4,4])
        for pr in prec:
            plotstr = timestr + " " + pr
            lbl = []
            lbl2 = []

            myplotE = os.path.join(myplotpath, plotstr + "_Error.pdf")
            myplotR = os.path.join(myplotpath, plotstr + "_Result.pdf")
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
                lbl.append("{:.4f}     {:.4f}".format(dts,dt_dx[i]))
                f = open(Varfile)
                fin = tuple(f)
                ar = [float(n) for n in fin[0].split()]
                dx = ar[2]
                dxhalf = 0.5*dx
                xax = np.arange(dxhalf,L,dx)

                for p in range(2,len(fin)):

                    ar = fin[p].split()
                    nameV = ar[0].lower()
                    tm = float(ar[1])
                    if tm>0:
                        idx = [dts,nameV,tm]
                        dMain.append( idx + [float(n) for n in ar[2:]])
                        exMain.append( idx + list(euler_exact(dx, tm, div, nameV)))

                f.close()

            for k in range(len(exMain)):
                err.append([dMain[k][1], dMain[k][0], dMain[k][2], float(rmse(exMain[k][3:], dMain[k][3:]))])


            err = pd.DataFrame(err)
            err = err.set_index(0)
            head = ['dt','tf','Error']
	    print err
            typ = err.index.get_level_values(0).unique()
            by_var = []

            for ty in typ:
                by_var.append(err.xs(ty))


            fig, ax = plt.subplots(2,2,figsize=(14.,8.))
            ax = ax.ravel()

            for i,df in enumerate(by_var):
                ax[i].set_title(typ[i], fontsize="medium")
                ax[i].hold(True)
                ax[i].grid(alpha=0.5)
                df.columns = head
                df = df.set_index('dt')
                dfidx = df.index.get_level_values(0).unique()
                Lin = []
                ax[i].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                for dfi in dfidx:
                    dn = df.xs(dfi)
                    ax[i].plot(dn['tf'], dn['Error'],'-o', label="{:.2e}       {:.5f}".format(dfi,dfi/dx))
                    ax[i].set_xlabel('Simulation time (s)')
                    ax[i].set_ylabel('RMS Error')

            hand, lbl = ax[0].get_legend_handles_labels()
            fig.legend(hand, lbl, 'upper_right', title="    dt  ---------  dt/dx", fontsize="medium")
            plt.suptitle(plotstr + ' | {0} spatial points                    {1}'.format(int(div)," "), fontsize="medium")
            plt.tight_layout(pad=0.2, w_pad=0.75, h_pad=1.5)
            plt.subplots_adjust(bottom=0.08, right=0.82, top=0.92)
            plt.savefig(myplotE, dpi=1000, bbox_inches="tight")
            #plt.show()

            simF = pd.DataFrame(dMain)
            exF = pd.DataFrame(exMain)
            simF = simF.set_index(0)
            exF = exF.set_index(0)
            typ = simF.index.get_level_values(0).unique()

            simF = simF.xs(typ[0])
            exF = exF.xs(typ[0])
            simF = simF.set_index(1)
            exF = exF.set_index(1)
            typ2 = simF.index.get_level_values(0).unique()

            fig2, ax2 = plt.subplots(2,2,figsize=(14.,8.))
            ax2 = ax2.ravel()
            for i,state in enumerate(typ2):
                df_sim = simF.xs(state)
                df_exact = exF.xs(state)
                df_sim = df_sim.set_index(2)
                df_exact = df_exact.set_index(2)
                df_sim.columns = xax
                df_exact.columns = xax
                df_sim = df_sim.transpose()
                df_exact = df_exact.transpose()
                cl = df_sim.columns.values.tolist()
                df_exact.reset_index(inplace=True)
                df_sim.reset_index(inplace=True)
                ax2[i].hold(True)
                ax2[i].set_title(state, fontsize="medium")
                ax2[i].grid(alpha=0.5)
                ax2[i].set_xlabel("Spatial point")

                for k,tfs in enumerate(cl):
                    ax2[i].plot(df_sim['index'], df_sim[tfs], color[k], label="{:.2f} (s) Simulated".format(tfs))
                    ax2[i].plot(df_exact.loc[pts,'index'], df_exact.loc[pts,tfs], 's'+color[k], label='Exact')

            hand, lbl = ax2[0].get_legend_handles_labels()
            fig2.suptitle(plotstr + ' | dt = {:.2e} | {:.0f} spatial points     '.format(typ[0],div), fontsize="medium")
            fig2.legend(hand, lbl, 'upper_right', fontsize="medium")
            plt.tight_layout(pad=0.2, w_pad=0.75, h_pad=1.5)
            plt.subplots_adjust(bottom=0.08, right=0.82, top=0.92)

            plt.savefig(myplotR, dpi=1000, bbox_inches="tight")
            plt.show()
