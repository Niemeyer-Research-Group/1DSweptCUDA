#!/anaconda2/bin/python2

'''Testing script.'''

import os
import os.path as op
import sys
print sys.path
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
import palettable.colorbrewer as pal
import numpy as np
import datetime as dtime
import subprocess as sp
import warnings
import collections
warnings.filterwarnings("ignore")

exactpath = op.abspath(op.dirname(__file__))
sourcepath = op.dirname(exactpath)
rsltpath = op.join(sourcepath,'Results')
binpath = op.join(sourcepath,'bin') #Binary directory
gitpath = op.dirname(sourcepath) #Top level of git repo
plotpath = op.join(op.join(gitpath, "ResultPlots"), "ExactTesting") #Folder for plots
modpath = op.join(gitpath, "pyAnalysisTools")

os.chdir(sourcepath)

sys.path.append(modpath)
import main_help as mh
import result_help as rh

from exactpack.solvers.riemann import Sod
import warnings

mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams["grid.alpha"] = 0.5
mpl.rcParams["axes.grid"] = True

plt.rc('axes', prop_cycle=cycler('color', pal.qualitative.Dark2_8.mpl_colors)+
    cycler('marker', ['D', 'o', 'h', '*', '^', 'x', 'v', '8']))

precision = "Double"
binary = precision + "Out"
alpha = 8.418e-5
dto = 1e-5
divs = 1024
tpbs = 128
fqCoeff = 0.3
ksexactpath = op.join(exactpath, "KS" + precision + '_Official.txt')

probs = [["Heat", 50.0],
        ["KS", 20.0], 
        ["Euler", 0.22]] 

def make_KSExact():
    p = "KS"
    binName = p + binary
    executable = op.join(binpath, binName)
    dt = 1e-7
    tf = probs[1][1]
    mh.runCUDA(executable, divs, tpbs, dt, tf, tf*fqCoeff, 1, 0, ksexactpath)

def Fo(dx,dt):
    alpha = 8.418e-5
    return alpha*dt/dx**2

def heat_exact(t, divs, thing):
    dx = 0.001
    L = divs*dx
    heats = lambda n, x: 1.0/n**2 * np.exp(-alpha*t*(n*np.pi/L)**2) * np.cos(n*x*np.pi/L)    
    xm = np.linspace(0, L, divs)
    c0 = 50.0*L/3.0
    cout = 400.0*L/(np.pi**2)
    Tf1 = np.empty(int(divs))
    for i,xr in enumerate(xm):
        c  = 2
        ser = heats(c, xr)
        h = np.copy(ser)
        for k in range(5000):
            c += 2
            ser = heats(c, xr)
            h += ser

        Tf1[i] = (c0 - cout * h)

    return Tf1

def euler_exact(t, divs, thing):
    warnings.filterwarnings("ignore")
    thing = thing.lower()
    dx = 1.0/float(divs-2)
    d = 0.5*dx
    r = np.arange(d, 1.0-d, dx)
    if thing == 'velocity':
        thing = "velocity_x"

    solver = Sod()
    soln = solver(r, t)
    
    return getattr(soln, thing)

def ks_exact(t, divs, thing):
    return rh.Solved(ksexactpath).stripInitial()[thing][t]

def rmse(exact, sim):
    return np.sqrt(np.mean((np.array(exact)-np.array(sim))**2))
        
#Test that classic and swept give the same results

def consistency(problem, tf, dt=dto, div=4096, tpb=128):

    binName = problem + binary
    executable = op.join(binpath, binName)
    vfile = op.join(sourcepath, 'temp.dat')
    typ = zip([1,1,0],[0,1,0])
    collect = []

    for s, a in typ:
        mh.runCUDA(executable, div, tpb, dt, tf, tf*2.0, s, a, vfile)
        antworten = rh.Solved(vfile)
        collect.append((antworten.varNames, antworten.tFinal, antworten.vals))
        print "Last tf = this tf? ", tf == antworten.tFinal[-1]
        tf = antworten.tFinal[-1]
        print "{!s} and tf = {:.10f}".format(problem, tf)

    return collect

def rmse(exact,sim):
    return np.sqrt(np.mean((np.array(exact)-np.array(sim))**2))

#Swap out the second and last levels
def switchDict(dct):

    dSw = dict()
    dSa = dict()
    for pkey in dct.keys():
        dSw[pkey] = dict()  
        dSa[pkey] = dict()
        for dtkey in dct[pkey].keys():
            for vn in dct[pkey][dtkey].keys():
                if vn not in dSw[pkey].keys():
                    dSw[pkey][vn] = collections.defaultdict(dict)
                    dSa[pkey][vn] = collections.defaultdict(dict)

                for tf in dct[pkey][dtkey][vn].keys():
                    dSw[pkey][vn][tf][dtkey] = dct[pkey][dtkey][vn][tf]
                    dSa[pkey][vn][dtkey][tf] = dct[pkey][dtkey][vn][tf]

    return dSw, dSa

def plotit(dct, basename, shower, dtbool):
    #Figure, Subplot, Line, Xaxis: Yaxis
    lbls = ["dt (s)", "tFinal (s)"]
    axlabel = lbls if dtbool else lbls[::-1]
    ylbl = "Error"
    for k1 in dct.keys():
        fig = plt.figure(figsize=(10,8))
        probpath = op.join(plotpath, k1)
        pltname = k1 + basename
        pltpath = op.join(probpath, pltname)
        rw = 1
        fig.suptitle(k1 + ' | {} spatial pts'.format(divs), fontsize="large", fontweight='bold')
        
        if len(dct[k1].keys()) > 2:
            rw = 2

        for i, k2 in enumerate(dct[k1].keys()):
            ax = fig.add_subplot(rw, rw, i+1)
            ax.set_title(str(k2))
            ax.set_ylabel(ylbl)
            ax.set_xlabel(axlabel[0])
            for k3 in sorted(dct[k1][k2].keys()):
                x = []
                y = []
                for k4 in sorted(dct[k1][k2][k3].keys()):
                    x.append(k4)
                    y.append(dct[k1][k2][k3][k4])
                if dtbool:
                    ax.loglog(x, y, label=str(k3))
                else:
                    ax.plot(x, y, label=str(k3))

        hand, lbl = ax.get_legend_handles_labels()
        fig.legend(hand, lbl, loc='upper right', fontsize="large", title=axlabel[1])
        
        fig.subplots_adjust(bottom=0.08, right=0.85, top=0.9, 
                                wspace=0.15, hspace=0.25)
        fig.savefig(pltpath, dpi=1000, bbox_inches="tight")

        if shower:
            plt.show()
        
    return None


if __name__ == "__main__":
    
    sp.call("make", cwd=sourcepath)
    make_KSExact()

    #Problem and finish time.  dt is set by end of swept run.

    tol = 1e-5
    upshot = ["Passed", "Failed"]
    algs = ["Swept", "Alternative", "Classic"]

    outfile = op.join(plotpath, 'consistency.out')
    of = open(outfile, 'w')
    of.write(str(dtime.datetime.now()) + '\n')

    for ty in probs:
        rslt = consistency(*ty)
        classic = rslt[-1]
        tf = classic[1][-1]
        col = dict()
        for r, a in zip(rslt, algs):
            col[a] = dict()
            for i, y in enumerate(r[1]):
                if y == tf:
                    col[a][r[0][i]] = r[2][i]

        for a in algs[:-1]:
            for k in col[a].keys():
                diff = rmse(col[a][k], col[algs[-1]][k])
                u = upshot[diff>tol]
                outstr = '{!s} -- {!s} -- {!s}: {!s} ({:.3e})\n'.format(ty, a, k, u, diff)
                of.write(outstr)
                print outstr
                if u == upshot[1]:
                    plt.plot(np.abs(np.array(col[a][k]) -  np.array(col[algs[-1]][k])))
                    plt.show()

    of.close()

    #Now exact testing
    deltat = [5.0e-7, 1.0e-6, 5.0e-6, 1.0e-5, 5.0e-5, 1.0e-4]
    exacts = {'Heat': heat_exact, 'KS': ks_exact, 'Euler': euler_exact}
    rlt = collections.defaultdict(dict)
    rltCompare = collections.defaultdict(dict)

    for prob in probs:
        binName = prob[0] + binary
        executable = op.join(binpath, binName)
        vfile = op.join(exactpath, 'temp.dat')
        for dt in deltat:
            mh.runCUDA(executable, divs, tpbs, dt, prob[1], prob[1]*fqCoeff, 0, 0, vfile)
            rlt[prob[0]][dt] = rh.Solved(vfile).stripInitial()
            ths = rlt[prob[0]][dt]
            for tk in ths.keys():
                for tks in ths[tk].keys():
                    print tk, tks
        
        #t, dx, divs, varnames (Temperature) Could be arrays?  What do
        rlt[prob[0]]['Exact'] = collections.defaultdict(dict)
        rd = rlt[prob[0]][deltat[-1]]
        for vn in rd.keys():
            for tf in rd[vn].keys():
                rlt[prob[0]]['Exact'][vn][tf] = exacts[prob[0]](tf, divs, vn)

    for pkey in sorted(rlt.keys()):
        tDict = rlt[pkey]
        for dtkey in tDict.keys():
            rltCompare[pkey][dtkey] = collections.defaultdict(dict)
            if isinstance(dtkey, str):
                continue

            for vn in tDict[dtkey].keys():
                for tf in tDict[dtkey][vn].keys():
                    print pkey, dtkey, vn, tf
                    rltCompare[pkey][dtkey][vn][tf] = rmse(tDict[dtkey][vn][tf], tDict['Exact'][vn][tf])

    rsltbydt, rsltbytf = switchDict(rltCompare)
    lbls = ["dt (s)", "tFinal (s)"]

    plotit(rsltbydt, "_ByDeltat.pdf", True, True)

    plotit(rsltbytf, "_ByFinalTime.pdf", True, False)

