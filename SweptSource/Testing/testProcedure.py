'''Testing script.'''

import os
import os.path as op
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
import pandas as pd
import palettable.colorbrewer as pal
import numpy as np
import datetime as dtime
import subprocess as sp
import warnings
warnings.filterwarnings("ignore")

exactpath = op.abspath(op.dirname(__file__))
sourcepath = op.dirname(exactpath)
rsltpath = op.join(sourcepath,'Results')
binpath = op.join(sourcepath,'bin') #Binary directory
gitpath = op.dirname(sourcepath) #Top level of git repo
plotpath = op.join(op.join(gitpath,"ResultPlots"),"ExactTesting") #Folder for plots
modpath = op.join(gitpath,"pyAnalysisTools")

os.chdir(sourcepath)

sys.path.append(modpath)
import main_help as mh
import result_help as rh

from exactpack.solvers.riemann import Sod
import warnings

mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.markersize'] = 20
mpl.rcParams["grid.alpha"] = 0.5
mpl.rcParams["axes.grid"] = True

plt.rc('axes', prop_cycle=cycler('color', pal.qualitative.Dark2_8.mpl_colors)+
    cycler('marker', ['D', 'o', 'h', '*', '^', 'x', 'v', '8']))

precision = "Double"
binary = precision+"Out"
alpha = 8.418e-5
dto = 1e-5
divs = 4096 
tpbs = 128

def Fo(dx,dt):
    alpha = 8.418e-5
    return alpha*dt/dx**2

def heat_exact(t, dx, divs):

    heats = lambda n, x: 1.0/n**2 * 
            np.exp(-alpha*t*(n*np.pi/L)**2) * np.cos(n*x*np.pi/L)

    xm = np.linspace(0, L, divs)
    c0 = 50.0*L/3.0
    cout = 400.0*L/(np.pi**2)
    Tf1 = np.empty(int(divs))
    for i,xr in enumerate(xm):
        c  = 2
        ser = heats(c,xr)
        h = np.copy(ser)
        for k in range(5000):
            c += 2
            ser = heats(c,L,xr,t)
            h += ser

        Tf1[i] = (c0 - cout * h)

    return Tf1

def euler_exact(t, dx, things):
    warnings.filterwarnings("ignore")
    d = 0.5*dx
    r = np.arange(d, 1.0-d, dx)
    if thing == 'velocity':
        thing = "velocity_x"

    solver = Sod()
    soln = solver(r, t)
    cl = []
    for t in things:
        cl.append(getattr(soln, thing))

    return cl

def ks_exact():
    filer = "KS" + precision + "_Official.txt"
    return rh.Solved(filer)    

def rmse(exact,sim):
    return np.sqrt(np.mean((np.array(exact)-np.array(sim))**2))
        
#Test that classic and swept give the same results

def exactRuns(problem, tf, dt=dto, div=divs, tpb=tpbs):
    
    binName = problem + binary
    executable = op.join(binpath, binName)
    vfile = op.join(sourcepath, 'temp.dat')
    typ = zip([1,1,0],[0,1,0])
    collect = []

    for s, a in typ:
        mh.runCUDA(executable, div, tpb, dt, tf, tf*2.0, s, a, vfile)
        antworten = rh.Solved(vfile)
        collect.append((antworten.varNames,antworten.tFinal,antworten.vals))
        print "Last tf = this tf? ", tf == antworten.tFinal[-1]
        tf = antworten.tFinal[-1]
        print "{!s} and tf = {:.10f}".format(problem, tf)

    return collect

def consistency(problem, tf, dt=dto, div=4096, tpb=128):

    binName = problem + binary
    executable = op.join(binpath, binName)
    vfile = op.join(sourcepath, 'temp.dat')
    typ = zip([1,1,0],[0,1,0])
    collect = []

    for s, a in typ:
        mh.runCUDA(executable, div, tpb, dt, tf, tf*2.0, s, a, vfile)
        antworten = rh.Solved(vfile)
        collect.append((antworten.varNames,antworten.tFinal,antworten.vals))
        print "Last tf = this tf? ", tf == antworten.tFinal[-1]
        tf = antworten.tFinal[-1]
        print "{!s} and tf = {:.10f}".format(problem, tf)

    return collect

def rmse(exact,sim):
    return np.sqrt(np.mean((np.array(exact)-np.array(sim))**2))

if __name__ == "__main__":
    
    sp.call("make", cwd=sourcepath)

    #Problem and finish time.  dt is set by end of swept run.
    probs = [["Heat", 100.0],
            ["KS", 100.0], 
            ["Euler", 0.22]] 

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
    deltat = [1.0e-7, 5.0e-7, 1.0e-6, 5.0e-6, 1.0e-5, 5.0e-5, 1.0e-4]
    exacts = {'Heat': heat_exact, 'KS': ks_exact, 'Euler': euler_exact}
    rlt = {'Heat': dict(), 'KS': dict(), 'Euler': dict()}

    for prob in probs:
        binName = prob[0] + binary
        executable = op.join(binpath, binName)
        vfile = op.join(sourcepath, 'temp.dat')
        for dt in deltat:
            mh.runCUDA(executable, divs, tpbs, dt, prob[1], prob[1]*2.0, 0, 0, vfile)
            rlt[prob[0]][dt] = rh.Solved(vfile)
        
        tempThis = rlt[prob[0]][deltat[0]]
        args = [tempThis.tFinal, tempThis.xGrid[1], ]
        rlt[prob[0]]['Exact'] = exacts[prob[0]](*args)







    
        
