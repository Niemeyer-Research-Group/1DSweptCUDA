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

# def euler_exact(dx, t, dv, thing):
#     warnings.filterwarnings("ignore")
#     r = np.arange(0.5*dx, 1, dx)
#     if thing == 'velocity':
#         thing = "velocity_x"

#     solver = Sod()
#     soln = solver(r, t)

#     return getattr(soln, thing)

# def heats(n,L,x,t):
#     alpha = 8.418e-5
#     return 1.0/n**2 * np.exp(-alpha*t*(n*np.pi/L)**2) * np.cos(n*x*np.pi/L)

# def heat_exact(t,L,divs):

#     xm = np.linspace(0,L,divs)
#     c0 = 50.0*L/3.0
#     cout = 400.0*L/(np.pi**2)
#     Tf1 = np.empty(int(divs))
#     for i,xr in enumerate(xm):
#         c  = 2
#         ser = heats(c,L,xr,t)
#         h = np.copy(ser)
#         for k in range(5000):
#             c += 2
#             ser = heats(c,L,xr,t)
#             h += ser

#         Tf1[i] = (c0 - cout * h)

#     return Tf1

# def Fo(dx,dt):
#     alpha = 8.418e-5
#     return alpha*dt/dx**2

mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.markersize'] = 20
mpl.rcParams["grid.alpha"] = 0.5
mpl.rcParams["axes.grid"] = True

plt.rc('axes', prop_cycle=cycler('color', pal.qualitative.Dark2_8.mpl_colors)+
    cycler('marker', ['D', 'o', 'h', '*', '^', 'x', 'v', '8']))

binary = "DoubleOut"

#Test program against exact values.
class ExactTest(object):
    
    def __init__(self,problem,plotpath):
        self.problem  = problem
        self.plotpath = plotpath
        
    def exact(self,L):
        pass
        
    def rmse(self):
        pass
        
#Test that classic and swept give the same results

def consistency(problem, tf, dt=0.00001, div=4096, tpb=128):

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
    probs = [["Heat", 115.0],
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

#Now the exact testing



    
        
