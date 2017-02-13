'''Testing script.'''

import os
import os.path as op
import matplotlib.pyplot as plt
from cycler import cycler
import pandas as pd
import palettable.colorbrewer as pal
import numpy as np

exactpath = op.abspath(op.dirname(__file__))
sourcepath = op.dirname(exactpath)
datapath = op.join()
rsltpath = op.join(sourcepath,'Results')
binpath = op.join(sourcepath,'bin') #Binary directory
gitpath = op.dirname(sourcepath) #Top level of git repo
plotpath = op.join(op.join(gitpath,"ResultPlots"),"performance") #Folder for plots
modpath = op.join(gitpath,"pyAnalysisTools")

os.chdir(sourcepath)

sys.path.append(modpath)
import main_help as mh
import result_help as rh

binary = "DoubleOut"

#Test program against exact values.
class ExactTest(object):
    
    def __init__(self,problem,plotpath):
        self.problem  = problem
        self.plotpath = plotpath
        
    def exact(self,L):
        
    def rmse(self)
        
#Test that classic and swept give the same results

def consistency(problem, dt, tf, div=4096, tpb=128):

    binName = problem + binary
    executable = op.join(binpath, binName)
    vfile = op.join(sourcepath, 'temp.dat')
    #Classic, Swept, Alternative
    typ = [[0,1,1],[0,0,1]]
    collect = []

    for s, a in typ:
        mh.runCUDA(executable, div, tpb, dt, tf, tf*2.0, s, a, vfile)
        collect.append(rh.Solved(vfile))

    return collect

if __name__ == "__main__":

    probs = {"Heat" : [],
    "KS": [], 
    "Euler": []}

    for k in probs.keys():
        
