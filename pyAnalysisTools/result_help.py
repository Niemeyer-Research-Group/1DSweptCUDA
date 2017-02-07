
'''Docstring'''

import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import pandas as pd
import palettable.colorbrewer as pal

plt.rc('axes', prop_cycle=cycler('color', pal.qualitative.Dark2_8.mpl_colors))

mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams["grid.alpha"] = 0.5
mpl.rcParams["axes.grid"] = True

#Test program against exact values.
class Solved(object):
    
    def __init__(self, vFile):
        self.dataTuple = tuple(open(vFile))
        self.datafilename = op.splitext(op.basename(vFile))[0]
        strdimz = self.dataTuple[0].split()
        self.xGrid = np.linspace(0,float(strdimz[0]),int(strdimz[1]))
        self.numpts = int(strdimz[1])
        self.vals = np.genfromtxt(self.dataTuple, skip_header=1)[:,2:]
        self.varNames = np.genfromtxt(self.dataTuple, skip_header=1, dtype='string')[:,0]
        self.tFinal = np.genfromtxt(self.dataTuple, skip_header=1)[:,1]
        self.plotNames = np.unique(self.varNames)

    def plotResult(self, plotpath):
        
        if np.unique(self.tFinal.size)>10:
            return gifify(self, pFile) 

        plotname = op.join(plotpath, self.datafilename + ".pdf")
        plt.hold(True)
        
        if len(self.plotNames) < 2:
            for i, t in enumerate(self.tFinal):
                
                plt.plot(self.xGrid, self.vals[i,:], label="{:.3f} (s)".format(t))
                plt.ylabel(self.plotNames[0])
                plt.xlabel('Spatial point')
                plotstr = self.datafilename.replace("_"," ")
                plt.title(plotstr+ " {0} spatial points".format(self.numpts))
                
            
            plt.legend(loc='upper right', fontsize="medium")
            plt.show()
            plt.savefig(plotname, dpi=1000, bbox_inches="tight")

    def gifify(self, plotpath):
        plotname = op.join(plotpath, self.datafilename + ".pdf")
        
