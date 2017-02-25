
'''Docstring'''

import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import pandas as pd
import palettable.colorbrewer as pal
import collections

plt.rc('axes', prop_cycle=cycler('color', pal.qualitative.Dark2_8.mpl_colors))

mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams["grid.alpha"] = 0.5
mpl.rcParams["axes.grid"] = True


#So you need to make the actual figure in the script and pick apart the axes.          

class Solved(object):
    
    ext = '.pdf'
    def __init__(self, vFile):
        dataTuple = tuple(open(vFile))
        strdimz = dataTuple[0].split()
        self.datafilename = op.splitext(op.basename(vFile))[0]
        self.xGrid = np.linspace(0,float(strdimz[0]),int(strdimz[1]))
        self.numpts = int(strdimz[1])
        self.vals = np.genfromtxt(dataTuple, skip_header=1)[:,2:]
        self.varNames = np.genfromtxt(dataTuple, skip_header=1, dtype='string')[:,0]
        self.tFinal = np.around(np.genfromtxt(dataTuple, skip_header=1)[:,1], decimals=7)
        self.plotTitles = np.unique(self.varNames)
        self.plotname = self.datafilename.split("_")[0]
        self.subpl = "Euler" in self.plotname

    def stripInitial(self):
        stripped = collections.defaultdict(dict)
        for i, t in enumerate(self.tFinal):
            if t == 0:
                continue
            
            stripped[self.varNames[i]][t] = self.vals[i,:]

        return stripped
        
    def plotResult(self, fhandle, axhandle):
       
        if np.unique(self.tFinal.size)>10:
            return self.gifify(self, plotpath)         
        
        if not self.subpl:
            for i, t in enumerate(self.tFinal):       
                axhandle.plot(self.xGrid, self.vals[i,:], label="{:.3f} (s)".format(t))
                
        else:
                        
            for axi, nm in zip(axhandle, self.plotTitles):
                idx = np.where(self.varNames == nm)
                vn = self.vals[idx, :].T
                tn = self.tFinal[idx]
                for i, tL in enumerate(tn):
                    axi.plot(self.xGrid, vn[:,i], label="{:.3f} (s)".format(tL))


    def annotatePlot(self, fh, axh):

        if not self.subpl:

            axh.set_ylabel(self.plotTitles[0])
            axh.set_xlabel('Spatial point')
            axh.set_title(self.plotname + " {0} spatial points".format(self.numpts))
            hand, lbl = axh.get_legend_handles_labels()
            fh.legend(hand, lbl, loc='upper right', fontsize="medium")
        
        else:
            fh.suptitle(self.plotname + 
                ' | {0} spatial points   '.format(self.numpts), 
                fontsize="large", fontweight='bold')

            for axi, nm in zip(axh, self.plotTitles):
                axi.set_title(nm)
                
            hand, lbl = axh[0].get_legend_handles_labels()
            fh.legend(hand, lbl, loc='upper right', fontsize="medium")

            fh.subplots_adjust(bottom=0.08, right=0.85, top=0.9, 
                                wspace=0.15, hspace=0.25)

    def savePlot(self, fh, plotpath):
        
        plotfile = op.join(plotpath, self.plotname + self.ext)
        fh.savefig(plotfile, dpi=1000, bbox_inches="tight")

    def gifify(self, plotpath):
        #plotfile = op.join(plotpath, self.plotname + self.ext)
        pass
        
