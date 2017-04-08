'''
    Classes and functions for plotting the actual results of the simulations.

'''

import numpy as np
import os
import os.path as op
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import pandas as pd
import palettable.colorbrewer as pal
import subprocess as sp
import collections

plt.rc('axes', prop_cycle=cycler('color', pal.qualitative.Dark2_8.mpl_colors))

mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams["grid.alpha"] = 0.5
mpl.rcParams["axes.grid"] = True

'''
   Solved takes a path to a file that holds the results of the simulation.  
   The first line in the file has three values:  length of domain, #divisions, grid step
   Then each subsequent line is of the form: variable, time-stamp, solution.

   The method stripInitial parses the solution and returns a nested dict with keys variable, time and takes no input.
   The other methods plot the solution or make a gif, they take a function handle and axis specification and possibly a path to a folder where the plot is to be saved.
'''

class Solved(object):
   
    def __init__(self, vFile):
        self.ext = '.pdf'
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
        self.typname = self.plotname
        self.subpl = "Euler" in self.plotname

    def stripInitial(self):
        stripped = collections.defaultdict(dict)
        for i, t in enumerate(self.tFinal):
            if t == 0:
                continue
            
            stripped[self.varNames[i]][t] = self.vals[i,:]

        return stripped
        
    def plotResult(self, fh, ax):      
        
        if not self.subpl:
            for i, t in enumerate(self.tFinal):       
                ax.plot(self.xGrid, self.vals[i,:], label="{:.3f} (s)".format(t))
                
        else:          
            for axi, nm in zip(ax, self.plotTitles):
                idx = np.where(self.varNames == nm)
                vn = self.vals[idx, :].T
                tn = self.tFinal[idx]
                for i, tL in enumerate(tn):
                    axi.plot(self.xGrid, vn[:,i], label="{:.3f} (s)".format(tL))


    def annotatePlot(self, fh, ax):

        if not self.subpl:

            ax.set_ylabel(self.plotTitles[0])
            ax.set_xlabel('Spatial point')
            ax.set_title(self.plotname + " {0} spatial points".format(self.numpts))
            hand, lbl = ax.get_legend_handles_labels()
            fh.legend(hand, lbl, loc='upper right', fontsize="medium")
        
        else:
            fh.suptitle(self.plotname + 
                ' | {0} spatial points   '.format(self.numpts), 
                fontsize="large", fontweight='bold')

            for axi, nm in zip(ax, self.plotTitles):
                axi.set_title(nm)
                
            hand, lbl = ax[0].get_legend_handles_labels()
            fh.legend(hand, lbl, loc='upper right', fontsize="medium")

            fh.subplots_adjust(bottom=0.08, right=0.85, top=0.9, 
                                wspace=0.15, hspace=0.25)

    def savePlot(self, fh, plotpath, dp=200):
        
        plotfile = op.join(plotpath, self.plotname + self.ext)
        fh.savefig(plotfile, dpi=dp, bbox_inches="tight")

    def gifify(self, plotpath, fh, ax):

        self.ext = '.png'
        ppath = op.join(plotpath, 'Temp')
        os.chdir(ppath)
        giffile = op.join(plotpath, self.plotname + '.gif')
        avifile = op.join(ppath, self.plotname + '.avi')
        pfn = "T_"
        

        if not self.subpl:
            for i, t in enumerate(self.tFinal):       
                ax.plot(self.xGrid, self.vals[i,:])
                self.plotname = pfn + str(i)
                ax.set_ylabel(self.plotTitles[0])
                ax.set_xlabel('Spatial point')
                ax.set_title(self.plotname + " | " + self.typname + " {:d} spatial points | t = {:.3f}".format(self.numpts, t))
                self.savePlot(fh, ppath)
                ax.clear()
                
        else:
            fdict = collections.defaultdict(dict)

            for i, tf in enumerate(self.tFinal):
                fdict[tf][self.varNames[i]] = self.vals[i,:]
                
            for i, k1 in enumerate(sorted(fdict.keys())):
                self.plotname = pfn + str(i)

                for axi, k2 in zip(ax, sorted(fdict[k1].keys())):
                    axi.plot(self.xGrid, fdict[k1][k2])
                    axi.set_title(k2)

                fh.suptitle(self.plotname + " | " + self.typname + " | {:d} spatial points | t = {:.3f}".format(self.numpts, k1),
                fontsize="large", fontweight='bold')
                self.savePlot(fh, ppath, dp=80)

                for a in ax:
                    a.clear()

        st = 'linux'

        if st in sys.platform:

            sp.call(['ffmpeg', '-i', pfn+'%d.png', '-r', '4', avifile])
            sp.call(['ffmpeg', '-i', avifile, giffile])
            # except:
            #     print('------------------')
            #     print('Install ffmpeg with: sudo apt-get install ffmpeg')
            #     f = os.listdir(".")
            #     for fm in f:
            #         os.remove(fm)
                    
            #     raise SystemExit

            f = os.listdir(".")
            for fm in f:
                os.remove(fm)
        else:
            print('------------------')
            print('This script only makes gifs on linux with ffmpeg.  The images are still in the folder under ResultPlots/Gifs/Temp.')
        

        


        
        
