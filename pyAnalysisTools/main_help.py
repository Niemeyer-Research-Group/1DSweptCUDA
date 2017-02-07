'''Docstring'''

import os
import os.path as op
import matplotlib.pyplot as plt
from cycler import cycler
import pandas as pd
import palettable.colorbrewer as pal


class Perform(object):
    
    def __init__(self,dataset):
        self.dataMatrix = pd.read_table(dataset, delim_whitespace=True);
        self.textfile = op.abspath(dataset)
        self.datafilename = op.splitext(op.basename(dataset))[0]
        self.fileLocation = op.dirname(dataset)
        self.headers = [h.replace("_"," ") for h in self.dataMatrix.columns.values.tolist()]
        self._pivotparse()

    def _pivotparse(self):
        self.dataMatrix.columns = self.headers
        self.dataMatrix = self.dataMatrix.pivot(self.headers[0],self.headers[1],self.headers[2])
        
    def plotframe(self,plotpath):
        plotname = op.join(plotpath, self.datafilename + ".pdf")
        plt.rc('axes', prop_cycle=cycler('color', pal.qualitative.Dark2_8.mpl_colors))
        self.dataMatrix.plot(logx = True, logy=True, grid=True, linewidth=2)
        plt.ylabel(self.headers[2])
        plt.xlabel(self.headers[0].replace("_"," "))
        plotstr = self.datafilename.replace("_"," ")
        plt.title(plotstr)
        plt.savefig(plotname, dpi=1000, bbox_inches="tight")
        plt.show()

