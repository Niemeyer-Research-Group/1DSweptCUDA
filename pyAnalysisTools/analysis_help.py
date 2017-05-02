'''
    Classes and functions for plotting the actual results of the simulations.

'''

import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import pandas as pd
import palettable.colorbrewer as pal
import collections
import main_help as mh

plt.rc('axes', prop_cycle=cycler('color', pal.qualitative.Dark2_8.mpl_colors))

mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams["grid.alpha"] = 0.5
mpl.rcParams["axes.grid"] = True

def plotItBar(axi, dat):

    rects = axi.patches
    for r, val in zip(rects, dat):
        axi.text(r.get_x() + r.get_width()/2, val+.5, val, ha='center', va='bottom')

    return

def gatherTest(resultPath):

    midx_name= ["Problem", "Precision", "Algorithm", "Num Spatial Points"]
    files = []
    for fl in os.listdir(resultPath):
        if fl.count("_") == 3:
            files.append(op.join(resultPath, fl))

    files = sorted(files)
    dfs_all = []

    #Put all files in dataframes.  Make a multiindex from names first.
    for f in files:
        mydf = mh.Perform(f)
        idx1 = mydf.dataMatrix.index.get_level_values(0)
        outidx = op.basename(f).split("_")
        midx = []
        for i in idx1:
            outidx[-1] = i
            midx.append(tuple(outidx))

        idx_real = pd.MultiIndex.from_tuples(midx, names=midx_name)
        mydf.dataMatrix.set_index(idx_real, inplace=True)
        dfs_all.append(mydf.dataMatrix)

    return pd.concat(dfs_all)

class QualityRuns(object):
    def __init__(self, dataMat):
        self.bestRun = pd.DataFrame(dataMat.min(axis=1))
        bestIdx = dataMat.idxmin(axis=1)
        self.bestLaunch = bestIdx.value_counts()
        self.bestLaunch.index = pd.to_numeric(self.bestLaunch)
        self.bestLaunch.sort_index(inplace=True)

    
