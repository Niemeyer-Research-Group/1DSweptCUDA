import pandas as pd #use to html for storage?
import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt
import palettable.colorbrewer as pal
from cycler import cycler

def plotItBar(axi, dat):

    rects = axi.patches
    for r,val in zip(rects,dat):
        axi.text(r.get_x() + r.get_width()/2, val+.5, val, ha='center', va='bottom')

    return

plt.rc('axes', prop_cycle=cycler('color', pal.qualitative.Dark2_8.mpl_colors))

thispath = op.abspath(op.dirname(__file__))
sourcepath = op.dirname(thispath)
gitpath = op.dirname(sourcepath) #Top level of git repo
plotpath = op.join(op.join(gitpath,"ResultPlots"),"performance") #Folder for plots
tablefile = op.join(plotpath,"SweptTestResults.html")
#Gather files

on = True

if on:
    df_result = pd.read_table(tablefile)

else:
    files = []
    for fl in os.listdir(thispath):
        if fl.endswith(".txt"):
            files.append(fl)

    files = sorted(files)
    dd = pd.read_table(files[0],delim_whitespace = True)
    headers = dd.columns.values.tolist()
    headers = [h.replace("_"," ") for h in headers]
    midx_name = ["Problem","Precision","Algorithm",headers[0]]
    dfs_all = []
    #Put all files in dataframes.  Make a multiindex from names first.
    for f in files:
        dd = pd.read_table(f,delim_whitespace = True)
        dd.columns = headers
        ds = dd.pivot(headers[0],headers[1],headers[2])
        idx1 = ds.index.get_level_values(0)
        outidx = f.split("_")
        midx = []
        for i in idx1:
            outidx[-1] = i
            midx.append(tuple(outidx))

        idx_real = pd.MultiIndex.from_tuples(midx,names=midx_name)
        ds.set_index(idx_real, inplace=True)
        dfs_all.append(ds)

    #Output all results to html table
    df_result = pd.concat(dfs_all)
    df_result.to_html(tablefile)

#Plot most common best launch.
df_best_idx = df_result.idxmin(axis=1)
df_launch = df_best_idx.value_counts()
df_launch.sort_index(inplace=True)

plt.figure(figsize=(14,8))
ax = df_launch.plot.bar(rot=0, legend=False)

plotItBar(ax, df_launch)

#Still need to annotate
plt.title()
plt.ylabel()
plt.xlabel()

plt.savefig()

#Now get the best scores for each.
df_best = df_result.min(axis=1)

#For each problem two subplots for two precisions and two or three lines.
for prob in df_best.index.get_level_values(midx_name[0]).unique():
    df_now = df_best.xs(prob,midx_name[0])
    fig, ax = plt.subplots(1,2, figsize=(14,8))
    fig.suptitle(prob)
    for i,prec in enumerate(df_best.index.get_level_values(midx_name[1]).unique()):
        ser = df_now.xs(prec,midx_name[1])
        ser = ser.unstack(midx_name[2])
        ser.plot.bar(ax=ax[i])
        ax[i].set_title(prec)
        #Set x and y probably

    #Legend outside plots?  Set it up like exact compare)

#So that wil be 6 plots in three subplots of best case scenarios.
