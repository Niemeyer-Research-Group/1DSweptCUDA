import pandas as pd
import numpy as np
import os
import sys
import os.path as op
import matplotlib.pyplot as plt
import palettable.colorbrewer as pal
from datetime import datetime
from cycler import cycler

#Flags for type of run.
readin = False
savepl = True
writeout = False

#Don't want to overwrite with previous version.
if readin:
    writeout = False

def plotItBar(axi, dat):

    rects = axi.patches
    for r, val in zip(rects, dat):
        axi.text(r.get_x() + r.get_width()/2, val+.5, val, ha='center', va='bottom')

    return

#Cycle through markers and colors.
plt.rc('axes', prop_cycle=cycler('color', pal.qualitative.Dark2_8.mpl_colors)+
    cycler('marker', ['D','o','h','*','^','x','v','8']))

#Set up directory structure.
thispath = op.abspath(op.dirname(__file__))
sourcepath = op.dirname(thispath)
gitpath = op.dirname(sourcepath) #Top level of git repo
plotpath = op.join(op.join(op.join(gitpath, "ResultPlots"), "performance"), "Summary") #Plots folder
tablefile = op.join(plotpath, "SweptTestResults.html")
storepath = op.join(thispath, "allResults.h5")
thisday = "Saved" + str(datetime.date(datetime.today())).replace("-", "_")
storage = pd.HDFStore(storepath)

#Gather files
if not op.isdir(plotpath):
    os.mkdir(plotpath)

if readin:
    vers = zip(range(len(storage.keys())), storage.keys())
    print vers
    choice = int(raw_input("Choose version by index:"))
    df_result = storage[storage.keys()[choice]]
    midx_name = df_result.index.names
    headers = df_result.columns.values.tolist()

else:
    files = []
    for fl in os.listdir(thispath):
        if fl.endswith(".txt"):
            files.append(fl)

    files = sorted(files)
    dd = pd.read_table(files[0], delim_whitespace=True)
    headers = dd.columns.values.tolist()
    headers = [h.replace("_", " ") for h in headers]
    midx_name = ["Problem", "Precision", "Algorithm", headers[0]]
    dfs_all = []
    #Put all files in dataframes.  Make a multiindex from names first.
    for f in files:
        dd = pd.read_table(f, delim_whitespace=True)
        dd.columns = headers
        ds = dd.pivot(headers[0], headers[1], headers[2])
        idx1 = ds.index.get_level_values(0)
        outidx = f.split("_")
        midx = []
        for i in idx1:
            outidx[-1] = i
            midx.append(tuple(outidx))

        idx_real = pd.MultiIndex.from_tuples(midx, names=midx_name)
        ds.set_index(idx_real, inplace=True)
        dfs_all.append(ds)


    df_result = pd.concat(dfs_all)

df_best = df_result.min(axis=1)
#Plot most common best launch.
df_best_idx = df_result.idxmin(axis=1)

df_launch = df_best_idx.value_counts()
df_launch.index = pd.to_numeric(df_launch.index)
df_launch.sort_index(inplace=True)

plotfile = op.join(plotpath,"Best_Launch.pdf")

plt.figure(figsize=(14,8))
ax = df_launch.plot.bar(rot=0, legend=False)

plotItBar(ax, df_launch)

plt.title("Best performance configuration all {:d} combinations".format(df_result.index.size))
plt.ylabel("Frequency")
plt.xlabel("Threads per block")
plt.ylim([0,df_launch.max()+5])
plt.grid(alpha=0.5)

if savepl:
    plt.savefig(plotfile, bbox_inches='tight')


#Get level values to iterate
algs = df_result.index.get_level_values("Algorithm").unique().tolist()
algs.sort()
probs = df_result.index.get_level_values("Problem").unique().tolist()
probs.sort()
precs = df_result.index.get_level_values("Precision").unique().tolist()
precs.sort()

#Now get the best scores for each.
df_best = df_result.min(axis=1)

#Plot all the best launch bounds in barplot by runtype
#Set up here, execution in following for loop.
dfbound = pd.DataFrame(df_best_idx)
dfbound = dfbound.unstack("Problem")
dfbound.columns = dfbound.columns.droplevel()
dfbound = dfbound.unstack("Precision")
dfbound = dfbound.unstack("Algorithm")
dfbound = dfbound.apply(pd.value_counts)
dfbound = dfbound.transpose()
dfbound.dropna(how="all", inplace=True)

fig2, ax2 = plt.subplots(3,2, figsize=(14,8))
fig2.suptitle("Best threads per block", fontsize='large', fontweight="bold")
ax2 = ax2.ravel()
cnt = 0

#For each problem two subplots for two precisions and two or three lines.
for prob in probs:
    df_now = df_best.xs(prob, level=midx_name[0])
    df_bdn = dfbound.xs(prob, level=midx_name[0])
    fig, ax = plt.subplots(1, 2, figsize=(14,8))

    ax = ax.ravel()
    fig.suptitle(prob+" Best Case", fontsize='large', fontweight="bold")
    for i, prec in enumerate(precs):
        ser = df_now.xs(prec, level=midx_name[1])
        ser = ser.unstack(midx_name[2])
        ser.plot(ax=ax[i], logx=True, logy=True, linewidth=2)
        ax[i].set_title(prec)
        ax[i].grid(alpha=0.5)
        ax[i].set_xlabel(midx_name[-1])

        #Only label first axis.
        if i == 0:
            ax[i].set_ylabel("Time per timestep (us)")

        ser2 = df_bdn.xs(prec, level=midx_name[1])
        ser2.plot.bar(ax=ax2[cnt], rot=0)
        ax2[cnt].set_title(prob+" "+prec, fontsize="medium")
        ax2[cnt].grid(alpha=0.5)
        ax2[cnt].set_ylabel("Frequency")
        ax2[cnt].set_xlabel("")

        if cnt > 0:
            lgg = ax2[cnt].legend()
            lgg.remove()

        cnt += 1



    lg = ax[0].legend()
    lg.remove()
    fig.subplots_adjust(bottom=0.08, right=0.82, top=0.9)
    ax[1].legend(bbox_to_anchor=(1.52, 1.0), fontsize='medium')
    plotfile = op.join(plotpath, "Best configuations " + prob + ".pdf")

    if savepl:
        fig.savefig(plotfile, bbox_inches='tight')

fig2.tight_layout(pad=0.2, w_pad=0.75, h_pad=1.0)
fig2.subplots_adjust(bottom=0.08, right=0.82, top=0.9)
hand, lbl = ax2[0].get_legend_handles_labels()
ax2[0].legend().remove()
fig2.legend(hand, lbl, 'upper right', title="Threads per block", fontsize="medium")
plotfile = op.join(plotpath, "Best configuration by runtype.pdf")

if savepl:
    fig2.savefig(plotfile, bbox_inches='tight')

#Now plot speedups, Time of best classic/Time best Swept.
df_classic = df_best.xs(algs[0], level="Algorithm")
df_sweptcpu = df_best.drop([algs[0],algs[-1]], level="Algorithm")
df_sweptcpu.index = df_sweptcpu.index.droplevel(2)
df_swept = df_best.xs(algs[-1], level="Algorithm")

df_gpuspeed = pd.DataFrame(df_classic/df_swept)
df_sharespeed = pd.DataFrame(df_classic/df_sweptcpu)
df_sharespeed.dropna(inplace=True) #Drop nans for KS CPU shared.
df_gpuspeed = df_gpuspeed.unstack("Problem")
df_gpuspeed.columns = df_gpuspeed.columns.droplevel()
df_gpuspeed = df_gpuspeed.unstack("Precision")
df_sharespeed = df_sharespeed.unstack("Problem")
df_sharespeed.columns = df_sharespeed.columns.droplevel()
df_sharespeed = df_sharespeed.unstack("Precision")

fig, ax = plt.subplots(1, 2, figsize=(14,8))
plt.suptitle("Swept algorithm speedup for best launch configuration", fontsize='large', fontweight="bold")

df_gpuspeed.plot(ax=ax[0], logx=True, linewidth=2)
ax[0].set_ylabel("Speedup vs Classic")
ax[0].grid(alpha=0.5)
ax[0].set_title("SweptGPU")
df_sharespeed.plot(ax=ax[1], logx=True, linewidth=2)
ax[1].grid(alpha=0.5)
ax[1].set_title("Alternative")

plotfile = op.join(plotpath, "Speedups.pdf")

if savepl:
    fig.savefig(plotfile, bbox_inches='tight')

#Plot MPI version results vs CUDA for KS.
fig, ax = plt.subplots(1,1, figsize=(14,8))
dfM = pd.read_csv("KS_MPI.csv")
mpihead = dfM.columns.values.tolist()
dfM = dfM.set_index(mpihead[0])
dfK = pd.DataFrame(df_best.xs("KS",level="Problem"))
dfK = dfK.unstack("Precision")
dfK = dfK.unstack("Algorithm")
dfK.columns = dfK.columns.droplevel()

dfKS = dfM.join(dfK)
dfKS.dropna(inplace=True)
dfKS.plot(ax=ax, logx=True, logy=True, linewidth=2,figsize=(14,8))
ax.set_ylabel(headers[2])

ax.set_title("KS MPI vs GPU implementation")

if savepl:
    fig.savefig(plotfile, bbox_inches='tight')

#Plot MPI version results vs CUDA for KS.
fig, ax = plt.subplots(1,1, figsize=(14,8))
dfM = pd.read_csv("KS_MPI.csv")
mpihead = dfM.columns.values.tolist()
dfM = dfM.set_index(mpihead[0])
dfK = pd.DataFrame(df_best.xs("KS",level="Problem"))
dfK = dfK.unstack("Precision")
dfK = dfK.unstack("Algorithm")
dfK.columns = dfK.columns.droplevel()

dfKS = dfM.join(dfK)
dfKS.dropna(inplace=True)
dfKS.plot(ax=ax, logx=True, logy=True, linewidth=2,figsize=(14,8))
ax.set_ylabel(headers[2])

ax.set_title("KS MPI vs GPU implementation")
ax.grid(alpha=0.5)

tblMPI = op.join(plotpath,"KS_MPI_GPU_Comparison.html")
dfKS.to_html(tblMPI)

plotfile = op.join(plotpath,"KS_MPI_GPU_Comparison.pdf")
if savepl:
    fig.savefig(plotfile, bbox_inches='tight')
ax.grid(alpha=0.5)

tblMPI = op.join(plotpath,"KS_MPI_GPU_Comparison.html")
dfKS.to_html(tblMPI)

if writeout:
    df_result.to_html(tablefile)
    if thisday in storage.keys():
        fl = raw_input("You've already written to the hd5 today.  Overwrite? [y/n]")
        if "n" in fl:
            sys.exit(1)

    storage.put(thisday,df_result)
    
    storage.close()
