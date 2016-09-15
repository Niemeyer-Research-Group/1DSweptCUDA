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
plotpath = op.join(op.join(op.join(gitpath,"ResultPlots"),"performance"),"Summary") #Folder for plots
tablefile = op.join(plotpath,"SweptTestResults2.html")
csvfile = op.join(thispath,"LastTestResults.csv")

#Gather files
if not op.isdir(plotpath):
    os.mkdir(plotpath)

on = False
doit = False

if on:
    df_result = pd.read_csv(csvfile,index_col=range(4))
    midx_name = df_result.index.names

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
df_launch.index = pd.to_numeric(df_launch.index)
df_launch.sort_index(inplace=True)

plotfile = op.join(plotpath,"Best_Launch.pdf")

plt.figure(figsize=(14,8))
ax = df_launch.plot.bar(rot=0, legend=False)

plotItBar(ax, df_launch)

#Still need to annotate
plt.title("Best performance configuration all {:d} combinations".format(df_result.index.size))
plt.ylabel("Frequency")
plt.xlabel("Threads per block")
plt.ylim([0,df_launch.max()+5])
plt.grid(alpha=0.5)

#plt.savefig(plotfile, bbox_inches='tight')
algs = df_result.index.get_level_values("Algorithm").unique().tolist()
algs.sort()
probs = df_result.index.get_level_values("Problem").unique().tolist()
probs.sort()
precs = df_result.index.get_level_values("Precision").unique().tolist()
precs.sort()

#Now get the best scores for each.
df_best = df_result.min(axis=1)

#Also plot all the best launch bounds in barplot by runtype
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
    fig, ax = plt.subplots(1,2, figsize=(14,8))

    ax = ax.ravel()
    fig.suptitle(prob+" Best Case", fontsize='large', fontweight="bold")
    for i,prec in enumerate(precs):
        ser = df_now.xs(prec, level=midx_name[1])
        ser = ser.unstack(midx_name[2])
        ser.plot(ax=ax[i], logx=True, linewidth=2)
        ax[i].set_title(prec)
        ax[i].grid(alpha=0.5)
        ax[i].set_xlabel(midx_name[-1])
        ax[i].set_ylabel("Time per timestep (us)")
        
        ser2 = df_bdn.xs(prec, level=midx_name[1])
        ser2.plot.bar(ax=ax2[cnt], rot=0)
        ax2[cnt].set_title(prob+" "+prec, fontsize="medium")
        ax2[cnt].grid(alpha=0.5)
        ax2[cnt].set_xlabel("Threads per block")
        ax2[cnt].set_ylabel("Frequency")
        
        if cnt>0:
            lgg = ax2[cnt].legend()
            lgg.remove()
            
        cnt += 1

        #Set x and y probably

    lg = ax[0].legend()
    lg.remove()
    fig.subplots_adjust(bottom=0.08, right=0.82, top=0.9)
    ax[1].legend(bbox_to_anchor=(1.52,1.0), fontsize='medium')
    plotfile = op.join(plotpath,"Best configuations " + prob + ".pdf")
#   fig.savefig(plotfile, bbox_inches='tight')
    
fig2.subplots_adjust(bottom=0.08, right=0.82, top=0.9)
ax2[0].legend(bbox_to_anchor=(1.52,1.0), fontsize='medium')   
plotfile = op.join(plotpath, "Best configuration by runtype.pdf")
fig2.savefig(plotfile, bbox_inches='tight')  

on = raw_input("Pause")

df_classic = df_best.xs(algs[0], level="Algorithm")
df_sweptcpu = df_best.xs(algs[1], level="Algorithm")
df_swept = df_best.xs(algs[2], level="Algorithm")

df_gpuspeed = pd.DataFrame(df_classic/df_swept)
df_sharespeed = pd.DataFrame(df_classic/df_sweptcpu)
df_sharespeed.dropna(inplace=True)
df_gpuspeed = df_gpuspeed.unstack("Problem")
df_gpuspeed.columns = df_gpuspeed.columns.droplevel()
df_gpuspeed = df_gpuspeed.unstack("Precision")
df_sharespeed = df_sharespeed.unstack("Problem")
df_sharespeed.columns = df_sharespeed.columns.droplevel()
df_sharespeed = df_sharespeed.unstack("Precision")

fig, ax = plt.subplots(1,2, figsize=(14,8))
plt.suptitle("Swept algorithm speedup for best launch configuration",fontsize='large', fontweight="bold")

df_gpuspeed.plot(ax=ax[0], logx=True, linewidth=2)
ax[0].set_ylabel("Speedup vs Classic")
ax[0].grid(alpha=0.5)
ax[0].set_title("SweptGPU")
df_sharespeed.plot(ax=ax[1], logx=True, linewidth=2)
ax[1].grid(alpha=0.5)
ax[1].set_title("SweptCPUShare")

plotfile = op.join(plotpath,"Speedups.pdf")
plt.savefig(plotfile, bbox_inches='tight')

#Histogram of best launch bounds

#Like iterate or something over the six options
#dfbound.plot.bar(ax=ax,stacked=True)
#hand = [item.get_text() for item in ax.get_xticklabels()]

# I think I should run the same test with the CPU code.  Just take best one
# Same num spatial points but don't worry about threads per block.
# Just use different number of threads up to 12 or 16.  And run all the even combos
# Output the results and concatenate them in a text file.



#So that wil be 6 plots in three subplots of best case scenarios.

if doit:
    df_result.to_csv(csvfile)
