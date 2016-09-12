import pandas as pd #use to html for storage?
import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt


thispath = op.abspath(op.dirname(__file__))
sourcepath = op.dirname(thispath)
gitpath = op.dirname(sourcepath) #Top level of git repo
plotpath = op.join(op.join(gitpath,"ResultPlots"),"performance") #Folder for plots
tablefile = op.join(plotpath,"SweptTestResults.html")
#Gather files
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


df_result = pd.concat(dfs_all)
df_result.to_html(tablefile)

df_best = df_result.idxmin(axis=1)
df_launch = df_best.value_counts()
df_launch.sort_index(inplace=True)

xp = np.arange(df_launch.size)

plt.figure(figsize=(14,8))
ax = df_launch.plot.bar(rot=0)

rects = ax.patches

for r,val in zip(rects,df_launch):
    ax.text(r.get_x() + r.get_width()/2, val+.5, val, ha='center', va='bottom') 
    

    

    
    






