import pandas as pd #use to html for storage?
import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt

#Gather files
thispath = op.abspath(op.dirname(__file__))
files = []
for fl in os.listdir(thispath):
    if fl.endswith(".txt"):
        files.append(fl)

files = sorted(files)
dd = pd.read_table(files[0],delim_whitespace = True)
headers = dd.columns.values.tolist()
midx_name = ["Problem","Precision","Algorithm",headers[0]] 
dfs_all = []
#Put all files in dataframes.  Make a multiindex from names first.
for f in files:
    dd = pd.read_table(f,delim_whitespace = True)
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
