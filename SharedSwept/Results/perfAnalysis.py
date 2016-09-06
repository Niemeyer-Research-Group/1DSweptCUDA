import pandas as pd #use to html for storage?
import numpy as np
import subprocess as sp
import shlex
import os
import os.path as op
import sys
import matplotlib.pyplot as plt

#Gather files
thispath = op.abspath(op.dirname(__file__))
files = []
for fl in os.listdir(thispath):
    if fl.endswith(".txt"):
        files.append(fl)

files = sorted(files)
dfs = []
#Put all files in dataframes.  Make a multiindex from names first.
for f in files:
    dd = pd.read_table(f,delim_whitespace = True)
    headers = dd.columns.values.tolist()
    ds = dd.pivot(headers[0],headers[1],headers[2])
    dfs.append(ds)
