import matplotlib.pyplot as plt
from cycler import cycler
import subprocess as sp
import shlex
import os
import os.path as op
import sys
import pandas as pd

sourcepath = op.abspath(op.dirname(__file__))
rslt = op.join(sourcepath,"RegisterTiming.txt")
tmp = op.join(sourcepath,"temp.dat")

if op.isfile(rslt):
    os.remove(rslt)

t_fn = open(rslt,'a+')
t_fn.write("Num_Spatial_Points\tThreads_per_Block\tTime_per_timestep_(us)\n")
t_fn.close()

div = [2**k for k in range(11, 21)]
blx = [2**k for k in range(5, 11)]
wps = [int(k/32) for k in blx]

tf = 1000.0
dt = 0.005
freq = tf*2.0
prog = "./bin/KSRegOut"
compStr = "nvcc -o " + prog + " KS1D_SweptRegister.cu -gencode arch=compute_35,code=sm_35 -lm -restrict "

for wp,tpb in zip(wps,blx):
    compit = shlex.split(compStr + "-DWPB=" + str(wp))
    proc = sp.Popen(compit)
    sp.Popen.wait(proc)
    for dv in div:
            print "---------------------"
            print "#divs #tpb"
            print dv, tpb
            execut = prog +  ' {0} {1} {2} {3} {4} {5}'.format(dv,dt,tf,freq,tmp,rslt)
            exeStr = shlex.split(execut)
            proc = sp.Popen(exeStr)
            sp.Popen.wait(proc)

times = pd.read_table(rslt, delim_whitespace = True)
headers = times.columns.values.tolist()
headers = [h.replace("_"," ") for h in headers]
times.columns = headers
time_split = times.pivot(headers[0],headers[1],headers[2])
time_split.plot(logx = True, logy=True, grid=True, linewidth=2)
plt.ylabel(headers[2])
plt.title("REGISTER SWEPT PERFORMANCE")
plt.show()
