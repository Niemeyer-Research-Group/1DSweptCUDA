
import os
import os.path as op
import sys

sourcepath = op.abspath(op.dirname(__file__))
rsltpath = sourcepath
binpath = op.join(sourcepath,'bin') #Binary directory
gitpath = op.dirname(sourcepath) #Top level of git repo

import numpy as np
import subprocess as sp
import shlex
fanz = os.listdir(op.join(sourcepath,'bin'))
fang = [op.join(binpath,f) for f in fanz]
swept = 1
cpu = 0

div = [2**k for k in range(11, 21)]
blx = [2**k for k in range(6, 10)]
print div, blx

dt = 5e-8
tf = 1e4*dt

#Ensures "make" won't fail if there's no bin directory.


for ExecL,nm in zip(fang,fanz):
    timeout = nm+'_Timing.txt'
    rsltout = nm+'_Result.dat'

    timepath = op.join(rsltpath,timeout)
    Varfile = op.join(rsltpath,rsltout)

    if op.isfile(timepath):
        os.remove(timepath)

    t_fn = open(timepath,'a+')
    t_fn.write("Num_Spatial_Points\tThreads_per_Block\tTime_per_timestep_(us)\n")
    t_fn.close()

    for tpb in blx:
        for i,dvs in enumerate(div):
            freq = tf*2.0
            print "---------------------"
            print "#divs #tpb dt endTime"
            print dvs, tpb, dt, tf
            execut = ExecL +  ' {0} {1} {2} {3} {4} {5} {6} {7} {8}'.format(dvs,tpb,dt,tf,freq,swept,cpu,Varfile,timepath)
            exeStr = shlex.split(execut)
            proc = sp.Popen(exeStr)
            sp.Popen.wait(proc)




