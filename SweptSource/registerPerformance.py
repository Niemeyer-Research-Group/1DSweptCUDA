
import os
import os.path as op
import sys
import shlex
import subprocess as sp
import numpy as np
import pandas as pd

sourcepath = op.abspath(op.dirname(__file__))
rsltpath = op.join(sourcepath, 'Results')
binpath = op.join(sourcepath, 'bin') #Binary directory
gitpath = op.dirname(sourcepath) #Top level of git repo
plotpath = op.join(op.join(gitpath, "ResultPlots"), "performance") #Folder for plots
modpath = op.join(gitpath, "pyAnalysisTools")

sys.path.append(modpath)

import main_help as mh

prec = [
    ["Single", ""],
    ["Double", "-DREAL=double "]
]
fname = "KS"
timeout = '_Timing.txt'
rsltout = '_Result.dat'
sch = "Register"

os.chdir(sourcepath)

div = [2**k for k in range(11, 21)]
blx = [2**k for k in range(5, 11)]
wps = [int(k/32) for k in blx]

dt = 0.005
tf = dt*5e4
freq = tf*2.0
prog = op.join(binpath, "KSRegOut")
compStr = "nvcc -o " + prog + " KS1D_SweptRegister.cu  -gencode arch=compute_35,code=sm_35 -lm -restrict -Xptxas=-v "
dm = []
for pr in prec:

    timename = fname + "_" + pr[0] + "_" + sch
    vf = fname + pr[0] + rsltout
    timefile = timename + timeout
    plotstr = timename.replace("_"," ")

    timepath = op.join(rsltpath,timefile)
    Varfile = op.join(rsltpath,vf)
    myplot = op.join(plotpath, plotstr + ".pdf")

    if op.isfile(timepath):
        os.remove(timepath)

    t_fn = open(timepath,'a+')
    t_fn.write("Num_Spatial_Points\tThreads_per_Block\tTime_per_timestep_(us)\n")
    t_fn.close()

    compStr += pr[1] 

    for wp,tpb in zip(wps,blx):
            compit = shlex.split(compStr + "-DWPB=" + str(wp))
            proc = sp.Popen(compit)
            sp.Popen.wait(proc)
            for dv in div:
                    print "---------------------"
                    print "#divs #tpb"
                    print dv, tpb
                    execut = prog + ' {0} {1} {2} {3} {4} {5}'.format(dv,dt,tf,freq,Varfile,timepath)
                    exeStr = shlex.split(execut)
                    proc = sp.Popen(exeStr)
                    sp.Popen.wait(proc)

myFrame = mh.Perform(timepath)
myFrame.plotframe(plotpath)

