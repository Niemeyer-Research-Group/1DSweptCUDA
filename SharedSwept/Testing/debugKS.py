import numpy as np
import subprocess as sp
import shlex
import os
import os.path as op
import matplotlib.pyplot as plt
import sys

prec = ["Single","Double"]

exactpath = op.abspath(op.dirname(__file__))
mainpath = op.dirname(exactpath)
gitpath = op.dirname(mainpath)

proc = sp.Popen("make", cwd=mainpath)
sp.Popen.wait(proc)

binpath = op.join(mainpath,'bin')

div = 64
bks = 32
tf = .02
freq = .001
dt = .005
cpu = 0

for pr in prec:

    Varfile = op.join(exactpath,"KS_Result.dat")
    execut = op.join(binpath, "KS" + pr + "Out")
    fil = op.join(exactpath,"KS_" + pr + "")

    for swept in range(2)[::-1]:
        #Use cout and redirect left and right to file output classic will output to file as usual
        fil = op.join(exactpath,"KS_" + pr + str(swept)+ ".dat")
        execstr = execut +  ' {0} {1} {2} {3} {4} {5} {6} {7}'.format(div,bks,dt,tf,freq,swept,cpu,Varfile,fil)

        exeStr = shlex.split(execstr)
        proc = sp.Popen(exeStr)
        sp.Popen.wait(proc)

        rs = np.genfromtxt(Varfile, skip_header=2)
        tf = rs[1]-0.1*dt
