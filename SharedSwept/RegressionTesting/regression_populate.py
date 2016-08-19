

import numpy as np
import subprocess as sp
import shlex
import os
import regression_test as rt
import sys

OPTIONS_ONE = [
    "KS",
    "Heat",
    "Euler"
]

SWEPT_OPT = [
    [0,0],
    [1,0],
    [1,1]
]

SCHEME = [
    "Classic",
    "SweptGPU",
    "SweptCPUshare"
]

thispath = os.path.abspath(os.path.dirname(__file__))
basepath = os.path.dirname(thispath)
binpath = os.path.join(basepath,'bin')
rsltpath = os.path.join(basepath,'Results')
rsltout = '1D_Result.dat'
dictout = os.path.join(thispath,"testData.json")

if not os.path.isdir(binpath):
    os.mkdir(binpath)

proc = sp.Popen(["make"],cwd = basepath)
sp.Popen.wait(proc)

div = [2**k for k in range(11,15)]
blk = 128
dt = [0.005,0.02,0.01]

all_runs = dict()

t_fin = [(blk*10-1)*k for k in dt]
freq = [tm*.4 for tm in t_fin]

nup = []

for i, prob in enumerate(OPTIONS_ONE):
    Varfile = os.path.join(rsltpath,prob + rsltout)
    ExecL = os.path.join(binpath, prob + 'Out')
    for k, oper in enumerate(SCHEME):
        if oper == SCHEME[-1] and prob == "KS":
            break
        for n in div:
            print "---------------------------"
            print prob, oper, n, t_fin[i]
            execut = ExecL +  ' {0} {1} {2} {3} {4} {5} {6} {7}'.format(n,blk,dt[i],t_fin[i],freq[i],SWEPT_OPT[k][0],SWEPT_OPT[k][1],Varfile)
            exeStr = shlex.split(execut)
            proc = sp.Popen(exeStr)
            sp.Popen.wait(proc)

            f = open(Varfile)
            for a, line in enumerate(f):
                if a>1:
                    ar = [float(p) for p in line.split()]
                    print ar[0]
                    nup.append([prob, oper, n, rt.consistency_test(prob,n,ar[0],ar[1:])])

for no in nup:
    print no
