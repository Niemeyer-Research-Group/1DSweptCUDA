

import numpy as np
import subprocess as sp
import shlex
import os
import json

OPTIONS_ONE = [
    "Heat",
    "Euler"
    "KS",
]

SWEPT_OPT = [
    [[0,0],
    [1,0],
    [1,1]],
]

SCHEME = [
    "Classic",
    "SweptGPU",
    "SweptCPUshare"
]

thispath = os.path.abspath(os.path.dirname(__file__))
basepath = os.path.basepath(thispath)
binpath = os.path.join(basepath,'bin')
rsltpath = os.path.join(basepath,'Results')
rsltout = '1D_Result.dat'
dictout = os.path.join(thispath,"testData.json")

if not os.path.isdir(binpath):
    os.mkdir(binpath)

sp.call("../make")

div = [2**k for k in range(11,21)]
blk = 256
dt = [0.2,0.2,0.005]

all_runs = dict()

t_fin = [blk*10*k for k in dt]
freq = [tm*.4 for tm in t_fin]

for i, prob in enumerate(OPTIONS_ONE):
    Varfile = os.path.join(rsltpath,prob + rsltout)
    ExecL = os.join(binpath, prob + 'Out')
    for k, oper in enumerate(SCHEME):
        if oper == SCHEME[-1] and prob == OPTIONS_ONE[-1]:
            break
        for n in div:
            execut = ExecL +  ' {0} {1} {2} {3} {4} {5} {6} {7} {8}'.format(n,blx,dt[i],t_fin[i],freq[i],SWEPT_OPT[k][0],SWEPT_OPT[k][1],Varfile)
            exeStr = shlex.split(execut)
            proc = sp.Popen(exeStr)
            sp.Popen.wait(proc)

            print "---------------------------"
            f = open(Varfile)
            for a, line in enumerate(f):
                if a>1:
                    ar = [float(p) for p in line.split()]
                    all_runs[prob][oper][n][ar[0]] = [ar[1:]]

dfile = open(dictout,'w')
json.dump(dfile,all_runs)
