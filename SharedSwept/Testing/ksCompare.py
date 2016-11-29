import numpy as np
import subprocess as sp
import shlex
import os
import os.path as op
import matplotlib.pyplot as plt
import sys

def comp(typ,cl,sw):
    diff = np.sum(np.abs(cl-sw))
    if diff < 1e-7:
        print "-------!!!!----------\nKS " + typ + " passed!\n"
    else:
        print "-------!!!!----------\nKS " + typ + " failed!\n"
        print "The total difference is: {0}".format(diff)
        df = np.abs(cl-sw)
        plt.plot(df)
        plt.title("Difference at each spatial point")
        plt.show()


prec = ["Single","Double"]

exactpath = op.abspath(op.dirname(__file__))
mainpath = op.dirname(exactpath)
gitpath = op.dirname(mainpath)

proc = sp.Popen("make", cwd=mainpath)
sp.Popen.wait(proc)

binpath = op.join(mainpath,'bin')

div = 8192
bks = 64
tf = 200
freq = 2*tf
dt = .005
cpu = 0

for pr in prec:

    Varfile = op.join(exactpath,"KS_Result.dat")
    execut = os.path.join(binpath, "KS" + pr + "Out")
    dMain = np.empty([2,div])

    for swept in range(2)[::-1]:
        execstr = execut +  ' {0} {1} {2} {3} {4} {5} {6} {7}'.format(div,bks,dt,tf,freq,swept,cpu,Varfile)
        exeStr = shlex.split(execstr)
        proc = sp.Popen(exeStr)
        sp.Popen.wait(proc)

        rs = np.genfromtxt(Varfile, skip_header=2)
        tf = rs[1]-0.1*dt
        dMain[swept,:] = rs[2:]

    comp(pr,dMain[0,:],dMain[1,:])
