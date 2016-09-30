import numpy as numpy
import pandas as pd


mxB = 16
mxW = 64
mxSh = 3*2**14
mxReg = 2**16

bks = [64*k for k in range(1,17)]
print mxReg

def smB(bk,vs,prec):
    return (bk+4)*vs*prec*2

vs = [3,4]
a = []
pr = [4,8]
rg = [50,76]
for i,p in enumerate(pr):
    for v in vs:
        a = []
        a2 = []
        a3 = []
        print "Shared Memory usage " + str(p) + " bytes " + str(v) + " vec"
        for b in bks:
            a.append(smB(b,v,p))
            a2.append(min(mxSh/smB(b,v,p),16))
            a3.append(min(mxSh/smB(b,v,p),16)*b)

        print bks
        print a
        print "Number of blocks."
        print a2
        print "Number of threads/SM"
        print a3
        print "Number of Warps/SM"
        print [k/32 for k in a3]
        print "Number of Registers/SM at {} reg/thread".format(rg[i])
        print [k*rg[i] for k in a3]
        print "Number of threads Total"
        print [k*15 for k in a3]
        print "Num Registers for max usage"
        print [mxReg/k for k in a3 if k>0]
        print "------------------"
