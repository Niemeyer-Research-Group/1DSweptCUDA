import numpy as np
import pandas as pd


mxB = 16
mxW = 64
mxSh = 3*2**14
mxReg = 2**16

bks = [64*k for k in range(1,17)]
print mxReg

def smB(bk,vs,prec):
    return (bk+4)*vs*prec*2

col = ["Threads/block", "SharedMem/Block", "Blocks/SM", "Threads/SM",
        "Warps/SM", "Registers/SM current reg/thd", "Number of threads Total",
        "Num Registers for max usage"
        ]

vs = [3,4]
a = []
pr = [4,8]
rg = [50,76]
tot = []
for i,p in enumerate(pr):

    for v in vs:
        atot = [bks]
        a = []
        a2 = []
        a3 = []

        print "Shared Memory usage " + str(p) + " bytes " + str(v) + " vec"
        for b in bks:
            a.append(smB(b,v,p))
            a2.append(min(mxSh/smB(b,v,p),16))
            a3.append(min(mxSh/smB(b,v,p),16)*b)


        atot.append(a)
        atot.append(a2)
        atot.append(a3)
        atot.append([k/32 for k in a3])
        atot.append([k*rg[i] for k in a3])
        atot.append([k*15 for k in a3])
        atot.append([mxReg/k for k in a3 if k>0])

        tot.append(atot)

tf = np.array(tot)
myDF = pd.DataFrame(tf)


myDF.columns=col


print myDF
