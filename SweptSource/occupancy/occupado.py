import numpy as np
import pandas as pd


mxB = 16
mxW = 64
mxSh = 3*2**14
mxReg = 2**16

bks = [64*k for k in range(1,17)]


def smB(bk,vs,prec):
    return (bk+4)*vs*prec*2

col = ["Threads/ block", "SharedMem/ Block", "Blocks/ SM", "Threads/ SM",
        "Warps/ SM", "Registers/SM current", "Total Threads",
        "Regs Ideal", "Limiting factor"]

vs = [3,4]
a = []
pr = [4,8]
prn = ["Single","Double"]
rg = [50,76]
tot = []
idx = []
atot = [bks]
lims = ["Regs","Smem"]

for i,p in enumerate(pr):

    for v in vs:

        a = []
        a2 = []
        a3 = []

        idx.append((prn[i],str(v)))
        for b in bks:
            a.append(smB(b,v,p))
            a2.append(min(mxSh/smB(b,v,p),16))
            a3.append(min(mxSh/smB(b,v,p),16)*b)
            reglist=[k*rg[i] for k in a3]


        atot.append(a)
        atot.append(a2)
        atot.append(a3)
        atot.append([k/32 for k in a3]) #Warps/SM
        atot.append(reglist)
        atot.append([k*15 for k in a3])
        atot.append([mxReg/k if k>0 else 0 for k in a3])
        atot.append([lims[1] if k<mxReg else lims[0] for k in reglist])
        
        tot.append(pd.DataFrame(np.array(atot)))
        atot=[]


df_tot = pd.concat(tot)

df_tot.columns = df_tot.iloc[0]
df_tot = df_tot.iloc[1:]
df_tot = df_tot.transpose()

midx = pd.MultiIndex.from_product([idx,col[1:]], names=["Algorithm","Metric"])

df_tot.columns = midx

df_tot = df_tot.transpose()

df_tot.to_html("occupancytable.html")


