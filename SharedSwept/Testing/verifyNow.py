import numpy as np
import subprocess as sp
import shlex
import os
import os.path as op
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import sys

def rmse(exact,sim):
    return np.sqrt(np.mean((np.array(exact)-np.array(sim))**2))

def heats(n,L,x,t):
    alpha = 8.418e-5
    return 1.0/n**2 * np.exp(-alpha*t*(n*np.pi/L)**2) * np.cos(n*x*np.pi/L)


def heat_exact(t,L,divs):

    xm = np.linspace(0,L,divs)
    c0 = 50.0*L/3.0
    cout = 400.0*L/(np.pi**2)
    Tf1 = np.empty(int(divs))
    for i,xr in enumerate(xm):
        c  = 2
        ser = heats(c,L,xr,t)
        h = np.copy(ser)
        for k in range(1000):
            c += 2
            ser = heats(c,L,xr,t)
            h += ser

        Tf1[i] = (c0 - cout * h)

    return Tf1

prob = ["Heat","KS"]
prec = ["Single","Double"]

mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams["grid.alpha"] = 0.5
mpl.rcParams["axes.grid"] = True

exactpath = op.abspath(op.dirname(__file__))
mainpath = op.dirname(exactpath)
gitpath = op.dirname(mainpath)
datfil = op.join(exactpath,"ErrData.txt")

proc = sp.Popen("make", cwd=mainpath)
sp.Popen.wait(proc)

binpath = op.join(mainpath,'bin')

div = 4096
dx = .001
L = dx * (div-1)
bks = 512
freq = 1000
d = [1,5]
dts = [10**(k//2) * d[k%2]for k in range(-10,-4)]
print dts
tft = 50.0
cpu = 0
swept = 0
j=1

rmst = []
fig, ax = plt.subplots(figsize=(12.,12.), nrows=2, ncols=2)
axs = ax.ravel()
for i,pb in enumerate(prob):
    for pi, pc in enumerate(prec):

        Varfile = op.join(exactpath, pb +"_Result.dat")
        oFile = op.join(exactpath,"KS" + pc +  "_Official.txt")
        execut = op.join(binpath, pb + pc + "Out")
        rs = []
	tn = []

        for dt in dts:
	    tf = tft-0.5*dt
            execstr = execut +  ' {0} {1} {2} {3} {4} {5} {6} {7}'.format(div,bks,dt,tf,freq,swept,cpu,Varfile)

            exeStr = shlex.split(execstr)
            proc = sp.Popen(exeStr)
            sp.Popen.wait(proc)

            nm = np.genfromtxt(Varfile, skip_header=2)
            rs.append(nm[2:])
            print "t tinal real: {:0.8f}".format(nm[1])
            tn.append(nm[1])

        print tn
        if i:
            official = np.genfromtxt(oFile, skip_header=2)[2:]
        elif j:
            official = heat_exact(tn[1], L, div)
            j=0

        rms = []
        for r in rs:
	    gwen = rmse(official,r)
            rms.append(gwen)
            rmst.append(gwen)

        idx = i*2 + pi
        axnow = axs[idx]
        axnow.loglog(dts, rms, '-^')
        axnow.set_title(pb  + " " + pc)
        axnow.set_ylabel("Error")
        axnow.set_xlabel("delta t")
	
        print rms, axnow

#plt.show()
np.savetxt(datfil,rms)

plt.tight_layout()
plt.savefig("ErrChk.pdf", dpi=1000, bbox_inches="tight")

