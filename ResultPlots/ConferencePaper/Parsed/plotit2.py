import numpy as np
import os
import sys
import os.path as op
import matplotlib as mpl
import matplotlib.pyplot as plt
import palettable.colorbrewer as pal
from datetime import datetime
from cycler import cycler

plt.rc('axes', prop_cycle=cycler('color', pal.qualitative.Dark2_8.mpl_colors)+
    cycler('marker',['D','o','v','*','^','x','h','8']))
cl = pal.qualitative.Dark2_6.mpl_colors
mk = ['D','o','v','*','^','x']

mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams["grid.alpha"] = 0.5
mpl.rcParams["axes.grid"] = True

thispath = op.abspath(op.dirname(__file__))
mpi = np.genfromtxt('MPICompare.txt')
heat = np.genfromtxt('HeatComplete.txt')
KSdiv = np.genfromtxt('Divides.txt')
KSall = np.genfromtxt('KSComplete.txt')
ylbl = "Time per timestep (us)"
xlbl = "Number of spatial points"
ylbl2 = "Speedup vs Classic"

#mpi
mpiLabels = ['MPIClassic', 'MPISwept', 'GPUClassic', 'GPUShared']
plotfile = op.join(thispath,"mpiRawPlot.pdf")
for i,mp in enumerate(mpiLabels):
    plt.loglog(mpi[:,0],mpi[:,i+1], color=cl[i], marker=mk[i])
    
plt.legend(mpiLabels, loc='upper left', fontsize='medium')
plt.xlabel(xlbl)
plt.ylabel(ylbl)
plt.xlim([heat[0,0],heat[-1,0]])    
plt.savefig(plotfile, bbox_inches='tight')
plt.close()

plt.legend(["Classic", "Shared"], loc='upper left', fontsize='medium')
plt.semilogx(mpi[:,0],mpi[:,-2], color=cl[2], marker=mk[2])
plt.hold(True)
plt.semilogx(mpi[:,0],mpi[:,-1],color=cl[3], marker=mk[3])
plt.xlabel(xlbl)
plt.legend(["Classic", "Shared"], loc='upper left', fontsize='medium')
plt.ylabel("GPU method speedup")
plotfile = op.join(thispath,"mpiSpeedupPlot.pdf")
plt.xlim([heat[0,0],heat[-1,0]])   

plt.savefig(plotfile, bbox_inches='tight')
plt.close()

#KSdiv

if True:
    divs = ["Divide","Multiply"]
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(14,8), sharey=True)
    #plt.suptitle("Improvement to KS from division avoidance",fontsize='large', fontweight="bold")
    ax1.loglog(KSdiv[:,0],KSdiv[:,1], KSdiv[:,0], KSdiv[:,2])
    ax1.set_title("Double Precision")
    ax2.loglog(KSdiv[:,0],KSdiv[:,3], KSdiv[:,0], KSdiv[:,4])
    
    ax2.set_title("Single Precision")
    ax1.set_ylabel(ylbl)
    ax1.set_xlabel(xlbl)
    ax2.set_xlabel(xlbl)
    ax1.set_xlim([heat[0,0],heat[-1,0]])
    
    plt.legend(divs, loc='upper left', fontsize='medium')
    plotfile = op.join(thispath,"divisionPlot.pdf")
    ax2.set_xlim([heat[0,0],heat[-1,0]])
    plt.savefig(plotfile, bbox_inches='tight')

#hand, lbl = ax.get_legend_handles_labels()
#Heat complete
prec = ["Double", "Single"]
plotfile = op.join(thispath,"HeatRaw.pdf")
ksorder = mpiLabels[2:]
heatorder = ['Classic', 'GPUShared', 'Hybrid']
ho=[prec[0]+" "+rd for rd in heatorder]+[prec[1]+" "+rd for rd in heatorder]
one = [1,2,3,6,7,8]
two = [4,5,9,10]
plt2 = [1,2,4,5]
for i, h in enumerate(ho):
    t = i+1+i//3*2
    plt.loglog(heat[:,0],heat[:,t], color=cl[i], marker=mk[i])
    plt.hold(True)


plt.legend(ho, loc='upper left', fontsize='medium')
plt.ylabel(ylbl)
plt.xlabel(xlbl)
plt.ylim([.5,100])
plt.xlim([heat[0,0],heat[-1,0]])

plt.savefig(plotfile, bbox_inches='tight')
plt.close()

ho.pop(3)
ho.pop(0)
plotfile = op.join(thispath,"HeatSpeed.pdf")

for i, h, p in zip(two,ho,plt2):
    plt.semilogx(heat[:,0],heat[:,i], color=cl[p], marker=mk[p])
    plt.hold(True)

plt.legend(ho, loc='upper right', fontsize='medium')
plt.xlabel(xlbl)
plt.xlim([heat[0,0],heat[-1,0]])  
plt.ylabel(ylbl2)
plt.savefig(plotfile, bbox_inches='tight')
plt.close()

reg = ["Register"]
ksorder += reg

#KS complete
ko=[prec[0]+" "+ rd for rd in ksorder]+[prec[1]+" "+ rd for rd in ksorder]

plotfile = op.join(thispath,"KSRaw.pdf")
for i, h in enumerate(ko):
    t = i+1+i//3*2
    plt.loglog(KSall[:,0],KSall[:,t], color=cl[i], marker=mk[i])
    plt.hold(True)


plt.legend(ko, loc='upper left', fontsize='medium')
plt.ylabel(ylbl)
plt.xlabel(xlbl)

plt.xlim([KSall[0,0],KSall[-1,0]])

plt.savefig(plotfile, bbox_inches='tight')
plt.close()

ko.pop(3)
ko.pop(0)
plotfile = op.join(thispath,"KSSpeed.pdf")

for i, h, p in zip(two,ko,plt2):
    plt.semilogx(KSall[:,0],KSall[:,i], color=cl[p], marker=mk[p])
    plt.hold(True)

plt.legend(ko, loc='upper right', fontsize='medium')
plt.xlabel(xlbl)
plt.xlim([KSall[0,0],KSall[-1,0]])  
plt.ylabel(ylbl2)
plt.savefig(plotfile, bbox_inches='tight')
plt.close()

