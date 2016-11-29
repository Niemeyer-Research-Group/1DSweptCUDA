import numpy as np
import os
import sys
import os.path as op
import matplotlib.pyplot as plt
import palettable.colorbrewer as pal
from datetime import datetime
from cycler import cycler

plt.rc('axes', prop_cycle=cycler('color', pal.qualitative.Dark2_8.mpl_colors)+
    cycler('marker',['D','o','h','*','^','x','v','8']))
    
thispath = op.abspath(op.dirname(__file__))
mpi = np.genfromtxt('MPICompare.txt')
heat = np.genfromtxt('HeatComplete.txt')
KSdiv = np.genfromtxt('Divides.txt')
KSall = np.genfromtxt('KSComplete.txt')

ylbl = "Time per timestep (us)"
xlbl = "Number of spatial points"

#mpi
mpiLabels = ['MPISwept', 'MPIClassic', 'Classic', 'GPUSwept', 'Register']
for i,mp in enumerate(mpiLabels):
    plt.loglog(mpi[:,0],mpi[:,i+1])
    plt.hold(True)
    
plt.legend(mpiLabels, loc='upper left', fontsize='medium')
plt.grid(alpha=0.5)
plt.title("MPI and GPU implementation comparison")
plt.ylabel(ylbl)
plt.xlabel(xlbl)
plotfile = op.join(thispath,"mpiPlot.pdf")
plt.xlim([heat[0,0],heat[-1,0]])
plt.savefig(plotfile, bbox_inches='tight')

#KSdiv
divs = ["Divide","Multiply"]
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(14,8), sharey=True)
plt.suptitle("Improvement to KS from division avoidance",fontsize='large', fontweight="bold")
ax1.loglog(KSdiv[:,0],KSdiv[:,1], KSdiv[:,0], KSdiv[:,2])
ax1.set_title("Double Precision")
ax2.loglog(KSdiv[:,0],KSdiv[:,3], KSdiv[:,0], KSdiv[:,4])

ax2.set_title("Single Precision")
ax1.set_ylabel(ylbl)
ax1.set_xlabel(xlbl)
ax2.set_xlabel(xlbl)
ax1.set_xlim([heat[0,0],heat[-1,0]])

plt.legend(divs, loc='upper left', fontsize='medium')
ax1.grid(alpha=0.5)
ax2.grid(alpha=0.5)
plotfile = op.join(thispath,"divisionPlot.pdf")
ax2.set_xlim([heat[0,0],heat[-1,0]])
plt.savefig(plotfile, bbox_inches='tight')

#hand, lbl = ax.get_legend_handles_labels()
#Heat complete
prec = ["Double", "Single"]
ksorder = mpiLabels[2:]
heatorder = ['Classic', 'GPUSwept', 'Hybrid']
ho=[prec[0]+" "+rd for rd in heatorder]+[prec[1]+" "+rd for rd in heatorder]

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(14,8))
plt.suptitle("Heat",fontsize='large', fontweight="bold")
ax1.loglog(heat[:,0],heat[:,1], heat[:,0], heat[:,2], heat[:,0], heat[:,3])
ax1.hold(True)
ax1.loglog(heat[:,0],heat[:,6], heat[:,0], heat[:,7], heat[:,0], heat[:,8])
ax1.legend(ho, loc='upper left', fontsize='medium')
ax1.set_ylabel(ylbl)
ax1.set_xlabel(xlbl)
ax1.set_title("a")
ax1.set_xlim([heat[0,0],heat[-1,0]])

ho.pop(3)
ho.pop(0)

ax2.semilogx(heat[:,0],heat[:,4], heat[:,0], heat[:,5])
ax2.hold(True)
ax2.semilogx(heat[:,0],heat[:,9], heat[:,0], heat[:,10])
ax2.legend(ho, loc='upper right', fontsize='medium')
ax1.grid(alpha=0.5)
ax2.grid(alpha=0.5)
ax2.set_xlabel(xlbl)
ax2.set_ylabel("Speedup vs Classic")
ax2.set_title("b")
fig.tight_layout(pad=0.2, w_pad=0.75, h_pad=1.5)
fig.subplots_adjust(bottom=0.08, right=0.92, top=0.92)
plotfile = op.join(thispath,"heatComplete.pdf")
ax2.set_xlim([heat[0,0],heat[-1,0]])
plt.savefig(plotfile, bbox_inches='tight')


#KS complete
ko=[prec[0]+" "+ rd for rd in ksorder]+[prec[1]+" "+ rd for rd in ksorder]

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(14,8))
plt.suptitle("KS",fontsize='large', fontweight="bold")
ax1.loglog(KSall[:,0],KSall[:,1], KSall[:,0], KSall[:,2], KSall[:,0], KSall[:,3])
ax1.hold(True)
ax1.loglog(KSall[:,0],KSall[:,6], KSall[:,0], KSall[:,7], KSall[:,0], KSall[:,8])
ax1.legend(ko, loc='upper left', fontsize='medium')
ax1.set_ylabel(ylbl)
ax1.set_xlabel(xlbl)
ax1.set_xlim([heat[0,0],heat[-1,0]])
ax1.set_title("a")

ko.pop(3)
ko.pop(0)

ax2.semilogx(KSall[:,0],KSall[:,4], KSall[:,0], KSall[:,5])
ax2.hold(True)
ax2.semilogx(KSall[:,0],KSall[:,9], KSall[:,0], KSall[:,10])
ax2.legend(ko, loc='upper right', fontsize='medium')
ax1.grid(alpha=0.5)
ax2.grid(alpha=0.5)
ax2.set_xlabel(xlbl)
ax2.set_ylabel("Speedup vs Classic")
ax2.set_title("b")
fig.tight_layout(pad=0.2, w_pad=0.75, h_pad=1.0)
fig.subplots_adjust(bottom=0.08, right=0.92, top=0.92)
plotfile = op.join(thispath,"KSallComplete.pdf")
ax2.set_xlim([heat[0,0],heat[-1,0]])
plt.savefig(plotfile, bbox_inches='tight')

