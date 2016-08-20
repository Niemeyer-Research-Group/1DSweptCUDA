
import numpy as np
import subprocess as sp
import shlex
import os
import matplotlib.pyplot as plt
from exactpack.solvers.riemann import Sod

def euler_exact():
    r = np.linspace(0.0,1.0,1025)
    print r[1]-r[0]
    t = .25

    solver = Sod()
    soln = solver(r,t)

    soln.plot('density')
    plt.show()

#All numeric first shot.
def heat_exact(f,x,t,L,n):
    rt = 1e-4
    at = 1e-5
    alpha = 8.418e-5;
    grid = np.linspace(0.0,L,n)
    c0 = 1.0/L*np.trapz(f,grid)

    Lhs1 = np.exp(-(np.pi*alpha/L)**2*t)*np.cos(x*np.pi/L)

    for k in range(3,52,2):
        cnt = float(k)
        Lhs1 +=  np.exp(-(cnt*np.pi*alpha/L)**2*t)*np.cos(cnt*x*np.pi/L)/(cnt**2)

    Tf1 = c0 - 4.0*L/(np.pi**2)*Lhs1

    return Tf1

if __name__ == '__main__':
    nm = 1024
    Lm = 2.0
    tm = np.linspace(2.0,500.0,6)
    xm = np.linspace(0.0,50.0*Lm,nm)
    lbl = []

    for k in tm:
        ar = []
        for g in xm:
            ar.append(heat_exact(xm,g,k,Lm,nm))

        plt.plot(xm,ar)
        plt.hold(True)
        lbl.append(str(k))

plt.legend(lbl)
plt.show()
