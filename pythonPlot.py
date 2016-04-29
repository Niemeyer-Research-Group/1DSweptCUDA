# -*- coding: utf-8 -*-

# Just writing a plotting script for the Swept rule CUDA.
# Perhaps this will also be the calling script.

import matplotlib.pyplot as plt
import numpy as np
import subprocess as sp
import shlex


compStr = "nvcc 1DSweptRule_main.cu -o SweptOut -arch sm_35"
execut = ['./SweptOut']
compArg = shlex.split(compStr)
proc = sp.Popen(compArg)
sp.Popen.wait(proc)
proc = sp.Popen(execut)
sp.Popen.wait(proc)

fin = open('1DHeatEQResult.dat')
data = []

for line in fin:
    ar = [float(n) for n in line.split()]

    if len(ar)<50:
        xax = np.linspace(0,ar[0],ar[1])
    else:
        data.append(ar)


print len(xax)
print len(data)
print len(data[0])
print len(data[1])


plt.plot(xax,data[0][1:])
plt.hold
plt.plot(xax,data[1][1:])
plt.hold
plt.plot(xax,data[2][1:])
plt.grid()
plt.show()
