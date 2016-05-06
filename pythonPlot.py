# -*- coding: utf-8 -*-

# Just writing a plotting script for the Swept rule CUDA.
# Perhaps this will also be the calling script.

import matplotlib.pyplot as plt
import numpy as np
import subprocess as sp
import shlex

c = int(raw_input("Do you want to compile it 1 for yes, 0 for no: ")) 

if c:
    compStr = "nvcc 1DSweptRule_mainregister.cu -o SweptOut -gencode arch=compute_35,code=sm_35 -lm -std=c++11"
    execut = ['./SweptOut']
    compArg = shlex.split(compStr)
    proc = sp.Popen(compArg)
    sp.Popen.wait(proc)
    print "Compiled"
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


print "Percent Difference in integrals:"
df = 100*abs(np.trapz(data[0][1:],xax)-np.trapz(data[1][1:],xax))/np.trapz(data[0][1:],xax)
print df

lbl = ["Initial Condition"]

plt.plot(xax,data[0][1:])
plt.hold
for k in range(1,len(data)):
    plt.plot(xax,data[k][1:])
    lbl.append("t = {} seconds".format(data[k][0]))
    

plt.legend(lbl)
plt.xlabel("Position on bar (m)")
plt.ylabel("Temperature (C)")
plt.title("Numerical solution to temperature along 1-D bar")
plt.grid()
plt.show()
