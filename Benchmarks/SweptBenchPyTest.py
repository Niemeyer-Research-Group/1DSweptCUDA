# -*- coding: utf-8 -*-

# Just writing a plotting script for the Swept rule CUDA.
# Perhaps this will also be the calling script.

import matplotlib.pyplot as plt
import numpy as np
import subprocess as sp
import shlex
import os

choose = int(raw_input("Enter 1 for GPU and 0 for CPU: "));

if choose:
    ofile = 'GPUBenchTiming.txt'
else:
    ofile = 'CPUBenchTiming.txt'

if os.path.isfile(ofile):
    os.remove(ofile)

basepath = os.path.dirname(__file__)
filepath = os.path.abspath(os.path.join(basepath, ofile))

div = [2**k for k in range(11,19)]
blx = [32,64,128,256,512,1024]
nums = range(0,9,2)
nums[0] = 1
uni = ["-DUNIFIED=1", ""]

if choose:
    execut = ['./GPUBenchOut']
    for u in uni:
        fn = open(filepath,'a+')
        fn.write("BlockSize\tXDimSize\tTime\n")
        fn.close()
        for k in blx:
            for n in div:
                fn = open(filepath,'a+')
                fn.write(str(k)+"\t"+str(n)+"\t")
                fn.close()
                compStr = "nvcc -DTHREADBLK={0} -DDIVISIONS={1} {2} -DFINISH=1e3 -o GPUBenchOut NaiveGPUPDE1D.cu -gencode arch=compute_35,code=sm_35 -lm -w -std=c++11".format(k,n,u)
                compArg = shlex.split(compStr)
                proc = sp.Popen(compArg)
                sp.Popen.wait(proc)
                print "Compiled"
                proc = sp.Popen(execut)
                sp.Popen.wait(proc)

else:
    execut = ['./CPUBenchOut']
    fn = open(filepath,'a+')
    fn.write("ArraySize\tThreadCount\tTime\n")
    fn.close()
    for d in div:
        for n in nums:
            fn = open(filepath,'a+')
            fn.write(str(d)+"\t"+str(n)+"\t")
            fn.close()
            compStr = "g++ -DDIVISIONS={0} -DNUMT={1} -DFINISH=1e3 -o CPUBenchOut ParaCPUPDE1D.cpp -lm -fopenmp -w -O3".format(d,n)
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
plt.title("Numerical solution to temperature along 1-D bar:" + execut[0][2:5])
plt.grid()
plt.show()
