# -*- coding: utf-8 -*-

# Just writing a plotting script for the Swept rule CUDA.
# Perhaps this will also be the calling script.

import matplotlib.pyplot as plt
import numpy as np
import subprocess as sp
import shlex
import os
from Tkinter import *

OPTIONS = [
    "KS",
    "Heat",
    "Euler"
]

master = Tk()

master.geometry("400x400")

variable = StringVar(master)
variable.set(OPTIONS[1]) # default value

w = apply(OptionMenu, (master, variable) + tuple(OPTIONS))
w.pack()

def ok():
    master.destroy()

def skip():
    master.destroy()

def on_closing():
    raise SystemExit

master.protocol("WM_DELETE_WINDOW", on_closing)
button = Button(master, text="OK", command=ok)
button.pack()
button = Button(master, text="Skip Run", command=skip)
button.pack()

master.mainloop()

Fname = variable.get()

timeout = '1D_Timing.txt'
rsltout = '1D_Result.dat'
sourcebase = '1D_SweptShared.cu'

sourcepath = os.path.dirname(__file__)
basepath = os.path.join(sourcepath,'Results')
ofile = Fname + timeout
filepath = os.path.abspath(os.path.join(basepath, ofile))
if os.path.isfile(filepath):
    os.remove(filepath)

div = [2**k for k in range(11,14)]
blx = [32,64,128,256,512,1024]
fn = open(filepath,'a+')

ExecL = './bin/' + Fname + 'Out'

compStr = 'nvcc -o ' + ExecL + ' ' + Fname + sourcebase + ' -gencode arch=compute_35,code=sm_35 -lm -w -std=c++11'
compArg = shlex.split(compStr)
proc = sp.Popen(compArg)
sp.Popen.wait(proc)
print "Compiled"
fn.write("BlockSize\tXDimSize\tTime\n")
fn.close()

for k in blx:
    for n in div:
        print n,k
        execut = ExecL + ' {0} {1} {2} {3} {4}'.format(n,k,.01,10000,0)
        exeStr = shlex.split(execut)
        proc = sp.Popen(exeStr)
        sp.Popen.wait(proc)

rslt = Fname + rsltout
rsltfile = os.path.abspath(os.path.join(basepath, rslt))
fin = open(rsltfile)
data = []

for line in fin:
    ar = [float(n) for n in line.split()]

    if len(ar)<50:
        xax = np.linspace(0,ar[0],ar[1])
    else:
        data.append(ar)




lbl = ["Initial Condition"]

plt.plot(xax,data[0][1:])
plt.hold
for k in range(1,len(data)):
    plt.plot(xax,data[k][1:])
    lbl.append("t = {} seconds".format(data[k][0]))

plt.legend(lbl)
plt.xlabel("Position on bar (m)")
plt.ylabel("Velocity")
plt.title(Fname + execut[len(ExecL):])
plt.grid()
plt.show()
