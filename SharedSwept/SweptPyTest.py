# -*- coding: utf-8 -*-

# A script to test the performance of the algorithm with various
# x dimension and block sizes

import matplotlib.pyplot as plt
import numpy as np
import subprocess as sp
import shlex
import os
import Tkinter as Tk
import pandas as pd

OPTIONS = [
    "KS",
    "Heat",
    "Euler"
]

master = Tk.Tk()

master.geometry("400x400")

variable = Tk.StringVar(master)
variable.set(OPTIONS[0]) # default value

w = apply(Tk.OptionMenu, (master, variable) + tuple(OPTIONS))
w.pack()

def ok():
    master.destroy()

def skip():
    master.destroy()

def on_closing():
    raise SystemExit

master.protocol("WM_DELETE_WINDOW", on_closing)
button = Tk.Button(master, text="OK", command=ok)
button.pack()
button = Tk.Button(master, text="Skip Run", command=skip)
button.pack()

master.mainloop()

Fname = variable.get()

timeout = '1D_Timing.txt'
rsltout = '1D_Result.dat'
sourcebase = '1D_SweptShared.cu'

sourcepath = os.path.dirname(__file__)
basepath = os.path.join(sourcepath,'Results')
tfile = Fname + timeout
t_filepath = os.path.abspath(os.path.join(basepath, tfile))

#Reset timing file.
if os.path.isfile(t_filepath):
    os.remove(t_filepath)

div = [2**k for k in range(11,20)]
blx = [2**k for k in range(5,11)]
t_fn = open(t_filepath,'a+')

dt = .005
tf = 1000
tst = 1

ExecL = './bin/' + Fname + 'Out'

compStr = 'nvcc -o ' + ExecL + ' ' + Fname + sourcebase + ' -gencode arch=compute_35,code=sm_35 -lm -Xcompiler -fopenmp -w -std=c++11'
compArg = shlex.split(compStr)
proc = sp.Popen(compArg)
sp.Popen.wait(proc)
print "Compiled ", proc
t_fn.write("BlockSize\tXDimSize\tTime\n")
t_fn.close()

for k in blx:
    for n in div:
        print n,k
        execut = ExecL + ' {0} {1} {2} {3} {4} {5}'.format(n,k,dt,tf,tst,tf*2)
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

#Timing Show
times = pd.read_table(t_filepath)
 