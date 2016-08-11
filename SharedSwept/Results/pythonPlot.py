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

def on_closing():
    raise SystemExit

master.protocol("WM_DELETE_WINDOW", on_closing)
button = Button(master, text="OK", command=ok)
button.pack()

master.mainloop()

Fname = variable.get()

execut = "./bin/"+ Fname+ "Out"

div = 2**10
print div
bks = 128
dt = 0.02 #timestep in seconds
tf = 1000 #timestep in seconds
swept = 1
cpu = 1

sourcepath = os.path.abspath(os.path.dirname(__file__))

Varfile = os.path.join(sourcepath, Fname + "1D_Result.dat")

if swept and cpu:
    print Fname + " Swept CPU Sharing"
elif swept:
    print Fname + " Swept GPU only"
else:
    print Fname + " Classic"

execstr = execut +  ' {0} {1} {2} {3} {4} {5} {6} {7}'.format(div,bks,dt,tf,tf*2,swept,cpu,Varfile)

exeStr = shlex.split(execstr)
proc = sp.Popen(exeStr)
sp.Popen.wait(proc)

fin = open(Varfile)
data = []

for line in fin:
    ar = [float(n) for n in line.split()]

    if len(ar)<50:
        xax = np.linspace(0,1,ar[1])
    else:
        data.append(ar)

lbl = ["Initial Condition"]

plt.plot(xax,data[0][1:])
plt.hold(True)
for k in range(1,len(data)):
    plt.plot(xax,data[k][1:])
    lbl.append("t = {} seconds".format(data[k][0]))
    plt.hold(True)


plt.legend(lbl)
plt.xlabel("Position on bar (m)")
plt.ylabel("Vel")
plt.title(Fname)
plt.grid()
plt.show()
