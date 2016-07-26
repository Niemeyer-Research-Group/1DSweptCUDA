# -*- coding: utf-8 -*-

# Just writing a plotting script for the Swept rule CUDA.
# Perhaps this will also be the calling script.

import matplotlib.pyplot as plt
import numpy as np
import subprocess as sp
import shlex
from Tkinter import *

OPTIONS = [
    "KS",
    "Heat",
    "Euler"
]

master = Tk()

master.geometry("400x400")

variable = StringVar(master)
variable.set(OPTIONS[2]) # default value

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

fname = variable.get()

execut = "./bin/"+fname+"Out"

div = 128
bks = 64
dt = .02
tf = .04
tst = 0

execstr = execut +  ' {0} {1} {2} {3} {4}'.format(div,bks,dt,tf,tst)
exeStr = shlex.split(execstr)
proc = sp.Popen(exeStr)
sp.Popen.wait(proc)


fin = open("Results/" +fname + "1D_Result.dat")
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
plt.ylabel("Vel")
plt.title(fname+execstr[len(execut):])
plt.grid()
plt.show()
