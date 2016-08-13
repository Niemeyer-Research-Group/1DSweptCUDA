# -*- coding: utf-8 -*-

# Just writing a plotting script for the Swept rule CUDA.
# Perhaps this will also be the calling script.

import matplotlib.pyplot as plt
import numpy as np
import subprocess as sp
import shlex
import os
import Tkinter as Tk

OPTIONS = [
    "KS",
    "Heat",
    "Euler"
]

#It's kinda getting there.
master = Tk.Tk()

dropframe = Tk.Frame(master)
entryframe = Tk.Frame(master)
endframe = Tk.Frame(master)
dropframe.pack()
endframe.pack(side = 'bottom')
entryframe.pack(side = 'bottom')

problem = Tk.StringVar(master)
problem.set(OPTIONS[1]) # default value

runit = Tk.BooleanVar(master)
runit.set(True)

def ok():
    master.destroy()

def replot():
    runit.set(False)
    master.destroy()

def on_closing():
    raise SystemExit

master.protocol("WM_DELETE_WINDOW", on_closing)
button_send = Tk.Button(endframe, text="OK", command=ok)
button_send.grid(row = 0, column = 0)
button_sk = Tk.Button(endframe, text="SKIP RUN", command=replot)
button_sk.grid(row = 0, column = 1)
problem_menu = Tk.OptionMenu(dropframe, problem, *OPTIONS)
problem_menu.grid(column=1, columnspan = 4)

fq.set(t_final.get()*2)

#Number of divisions power of two
divpow = Tk.IntVar(master)
divpow.set(12)

#Threads per block.
blkpow = Tk.IntVar(master)
blkpow.set(8)

#dt
deltat = Tk.DoubleVar(master)

#tf
t_final = Tk.DoubleVar(master)

#freq
fq = Tk.DoubleVar(master)

#Swept or classic
sw = Tk.BooleanVar(master)

#CPU or no
proc_share = Tk.BooleanVar(master)

if problem == 'KS':
    deltat.set(0.005)
    t_final.set(1000)
elif problem == 'Euler':
    deltat.set(0.02)
    t_final.set(100)
else:
    deltat.set(0.02)
    t_final.set(1000)

fq.set(t_final.get()*2)

check_one = Tk.Checkbutton(entryframe, text = "Swept Scheme ?", variable = sw)
check_two = Tk.Checkbutton(entryframe, text = "CPU/GPU sharing ?", variable = proc_share)

check_one.grid(row = 6, column = 0)
check_two.grid(row = 7, column = 0)

Tk.Label(entryframe, text="Number of divisions: 2^").grid(row=0, column = 0)
Tk.Entry(entryframe, textvariable=divpow).grid(row = 0, column = 1)

master.mainloop()

##Interface end

Fname = problem.get()

sourcepath = os.path.abspath(os.path.dirname(__file__))

Varfile = os.path.join(sourcepath, Fname + "1D_Result.dat")

if runit.get():
    sp.call("make")

    execut = "./bin/"+ Fname+ "Out"

    div = 2**divpow.get()
    print div
    bks = 2**blkpow.get()
    dt = deltat.get() #timestep in seconds
    tf = t_final.get() #timestep in seconds
    freq = fq.get()
    swept = sw.get()
    cpu = proc_share.get()

    if swept and cpu:
        print Fname + " Swept CPU Sharing"
    elif swept:
        print Fname + " Swept GPU only"
    else:
        print Fname + " Classic"

    execstr = execut +  ' {0} {1} {2} {3} {4} {5} {6} {7}'.format(div,bks,dt,tf,freq,swept,cpu,Varfile)

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

print data[1][5]
plt.plot(xax,data[0][1:])
plt.hold(True)
for k in range(1,len(data)):
    plt.plot(xax,data[k][1:],linewidth = 2.0, marker = '.', markersize = 10)
    lbl.append("t = {} seconds".format(data[k][0]))
    plt.hold(True)


plt.legend(lbl)
plt.xlabel("Position on bar (m)")
plt.ylabel("Vel")
plt.title(Fname)
plt.grid()
plt.show()
