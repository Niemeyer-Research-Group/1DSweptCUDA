# -*- coding: utf-8 -*-

'''
This file is the current iteration of research being done to implement the
swept rule for Partial differential equations in one dimension.  This research
is a collaborative effort between teams at MIT, Oregon State University, and
Purdue University.

Copyright (C) 2015 Kyle Niemeyer, niemeyek@oregonstate.edu AND
Daniel Magee, mageed@oregonstate.edu

This program is free software: you can redistribute it and/or modify
it under the terms of the MIT license.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

You should have received a copy of the MIT license along with this program.
If not, see <https://opensource.org/licenses/MIT>.
'''

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
    "Heat",
    "KS",
    "Euler"
]

#This is a very long python script.
master = Tk.Tk()

dropframe = Tk.Frame(master, pady = 2)
entryframe = Tk.Frame(master, pady = 1)
endframe = Tk.Frame(master, pady = 2)
dropframe.pack()
endframe.pack(side = 'bottom')
entryframe.pack(side = 'bottom')
master.title("Swept Rule 1-D GPU performance test")

problem = Tk.StringVar(master)
problem.set(OPTIONS[0]) # default value

#Number of divisions power of two
divpow = Tk.IntVar(master)
divpow.set(11)

divpowend = Tk.IntVar(master)
divpowend.set(20)

#Threads per block.
blkpow = Tk.IntVar(master)
blkpow.set(5)

blkpowend = Tk.IntVar(master)
blkpowend.set(10)

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

runit = Tk.BooleanVar(master)
runit.set(True)

def ok():
    master.destroy()

def ret(event):
    master.destroy()

def replot():
    runit.set(False)
    master.destroy()

def on_closing():
    raise SystemExit

def reset_vals(problem):
    if problem == 'KS':
        deltat.set(0.005)
        t_final.set(100)

    elif problem == 'Euler':
        deltat.set(0.02)
        t_final.set(100)
    else:
        deltat.set(0.01)
        t_final.set(500)

    fq.set(t_final.get()*2.0)

def reset_label(event):
    res_one.config(text = str(2**divpow.get()))
    res_two.config(text = str(2**blkpow.get()))
    res_three.config(text = str(2**divpowend.get()))
    res_four.config(text = str(2**blkpowend.get()))

master.protocol("WM_DELETE_WINDOW", on_closing)
master.bind('<Return>', ret)

Tk.Checkbutton(entryframe, text = "Swept Scheme", variable = sw).grid(row = 9, column = 0)
Tk.Checkbutton(entryframe, text = "CPU/GPU sharing \n(Not available on KS equation)", variable = proc_share).grid(row = 10, column = 0)

Tk.Label(entryframe, text= "Number of divisions: 2^").grid(row=1, column = 0)
div_one = Tk.Entry(entryframe, textvariable=divpow)
div_one.grid(row = 1, column = 1)

Tk.Label(entryframe, text= "Threads per block: 2^").grid(row=3, column = 0)
blk_one = Tk.Entry(entryframe, textvariable=blkpow)
blk_one.grid(row = 3, column = 1)

Tk.Label(entryframe, text= " to: 2^").grid(row=1, column = 2)
div_two = Tk.Entry(entryframe, textvariable=divpowend)
div_two.grid(row = 1, column = 3)

Tk.Label(entryframe, text= " to: 2^").grid(row=3, column = 2)
blk_two = Tk.Entry(entryframe, textvariable=blkpowend)
blk_two.grid(row = 3, column = 3)

Tk.Label(entryframe, text= u"\N{GREEK CAPITAL LETTER DELTA}"+"t (seconds): ").grid(row=5, column = 0)
Tk.Entry(entryframe, textvariable=deltat).grid(row = 5, column = 1)

Tk.Label(entryframe, text= "Stopping time (seconds): ").grid(row=6, column = 0)
Tk.Entry(entryframe, textvariable=t_final).grid(row = 6, column = 1)

Tk.Label(entryframe, text= "Output frequency (seconds): ").grid(row=7, column = 0)
Tk.Entry(entryframe, textvariable=fq).grid(row = 7, column = 1)

res_one = Tk.Label(entryframe, text = str(2**divpow.get()), anchor = Tk.W)
res_one.grid(row = 2, column = 1)
res_two = Tk.Label(entryframe, text = str(2**blkpow.get()), anchor = Tk.W)
res_two.grid(row = 4, column = 1)
res_three = Tk.Label(entryframe, text = str(2**divpowend.get()))
res_three.grid(row = 2, column = 3)
res_four = Tk.Label(entryframe, text = str(2**blkpowend.get()), anchor = Tk.W)
res_four.grid(row = 4, column = 3)

master.bind_class("Entry","<FocusOut>", reset_label)

button_send = Tk.Button(endframe, text="OK", command=ok)
button_send.grid(row = 0, column = 0)
button_sk = Tk.Button(endframe, text="REPLOT W/O RUNNING", command=replot)
button_sk.grid(row = 0, column = 1)
problem_menu = Tk.OptionMenu(dropframe, problem, *OPTIONS, command=reset_vals)
problem_menu.grid()

reset_vals(problem.get())

master.mainloop()

## -------Tkinter End----------

Fname = problem.get()
dt = deltat.get()
tf = t_final.get()
freq = fq.get()
swept = int(sw.get())
cpu = int(proc_share.get())

timeout = '1D_Timing.txt'
rsltout = '1D_Result.dat'

if swept and cpu:
    timestr = Fname + "_Swept_CPU_Sharing"
elif swept:
    timestr = Fname + "_Swept_GPU_only"
else:
    timestr = Fname + "_Classic"


print timestr
print dt, tf, freq, swept, cpu

sourcepath = os.path.abspath(os.path.dirname(__file__))
basepath = os.path.join(sourcepath,'Results')

binpath = os.path.join(sourcepath,'bin')

#Ensures "make" won't fail if there's no bin directory.
if not os.path.isdir(binpath):
    os.mkdir(binpath)

if not os.path.isdir(basepath):
    os.mkdir(basepath)

timer = timestr + timeout
rslt = Fname + rsltout
timefile = os.path.join(basepath, timer)
rsltfile = os.path.join(basepath, rslt)

if runit:
    #Reset timing file.
    cyc = 0
    if os.path.isfile(timefile):
        os.remove(timefile)

    div = [2**k for k in range(divpow.get(),divpowend.get()+1)]
    blx = [2**k for k in range(blkpow.get(),blkpowend.get()+1)]
    t_fn = open(timefile,'a+')

    ExecL = './bin/' + Fname + 'Out'

    sp.call("make")

    t_fn.write("XDimSize\tBlockSize\tTime\n")
    t_fn.close()

    for k in blx:
        for n in div:
            if swept:
                if Fname == "Heat":
                    cyc = int(tf/(k*dt))
                else:
                    cyc = int(tf/(k*dt*.25))
            print "---------------------------"
            print n, k, cyc
            execut = ExecL +  ' {0} {1} {2} {3} {4} {5} {6} {7} {8}'.format(n,k,dt,tf,freq,swept,cpu,rsltfile,timefile)
            exeStr = shlex.split(execut)
            proc = sp.Popen(exeStr)
            sp.Popen.wait(proc)
            print "---------------------------"

fin = open(rsltfile)
data = []

for line in fin:
    ar = [float(n) for n in line.split()]

    if len(ar)<50:
        xax = np.linspace(0,ar[0],ar[1])
    else:
        data.append(ar)


# lbl = ["Initial Condition"]
#
# plt.plot(xax,data[0][1:])
# plt.hold
# for k in range(1,len(data)):
#     plt.plot(xax,data[k][1:])
#     lbl.append("t = {} seconds".format(data[k][0]))
#
# plt.legend(lbl)
# plt.xlabel("Position on bar (m)")
# plt.ylabel("Velocity")
# plt.title(Fname + execut[len(ExecL):])
# plt.grid()

#Timing Show
if not os.path.isfile(timefile):
    print "There is no file for the specified procedure: " + timestr
    os.exit(-1)

times = pd.read_table(timefile, delim_whitespace = True)
headers = times.columns.values.tolist()
time_split = times.pivot(headers[0],headers[1],headers[2])
time_split.plot(logx = True, grid = True)
plt.ylabel("Time per timestep (us)")
plt.title(timestr)
plt.show()
