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

import os
import os.path as op
import sys

sourcepath = op.abspath(op.dirname(__file__))
rsltpath = op.join(sourcepath,'Results')
binpath = op.join(sourcepath,'bin') #Binary directory
gitpath = op.dirname(sourcepath) #Top level of git repo
plotpath = op.join(op.join(gitpath,"ResultPlots"),"performance") #Folder for plots
modpath = op.join(gitpath,"pyAnalysisTools")
os.chdir(sourcepath)

sys.path.append(modpath)
import main_help as mh

import numpy as np
import subprocess as sp
import shlex
import Tkinter as Tk
import ttk

OPTIONS = [
    "Heat",
    "Euler",
    "KS"
]

SCHEME = [
    "ClassicGPU",
    "SharedGPU",
    "Hybrid"
]

PRECISION = [
    "Single",
    "Double"
]

if len(sys.argv) < 2:

    #GUI start
    master = Tk.Tk()

    dropframe = Tk.Frame(master, pady = 2)
    entryframe = Tk.Frame(master, pady = 1)
    endframe = Tk.Frame(master, pady = 2)
    errframe = Tk.Frame(master)

    dropframe.pack()
    endframe.pack(side='bottom')
    entryframe.pack(side='bottom')
    errframe.pack(side='top')
    errlbl = ttk.Label(errframe, text="There is no CPU share scheme for the KS equation")
    errlbl.pack()
    master.title("Swept Rule 1-D GPU performance test")

    problem = Tk.StringVar(master)
    problem.set(OPTIONS[2]) # default value

    alg = Tk.StringVar(master)
    alg.set(SCHEME[2]) # default value

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

    #cycles
    steps = Tk.IntVar(master)
    steps.set(500000)
    prec = Tk.BooleanVar(master)
    prec.set(False)

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

    def reset_vals(p):
        if OPTIONS.index(problem.get()) == 2 and SCHEME.index(alg.get()) == 2:
            errframe.pack(side='top')
        else:
            try:
                errframe.pack_forget()
            except:
                pass


    def reset_label(event):
        res_one.config(text=str(2**divpow.get()))
        res_three.config(text=str(2**divpowend.get()))
        res_two.config(text=str(2**blkpow.get()))
        res_four.config(text=str(2**blkpowend.get()))

    master.protocol("WM_DELETE_WINDOW", on_closing)
    master.bind('<Return>', ret)

    Tk.Checkbutton(entryframe, text="Double Precision", variable=prec).grid(row = 8, column = 0)

    Tk.Label(entryframe, text="Number of divisions: 2^").grid(row=1, column = 0)
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

    Tk.Label(entryframe, text= "Number of timesteps: ").grid(row=6, column = 0)
    Tk.Entry(entryframe, textvariable=steps).grid(row = 6, column = 1)
    Tk.Label()

    res_one = Tk.Label(entryframe, text = str(2**divpow.get()))
    res_one.grid(row = 2, column = 1)
    res_two = Tk.Label(entryframe, text = str(2**blkpow.get()))
    res_two.grid(row = 4, column = 1)
    res_three = Tk.Label(entryframe, text = str(2**divpowend.get()))
    res_three.grid(row = 2, column = 3)
    res_four = Tk.Label(entryframe, text = str(2**blkpowend.get()))
    res_four.grid(row = 4, column = 3)

    master.bind_class("Entry","<FocusOut>", reset_label)

    button_send = Tk.Button(endframe, text="OK", command=ok)
    button_send.grid(row = 0, column = 0)
    button_sk = Tk.Button(endframe, text="REPLOT W/O RUNNING", command=replot)
    button_sk.grid(row = 0, column = 1)

    problem_menu = Tk.OptionMenu(dropframe, problem, *OPTIONS, command=reset_vals)
    problem_menu.grid(row=0, column=0)

    alg_menu = Tk.OptionMenu(dropframe, alg, *SCHEME, command=reset_vals)
    alg_menu.grid(row=0, column=1)

    reset_vals(alg.get())

    master.mainloop()

## -------Tkinter End----------

    fname = problem.get()
    precise = PRECISION[int(prec.get())]
    sch = alg.get()
    algs = SCHEME.index(sch)
    dv1 = divpow.get()
    dv2 = divpowend.get()
    bk1 = blkpow.get()
    bk2 = blkpowend.get()
    st = steps.get()
    runb = runit.get()

else:
    print sys.argv
    fname = sys.argv[1]
    precise = sys.argv[2]
    sch = SCHEME[int(sys.argv[3])]
    algs = int(sys.argv[3])
    dv1 = int(sys.argv[4])
    dv2 = int(sys.argv[5])
    bk1 = int(sys.argv[6])
    bk2 = int(sys.argv[7])
    st = float(sys.argv[8])
    runb = True

prob_idx = OPTIONS.index(fname)

swept = int(bool(schsym))

div = [2**k for k in range(dv1, dv2+1)]
blx = [2**k for k in range(bk1, bk2+1)]

timeout = '_Timing.txt'
rsltout = '_Result.dat'
timename = fname + "_" + precise + "_" + sch
binf = fname + precise + 'Out'
vf = fname + precise + rsltout
timefile = timename + timeout
plotstr = timename.replace("_"," ")

timepath = op.join(rsltpath,timefile)
Varfile = op.join(rsltpath,vf)

#Standard testing runs for these algorithms
dts = [.001, float(format(1.0/div[-1],'.0e'))*.1, .005]
tfs = [st*m for m in dts]
dt, tf = dts[prob_idx], tfs[prob_idx]

errchk = False

#Ensures "make" won't fail if there's no bin directory.
if not op.isdir(binpath):
    os.mkdir(binpath)

if not op.isdir(rsltpath):
    os.mkdir(rsltpath)

t_fn = open(timepath,'w')

ExecL = op.join(binpath,binf)

sp.call("make")

#Parse it out afterward.
t_fn.write("Num_Spatial_Points\tThreads_per_Block\tTime_per_timestep_(us)\n")
t_fn.close()

mh.runCUDA(ExecL, div, blx, dt, tf, tf*2.0, algs, Varfile, timepath)
print div, blx

myFrame = mh.Perform(timepath)
myFrame.plotframe(plotpath, false)
