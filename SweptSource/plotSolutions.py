#!/usr/bin/env python
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

# Just writing a plotting script for the Swept rule CUDA.
# Perhaps this will also be the calling script.

import matplotlib.pyplot as plt
#from math import *
from cycler import cycler
import numpy as np
import subprocess as sp
import shlex
import os
import os.path as op
import sys
import Tkinter as Tk
import pandas as pd
import palettable.colorbrewer as pal
import time

sourcepath = op.abspath(op.dirname(__file__))
rsltpath = op.join(sourcepath,'Results')
binpath = op.join(sourcepath,'bin') #Binary directory
gitpath = op.dirname(sourcepath) #Top level of git repo
plotpath = op.join(op.join(gitpath,"ResultPlots"),"performance") #Folder for plots
modpath = op.join(gitpath,"pyAnalysisTools")
os.chdir(sourcepath)

sys.path.append(modpath)
import result_help as rh
import main_help as mh

alpha = 8.418e-5
dx = 0.001

OPTIONS = [
    "KS",
    "Heat",
    "Euler"
]

OPT_PREC = [
    "Single",
    "Double"
]

def heat_msg(div,dt,bks,tf,fr):
    dx = 0.001
    div = 2**div
    bks = 2**bks
    cyc = int(tf/(bks*dt))
    Fo = alpha*dt/(dx**2)
    stab = Fo<.5
    ot = int(tf/fr)+2
    if stab:
        return "STABLE \n Fo = {0}  |  nCycles: {1}  |  #Outputs: {2}".format(Fo, cyc, ot)
    else:
        return "UNSTABLE UNSTABLE \n Fo: {0} is too low.".format(Fo)

def ks_msg(div,dt,bks,tf,fr):
    dx = 0.5
    div = 2**div
    bks = 2**bks
    cyc = int(4*tf/(bks*dt))
    dtdx = dt/dx
    stab = dtdx<.015
    ot = int(tf/fr)+2
    if stab:
        return "STABLE \n dt/dx = {0}  |  nCycles: {1}  |  #Outputs: {2}".format(dtdx, cyc, ot)
    else:
        return "UNSTABLE UNSTABLE \n dt/dx: {0} is too low.".format(dtdx)

def euler_msg(div,dt,bks,tf,fr):
    div = 2**div
    dx = 1.0/(float(div-1.0))
    bks = 2**bks
    cyc = int(4*tf/(bks*dt))
    dtdx = dt/dx
    stab = dtdx<.05
    ot = int(tf/fr)+2
    if stab:
        return "STABLE \n dt/dx = {0}  |  nCycles: {1}  |  #Outputs: {2}".format(dtdx, cyc, ot)
    else:
        return "UNSTABLE UNSTABLE \n dt/dx: {0} is too low.".format(dtdx)

funs = [ks_msg, heat_msg, euler_msg]

master = Tk.Tk()

dropframe = Tk.Frame(master, pady=2)
entryframe = Tk.Frame(master, pady=1 ,padx=15)
endframe = Tk.Frame(master, pady=2, padx=15)
dropframe.pack()
endframe.pack(side='bottom')
entryframe.pack(side='bottom')
master.title("Plot the result of the numerical solution")

problem = Tk.StringVar(master)
problem.set(OPTIONS[2]) # default value

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
prec = Tk.BooleanVar(master)

runit = runit = Tk.BooleanVar(master)
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

def reset_label(event):
    res_one.config(text=str(2**divpow.get()))
    res_two.config(text=str(2**blkpow.get()))
    res_three.config(text=funs[OPTIONS.index(problem.get())](divpow.get(), deltat.get(), blkpow.get(), t_final.get(),fq.get()))

def reset_vals(problem):
    if problem == 'KS':
        deltat.set(0.005)
        t_final.set(1000)
    elif problem == 'Euler':
        deltat.set(1e-5)
        t_final.set(0.25)
    else:
        deltat.set(0.001)
        t_final.set(100)

    fq.set(t_final.get()*2.0)
    reset_label(1)


master.protocol("WM_DELETE_WINDOW", on_closing)
master.bind('<Return>', ret)

check_one = Tk.Checkbutton(entryframe, text="Swept Scheme ", variable=sw)
check_two = Tk.Checkbutton(entryframe, text="Alternative Swept ", variable=proc_share)
check_three = Tk.Checkbutton(entryframe, text="Double Precision ", variable=prec)

check_one.grid(row = 9, column = 0)
check_two.grid(row = 10, column = 0)
check_three.grid(row = 9, column = 1)

# Just have one update routine and update for all changes.

Tk.Label(entryframe, text= "Number of divisions: 2^").grid(row=1, column = 0)
div_ent = Tk.Entry(entryframe, textvariable=divpow)
div_ent.grid(row = 1, column = 1)
res_one = Tk.Label(entryframe, text = str(2**divpow.get()), anchor = Tk.W)
res_one.grid(row = 2, column = 1)
master.bind_class("Entry","<FocusOut>", reset_label)

Tk.Label(entryframe, text= "Threads per block: 2^").grid(row=3, column = 0)
blk_ent = Tk.Entry(entryframe, textvariable=blkpow)
blk_ent.grid(row = 3, column = 1)
res_two = Tk.Label(entryframe, text = str(2**blkpow.get()), anchor = Tk.W)
res_two.grid(row = 4, column = 1)

Tk.Label(entryframe, text= u"\N{GREEK CAPITAL LETTER DELTA}"+"t (seconds): ").grid(row=5, column = 0)
Tk.Entry(entryframe, textvariable=deltat).grid(row = 5, column = 1)

Tk.Label(entryframe, text= "Stopping time (seconds): ").grid(row=6, column = 0)
Tk.Entry(entryframe, textvariable=t_final).grid(row = 6, column = 1)

Tk.Label(entryframe, text= "Output frequency (seconds): ").grid(row=7, column = 0)
Tk.Entry(entryframe, textvariable=fq).grid(row = 7, column = 1)
res_three = Tk.Label(entryframe)
res_three.grid(row = 11, column = 0, columnspan = 2)

button_send = Tk.Button(endframe, text="OK", command=ok)
button_send.grid(row = 0, column = 0)
button_sk = Tk.Button(endframe, text="REPLOT W/O RUNNING", command=replot)
button_sk.grid(row = 0, column = 1)
problem_menu = Tk.OptionMenu(dropframe, problem, *OPTIONS, command=reset_vals)
problem_menu.grid()

reset_vals(problem.get())

master.mainloop()

##Interface end --------------------

precision = ["Single", "Double"]

op = os.path

Fname = problem.get()
typename = Fname + precision[prec.get()]

#Ensures "make" won't fail if there's no bin directory.
if not op.isdir(binpath):
    os.mkdir(binpath)

if not op.isdir(rsltpath):
    os.mkdir(rsltpath)

Varfile = op.join(rsltpath, typename + "_Result.dat")
respath = op.join(gitpath,'ResultPlots')
pltpath = op.join(respath,'SimResults')
gifpath = op.join(respath,'Gifs')
temppath = op.join(gifpath,"Temp")

if not op.isdir(temppath):
    os.mkdir(temppath)

avifile = op.join(temppath,typename+".avi")

div = 2**divpow.get()
bks = 2**blkpow.get()
dt = deltat.get() #timestep in seconds
tf = t_final.get() #timestep in seconds
freq = fq.get()
alg = int(sw.get()) + int(proc_share.get())

SCHEME = [
    "Classic",
    "GPUShared",
    "Alternative"
]

sch = SCHEME[alg]
timestr = Fname + " " + sch

if runit.get():
    
    print "---------------------"
    print "Algorithm #divs #tpb dt endTime"
    print sch, div, bks, dt, tf

    ExecL = op.join(binpath, typename + "Out")

    sp.call("make")

    mh.runCUDA(ExecL, div, bks, dt, tf, freq, alg, Varfile)
    print div, bks

road = rh.Solved(Varfile)
tst = "Euler" in Varfile

if tst:
    fh, a = plt.subplots(2, 2, figsize=(11,7))
    a = a.ravel()
else:
    fh, a = plt.subplots(1, 1, figsize=(11,7))

if road.tFinal.size > 10:
    road.plotResult(fh, a)
    road.annotatePlot(fh,a)
    plt.show()
    road.savePlot(fh, pltpath)
else:
    road.gifify(gifpath, fh, a)
