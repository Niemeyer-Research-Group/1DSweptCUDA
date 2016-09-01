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
from math import *
import numpy as np
import subprocess as sp
import shlex
import os
import Tkinter as Tk
import pandas as pd
from RegressionTesting.exactcomparison import *

alpha = 8.418e-5
dx = 0.001

OPTIONS = [
    "KS",
    "Heat",
    "Euler"
]

Param = {"Heat": "Fo = ", "KS":"dt/dx = ", "Euler":"dt/dx = "}

OPT_PREC = [
    "Single",
    "Double"
]

#It's kinda getting there.
master = Tk.Tk()

dropframe = Tk.Frame(master, pady = 2)
entryframe = Tk.Frame(master, pady = 1,padx = 15)
endframe = Tk.Frame(master, pady = 2,padx = 15)
dropframe.pack()
endframe.pack(side = 'bottom')
entryframe.pack(side = 'bottom')
master.title("Plot the result of the numerical solution")

problem = Tk.StringVar(master)
problem.set(OPTIONS[1]) # default value

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
    res_one.config(text = str(2**divpow.get()))
    res_two.config(text = str(2**blkpow.get()))
    res_three.config(text = Param[problem.get()] + str(alpha*deltat.get()/(dx**2)))

def reset_vals(problem):
    if problem == 'KS':
        deltat.set(0.005)
        t_final.set(1000)
    elif problem == 'Euler':
        deltat.set(1e-5)
        t_final.set(0.25)
    else:
        deltat.set(0.01)
        t_final.set(200)

    fq.set(t_final.get()*2.0)
    reset_label(1)


master.protocol("WM_DELETE_WINDOW", on_closing)
master.bind('<Return>', ret)

check_one = Tk.Checkbutton(entryframe, text = "Swept Scheme ", variable = sw)
check_two = Tk.Checkbutton(entryframe, text = "CPU/GPU sharing ", variable = proc_share)
check_three = Tk.Checkbutton(entryframe, text = "Double Precision ", variable = prec)

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

res_three = Tk.Label(entryframe, text = Param[problem.get()] + str(alpha*deltat.get()/(dx**2)))
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
sourcepath = op.abspath(op.dirname(__file__))
basepath = op.join(sourcepath,'Results')
binpath = op.join(sourcepath,'bin')

#Ensures "make" won't fail if there's no bin directory.
if not op.isdir(binpath):
    os.mkdir(binpath)

if not op.isdir(basepath):
    os.mkdir(basepath)

Varfile = op.join(basepath, typename + "_1D_Result.dat")
gitpath = op.dirname(sourcepath)
gifpath = op.join(op.join(op.join(gitpath,'ResultPlots'),'performance'),typename+".gif")

div = 2**divpow.get()
bks = 2**blkpow.get()
dt = deltat.get() #timestep in seconds
tf = t_final.get() #timestep in seconds
freq = fq.get()
swept = int(sw.get())
cpu = int(proc_share.get())

SCHEME = [
    "Classic",
    "SweptGPU",
    "SweptCPUshare"
]

if swept and cpu:
    sch = SCHEME[2]
    timestr = Fname + " " + sch
elif swept:
    sch = SCHEME[1]
    timestr = Fname + " " + sch
else:
    sch = SCHEME[0]
    timestr = Fname + " " + sch

if runit.get():
    sp.call("make")

    execut = op.join(binpath, typename + "Out")

    print div, bks, dt, tf, freq, swept, cpu

    if swept:
        if Fname == "Heat":
            print "Number of cycles: {}".format(int(tf/(bks*dt)))
        else:
            print "Number of cycles: {}".format(int(tf/(bks*dt*.25)))

    print timestr

    execstr = execut +  ' {0} {1} {2} {3} {4} {5} {6} {7}'.format(div,bks,dt,tf,freq,swept,cpu,Varfile)

    exeStr = shlex.split(execstr)
    proc = sp.Popen(exeStr)
    sp.Popen.wait(proc)

f = open(Varfile)
fin = tuple(f)
ar = [float(n) for n in fin[0].split()]
xax = np.linspace(0,ar[0],ar[1])
data = []

for p in range(1,len(fin)):
    ar = fin[p].split()
    tm = float(ar[1])
    dMain.append([ar[0], [float(n) for n in ar[1:]]])

lbl = ["Initial Condition"]
simF = pd.DataFrame(dMain)
simF = simF.set_index(0)
typ = simF.index.get_level_values(0).unique()
cross = simF.xs( typ[0] )
cnt = len(cross.index)

if cnt < 6:
    if len(typ) == 1:
        fig, ax = plt.subplots((1,1),figsize=(14.,8.))
        ax = [ax]
        ax.set_title(plotstr + ' | {0} spatial points                    {1}'.format(int(div)," "), fontsize="medium")
    else:
        ax.ravel()
        fig, ax = plt.subplots((2,2),figsize=(14.,8.))
        plt.suptitle(plotstr + ' | {0} spatial points                    {1}'.format(int(div)," "), fontsize="medium")

    for i, ty in enumerate(typ):

        df_sim = simF.xs(ty)
        df_sim = df_sim.set_index(1)
        df_sim.columns = xax
        df_sim = df_sim.transpose()
        cl = df_sim.columns.values.tolist()
        df_sim.reset_index(inplace=True)

        ax[i].hold(True)
        #Check if title already exists.  That's probably not the most efficient way to do this.
        ax[i].set_title(ty, fontsize="medium")
        ax[i].grid(alpha=0.5)
        ax[i].set_xlabel("Spatial point")
        ax[i].plot(xax,data[k],linewidth = 2.0)
        ax[i].set_ylabel(ty)

        for tfs in cl:
            ax[i].plot(df_sim['index'], df_sim[tfs], label="{:.2f} (s)".format(tfs))

    plt.show()

#Don't do this.  Animate!  A gif?
else:
    simF.reset_index(inplace=True)
    #Now were indexing by time rather than type.
    #import Figtodat and images2gif library then append the fig
    #to dat with .fig2img(fig) and append it to images then call
    #writeGif.  Make figsize big!  view with xnview.  That's gifpath.
    simF.set_index(1)
    if len(typ) == 1:
        fig, ax = plt.subplots((1,1),figsize=(14.,8.))
        ax = [ax]
        ax.set_title(plotstr + ' | {0} spatial points                    {1}'.format(int(div)," "), fontsize="medium")
    else:
        ax.ravel()
        fig, ax = plt.subplots((2,2),figsize=(14.,8.))
        plt.suptitle(plotstr + ' | {0} spatial points

    ax = fig.add_subplot(111, projection='3d')
    dop = np.array(data)
    X,Y = np.meshgrid(xax,np.transpose(dop[:,1]))
    print dop.shape, X.shape
    Z = dop[:,1:]
    print Z.shape
    ax.plot_surface(X,Y,Z)
    plt.title(timestr + ": " + str(div) + " points")
    plt.hold(True)
    plt.show()


# for k in range(1,len(data)):
#     rt.consistency_test(Fname,sch,div,data[k][0],data[k][1:])
# #Maybe it's time for chdir and a function file.
