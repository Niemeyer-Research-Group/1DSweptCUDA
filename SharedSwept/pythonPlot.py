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

dropframe = Tk.Frame(master, pady = 2)
entryframe = Tk.Frame(master, pady = 1)
endframe = Tk.Frame(master, pady = 2)
dropframe.pack()
endframe.pack(side = 'bottom')
entryframe.pack(side = 'bottom')
master.title("Plot the result of the numerical solution")

problem = Tk.StringVar(master)
roblem.set(OPTIONS[1]) # default value

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

fq.set(t_final.get()*2.0)

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
        t_final.set(1000)
    elif problem == 'Euler':
        deltat.set(0.02)
        t_final.set(100)
    else:
        deltat.set(0.02)
        t_final.set(1000)

def reset_label(event):
    res_one.config(text = str(2**divpow.get()))
    res_two.config(text = str(2**blkpow.get()))    
    
        
master.protocol("WM_DELETE_WINDOW", on_closing)
master.bind('<Return>', ret)

check_one = Tk.Checkbutton(entryframe, text = "Swept Scheme ", variable = sw)
check_two = Tk.Checkbutton(entryframe, text = "CPU/GPU sharing ", variable = proc_share)

check_one.grid(row = 9, column = 0)
check_two.grid(row = 10, column = 0)

# Just have one update routine and update for all changes.

Tk.Label(entryframe, text= "Number of divisions: 2^").grid(row=1, column = 0)
div_ent = Tk.Entry(entryframe, textvariable=divpow)
div_ent.grid(row = 1, column = 1)
res_one = Tk.Label(entryframe, text = str(2**divpow.get()), anchor = Tk.W)
res_one.grid(row = 2, column = 1)
div_ent.bind("<Key-Tab>", reset_label)

Tk.Label(entryframe, text= "Threads per block: 2^").grid(row=3, column = 0)
blk_ent = Tk.Entry(entryframe, textvariable=blkpow)
blk_ent.grid(row = 3, column = 1)
res_two = Tk.Label(entryframe, text = str(2**blkpow.get()), anchor = Tk.W)
res_two.grid(row = 4, column = 1)
blk_ent.bind("<Key-Tab>", reset_label)

Tk.Label(entryframe, text= u"\N{GREEK CAPITAL LETTER DELTA}"+"t (seconds): ").grid(row=5, column = 0)
Tk.Entry(entryframe, textvariable=deltat).grid(row = 5, column = 1)

Tk.Label(entryframe, text= "Stopping time (seconds): ").grid(row=6, column = 0)
Tk.Entry(entryframe, textvariable=t_final).grid(row = 6, column = 1)

Tk.Label(entryframe, text= "Output frequency (seconds): ").grid(row=7, column = 0)
Tk.Entry(entryframe, textvariable=fq).grid(row = 7, column = 1)

button_send = Tk.Button(endframe, text="OK", command=ok)
button_send.grid(row = 0, column = 0)
button_sk = Tk.Button(endframe, text="SKIP RUN", command=replot)
button_sk.grid(row = 0, column = 1)
problem_menu = Tk.OptionMenu(dropframe, problem, *OPTIONS, command=reset_vals)
problem_menu.grid(sticky="WE")

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
    print bks
    dt = deltat.get() #timestep in seconds
    print dt
    tf = t_final.get() #timestep in seconds
    print tf
    freq = fq.get()
    print freq
    swept = sw.get()
    print swept
    cpu = proc_share.get()
    print cpu

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
