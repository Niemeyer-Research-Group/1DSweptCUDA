#Just working on the GUI.

import os
import Tkinter as Tk

OPTIONS = [
    "KS",
    "Heat",
    "Euler"
]

master = Tk.Tk()

master.geometry("400x400")

problem = Tk.StringVar(master)
problem.set(OPTIONS[0]) # default value

comp = Tk.BooleanVar(master)
comp.set(False)

shared_proc = Tk.BooleanVar(master)
shared_proc.set(False)

#Should I enter a range for div and blocks?
arg1 = Tk.IntVar(master)
arg1.set(1)

#A Boolean variable for test or not test.

#final time, tf
arg2 = Tk.IntVar(master)
arg2.set(2)

#dt
arg3 = Tk.DoubleVar(master)
arg3.set(2.0)

#freq
arg4 = Tk.DoubleVar(master)
arg4.set(2.0)

drop = apply(Tk.OptionMenu, (master, problem) + tuple(OPTIONS))
drop.pack()

check_one = Tk.Checkbutton(master, text = "Compile? ", variable = comp)
check_one.pack(side = 'left')

check_two = Tk.Checkbutton(master, text = "CPU/GPU sharing", variable = shared_proc)
check_two.pack(side = 'right')

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

Fname = problem.get()
compilate = comp.get()
share = shared_proc.get()

print Fname, compilate, share
