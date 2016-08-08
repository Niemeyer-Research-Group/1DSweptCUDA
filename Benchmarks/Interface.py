#Just working on the GUI.
#This is not the best way to do this, but Parts are working
import os
import sys
import subprocess as sp
import shlex
import Tkinter as Tk

#It's closer but a certain combination of them should make a certain menu appear.

OPTIONS_A = ["Shared", "Registers"]

OPTIONS_O = ["Plot Result", "Performance Test"]

OPTIONS_P = ["K-S", "Heat", "Euler"]

def algorithm_choice(alg_sel):

    op_sel = Tk.StringVar(master)
    op_sel.set(OPTIONS_O[0]) # default value

    def operation_choice(op_sel):

        prob_sel = Tk.StringVar(master)
        prob_sel.set(OPTIONS_P[0]) # default value


        def problem_choice(prob_sel):

            shared_proc = Tk.BooleanVar(master)
            shared_proc.set(False)

            if prob_sel == "K-S":
                checkme = "Hi"

            elif prob_sel == "Heat":
                check_two = Tk.Checkbutton(entryframe, text = "CPU/GPU sharing", variable = shared_proc)
                check_two.pack(side = 'left')

            else:
                check_two = Tk.Checkbutton(entryframe, text = "CPU/GPU sharing", variable = shared_proc)
                check_two.pack(side = 'left')

            print prob_sel, alg_sel, op_sel

        problem_menu = Tk.OptionMenu(dropframe, prob_sel, *OPTIONS_P, command = problem_choice)
        problem_menu.pack(side = 'left')


    op_menu = Tk.OptionMenu(dropframe, op_sel, *OPTIONS_O, command = operation_choice)
    op_menu.pack(side = 'left')

def ok():
    master.destroy()

def skip():
    master.destroy()

def on_closing():
    raise SystemExit

master = Tk.Tk()
dropframe = Tk.Frame(master)
entryframe = Tk.Frame(master)
endframe = Tk.Frame(master)
dropframe.pack()
endframe.pack(side = 'bottom')
entryframe.pack(side = 'bottom')

alg_sel = Tk.StringVar(master)
alg_sel.set(OPTIONS_A[0]) # default value

algol_menu = Tk.OptionMenu(dropframe, alg_sel, *OPTIONS_A, command = algorithm_choice)
algol_menu.pack(side = 'left')

plt_only = Tk.IntVar(master)
plt_only.set(0)

#Should I enter a range for div and blocks?
arg1 = Tk.IntVar(master)
arg1.set(1)

#dt
arg3 = Tk.DoubleVar(master)
arg3.set(2.0)

#freq
arg4 = Tk.DoubleVar(master)
arg4.set(2.0)

master.protocol("WM_DELETE_WINDOW", on_closing)
button = Tk.Button(endframe, text="OK", command=ok)
button.pack(side = "left")
button = Tk.Button(endframe, text="Skip Run", command=skip)
button.pack(side = "left")

master.mainloop()

Fname = problem.get()
compilate = comp.get()
share = shared_proc.get()

print Fname, compilate, share
