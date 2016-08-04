#Just working on the GUI.

import os
import sys
import Tkinter as Tk


OPTIONS_A = [
    "Shared",
    "Registers"
]

OPTIONS_O = [
    "Plot Result",
    "Performance Test"
]

OPTIONS_P = [
    "K-S",
    "Heat",
    "Euler"
]

def algorithm_choice(alg_sel):

    op_sel = Tk.StringVar(master)
    op_sel.set(OPTIONS_O[0]) # default value



    def operation_choice(op_sel):

        prob_sel = Tk.StringVar(master)
        prob_sel.set(OPTIONS_P[0]) # default value
        problem_choice(prob_sel)

        def problem_choice(prob_sel):

            shared_proc = Tk.BooleanVar(master)
            shared_proc.set(False)

            if prob_sel == "K-S":
                checkme = "Hi"

            elif prob_sel == "Heat":
                check_two = Tk.Checkbutton(master, text = "CPU/GPU sharing", variable = shared_proc)
                check_two.pack(side = 'left')

            else:
                check_two = Tk.Checkbutton(master, text = "CPU/GPU sharing", variable = shared_proc)
                check_two.pack(side = 'left')

            print prob_sel, alg_sel, op_sel

        problem_choice(prob_sel)

        problem_menu = Tk.OptionMenu(master, prob_sel , *OPTIONS_P, command = problem_choice)
        problem_menu.pack()

    op_menu = Tk.OptionMenu(master, op_sel, *OPTIONS_O, command = operation_choice)
    op_menu.pack()

def ok():
    master.destroy()

def skip():
    master.destroy()

def on_closing():
    raise SystemExit

master = Tk.Tk()

alg_sel = Tk.StringVar(master)
alg_sel.set(OPTIONS_A[0]) # default value

algol_menu = Tk.OptionMenu(master, alg_sel, *OPTIONS_A, command = algorithm_choice)
algol_menu.pack()

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
button = Tk.Button(master, text="OK", command=ok)
button.pack()
button = Tk.Button(master, text="Skip Run", command=skip)
button.pack()

master.mainloop()

Fname = problem.get()
compilate = comp.get()
share = shared_proc.get()

print Fname, compilate, share
