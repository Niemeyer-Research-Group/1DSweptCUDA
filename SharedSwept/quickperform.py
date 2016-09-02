import subprocess as sp
import shlex
import os

execut = 'python sweptPerformanceTest.py '

OPTIONS = ["Heat", "Euler", "KS"]

PRECISION = ["Single", "Double"]
cy = 100
bnd = [11, 20, 5, 32]

for opt in OPTIONS:
    for pr in PRECISION:
        for sch in range(3):
                execstr = execut + '{0} {1} {2} {3} {4} {5} {6} {7}'.format(opt, pr, sch, bnd[0], bnd[1], bnd[2], bnd[3], cy)
                exeStr = shlex.split(execstr)
                proc = sp.Popen(exeStr)
                sp.Popen.wait(proc)
