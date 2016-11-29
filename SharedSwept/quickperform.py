import subprocess as sp
import shlex
import os
import time

execut = 'python sweptPerformanceTest.py '

OPTIONS = ["Heat", "Euler", "KS"]

PRECISION = ["Single", "Double"]
bnd = [11, 20, 5, 10]
st = 5e4 #Number of timesteps
OPTIONS.pop(1)
print OPTIONS

tt = time.time()
for opt in OPTIONS:
    for pr in PRECISION:
        for sch in range(3):
                if OPTIONS.index(opt) == 2 and sch == 2:
                    break
                execstr = execut + '{0} {1} {2} {3} {4} {5} {6} {7}'.format(opt, pr, sch, bnd[0], bnd[1], bnd[2], bnd[3], st)
                exeStr = shlex.split(execstr)
                proc = sp.Popen(exeStr)
                sp.Popen.wait(proc)

print "The whole thing took: {:.3f} minutes".format((time.time() - tt)/60.0)
