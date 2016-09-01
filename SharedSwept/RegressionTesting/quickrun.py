import subprocess as sp
import shlex
import os

execut = 'python exactcomparison.py '
for k in range(2):
    for n in range(3):
        execstr = execut + '{0} {1}'.format(k, n)
        exeStr = shlex.split(execstr)
        proc = sp.Popen(exeStr)
        sp.Popen.wait(proc)
