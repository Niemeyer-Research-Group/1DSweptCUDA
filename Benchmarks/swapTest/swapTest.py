import subprocess as sp
import shlex
import os
import os.path as op

sourcepath = op.abspath(op.dirname(__file__))

typers = ["float","double","float3","double3"]

div = [2**k for k in range(13, 21)]
blx = [2**k for k in range(5, 11)]

prog = "./bin/swapTestOut"
compStr = "nvcc -o " + prog + " swapTest.cu -gencode arch=compute_35,code=sm_35 -lm -restrict "

for i, ty in enumerate(typers):
    rslt = op.join(sourcepath,"swapTiming" + ty + ".txt")
    compit = shlex.split(compStr + "-DREAL={0} -DRSINGLE={1}".format(ty,typers[i%2]))
    proc = sp.Popen(compit)
    sp.Popen.wait(proc)

    for dv in div:
        for tpb in blx:
            execut = prog + " {0} {1} {2}".format(dv,tpb,rslt)
            exeStr = shlex.split(execut)
            proc = sp.Popen(exeStr)
            sp.Popen.wait(proc)
