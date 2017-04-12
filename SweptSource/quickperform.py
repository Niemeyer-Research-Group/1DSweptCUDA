'''
    Run the performance test described in the paper.
    This just might work.
'''

import os
import os.path as op
import sys

sourcepath = op.abspath(op.dirname(__file__))
rsltpath = op.join(sourcepath,'Results')
binpath = op.join(sourcepath,'bin') #Binary directory
gitpath = op.dirname(sourcepath) #Top level of git repo
modpath = op.join(gitpath,"pyAnalysisTools")
os.chdir(sourcepath)

sys.path.append(modpath)
import main_help as mh
import analysis_help as ah
import subprocess as sp
import pandas as pd

SCHEMES = [
    "Classic",
    "Shared",
    "Hybrid"
]

OPTIONS = {
    "Heat": SCHEMES,
    "Euler": SCHEMES,
    "KS": SCHEMES
}

OPTIONS['KS'][-1] = "Register"

PRECISION = [
    "Single",
    "Double"
]

DT = {
    "Heat": 0.001,
    "Euler": 1e-7,
    "KS": 0.005,
}

#Set this for numer of times to run it.
nRuns = 5

if not op.isdir(rsltpath):
    os.mkdir(rsltpath)

if not op.isdir(binpath):
    os.mkdir(binpath)

sp.call("make")
st = 5e4 #Number of timesteps
timeend = '_Timing.txt'
stfile = op.join(rsltpath, 'performanceStore.hd5')
if os.isfile(stfile):
    os.remove(stfile)
store = pd.HDFStore(stfile)

div = [2**k for k in range(11, 21)]
blx = [2**k for k in range(5, 11)]

for n in xrange(nRuns):
    tt = time.time()
    for opt in OPTIONS.keys():
        dt = DT[opt]
        tf = dt*st
        for pr in PRECISION:
            #Path to this executable file.
            binf = opt + pr + 'Out'
            ExecL = op.join(binpath,binf)

            for sch in range(3):
                timename = opt + "_" + pr + "_" + OPTIONS[opt][sch] + timeend
                plotstr = timename.replace("_"," ")
                timepath = op.join(rsltpath,timefile)

                #Erase existing run and write header.
                t_fn = open(timepath,'w')
                t_fn.write("Num_Spatial_Points\tThreads_per_Block\tTime_per_timestep_(us)\n")
                t_fn.close()
                mh.runCUDA(ExecL, div, blx, dt, tf, tf*2.0, sch, timefile=timepath)


    speedTest = ah.gatherTest(rsltpath)
    speedObj = ah.QualityRuns(speedTest)
    store.put(n, speedObj.bestRun)

    print("The whole thing took: {:.3f} minutes".format((time.time() - tt)/60.0))

store.close()


#Then move the hd5 to the paper folder and fiddle it to get the average and standard deviation.  Then use just read the average to make the plots.
