'''
    Run the performance test described in the paper.  
    Will save all best runs to an hd5 file in a pandas dataframe in Results folder.
    Will also save ALL timing for the last run to appropriately named text files in Results folder.
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
import time

SCHEMES = [
    "Classic",
    "Shared",
]

OPTIONS = {
#    "Heat": SCHEMES[:],
    "Euler": SCHEMES[:],
#    "KS": SCHEMES[:]
}

# OPTIONS["KS"][-1] = "Register"

PRECISION = [
#    "Single",	
    "Double"
]

DT = {
    "Heat": 0.001,
    "Euler": 1e-7,
    "KS": 0.005,
}

#Set this for numer of times to run it.  If no argument run it once.  Otherwise run it 
if len(sys.argv) == 1:
    nRuns = 1
else:
    nRuns = sys.argv[1]

if not op.isdir(rsltpath):
    os.mkdir(rsltpath)

if not op.isdir(binpath):
    os.mkdir(binpath)

sp.call("make")
st = 5e5 #Number of timesteps
timeend = '_Timing.txt'
stfile = op.join(rsltpath, 'performanceData.h5')
finalfile = op.join(rsltpath, 'performanceParsed.h5')

if op.isfile(stfile):
    os.remove(stfile)

store = pd.HDFStore(stfile)

div = [2**k for k in range(11,20)]
blx = [2**k for k in range(6, 10)]
print nRuns

for n in xrange(nRuns):
    tt = time.time()
    for opt in sorted(OPTIONS.keys()):
        dt = DT[opt]
        tf = dt*st
        for pr in PRECISION:
            #Path to this executable file.
            binf = opt + pr + 'Out'
            ExecL = op.join(binpath,binf)

            for sch in range(len(SCHEMES)):
                timename = opt + "_" + pr + "_" + OPTIONS[opt][sch] + timeend
                
                timepath = op.join(rsltpath, timename)

                #Erase existing run and write header.
                t_fn = open(timepath,'w')
                t_fn.write("Num_Spatial_Points\tThreads_per_Block\tTime_per_timestep_(us)\n")
                t_fn.close()
                mh.runCUDA(ExecL, div, blx, dt, tf, tf*2.0, sch, timefile=timepath)


    speedTest = ah.gatherTest(rsltpath)
    speedObj = ah.QualityRuns(speedTest)
    store.put('n'+str(n), speedObj.bestRun)

    print("The whole thing took: {:.3f} minutes".format((time.time() - tt)/60.0))

names = ['n'+str(k) for k in range(nRuns)] + ['mean', 'std']
df_all = pd.concat([store[k] for k in store.keys()], axis=1)
df_all = pd.concat([df_all, df_all.mean(axis=1), df_all.std(axis=1)], axis=1)
store.close()

store2 = pd.HDFStore(finalfile)
store2.put('runs' ,df_all)
store2.close()

