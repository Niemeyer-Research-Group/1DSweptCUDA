# -*- coding: utf-8 -*-

# Just writing a plotting script for the Swept rule CUDA.
# Perhaps this will also be the calling script.

import matplotlib.pyplot as plt
import numpy as np
import subprocess as sp
import shlex
import os
from scipy.interpolate import griddata

timingfile = 'TriangleAlgTestTime.txt'
outputf = '1DHeatEQResult.dat'

Algor = ['Original','Justified_Left','Unified_IO']
sourcepath = os.path.dirname(__file__)
basepath = os.path.join(sourcepath,'Results')
timepath = os.path.abspath(os.path.join(basepath,timingfile))
rsltpath = os.path.abspath(os.path.join(basepath,outputf))

typic = int(raw_input("Enter 0 to test timing enter 1 to test output enter 2 for both: "))

if (typic > 3) or (typic < 0):
    os.exit(-1)

execut = './bin/TriTestOut'

if typic:
	for k in range(3):
		# Using a standard small number of divisions and tpb to check.
		runcall = execut + ' {0} {1} {2} {3} {4} '.format(2048,256,.01,10000,k)
		runc = shlex.split(runcall)
		proc = sp.Popen(runc)
		sp.Popen.wait(proc)

		fin = open(rsltpath,'a+')
		data = []

		for line in fin:
			ar = [float(n) for n in line.split()]

			if len(ar)<50:
				xax = np.linspace(0,ar[0],ar[1])
			else:
				data.append(ar)


		lbl = ["Initial Condition"]

		plt.subplot(1,3,k+1)
		plt.plot(xax,data[0][1:])
		plt.hold
		for n in range(1,len(data)):
			plt.plot(xax,data[n][1:])
			lbl.append("t = {} seconds".format(data[n][0]))


		plt.legend(lbl)
		plt.xlabel("Position on bar (m)")
		plt.ylabel("Temperature (C)")
		plt.title(Algor[k])
		plt.grid()

	plt.show()

if (typic - 1):

    ifstat = int(raw_input('Do you already have the data and would rather not rerun the test? 1 for yes 0 for no: '))
    div = [2**k for k in range(10,20)]
    blx = [32,64,128,256,512,1024]
    if ifstat == 0:
        #Just start the file with some parser information rather than human text.
        if os.path.isfile(timepath):
            os.remove(timepath)
        for a in range(3):
            fn = open(timepath,'a+')
            fn.write(Algor[a] + "\t" + str(len(div)) + "\t" + str(len(blx)) + "\n")
            fn.close()
            for k in blx:
                for n in div:
                    runcall = execut + ' {0} {1} {2} {3} {4}'.format(n,k,.01,1000,a)
                    runc = shlex.split(runcall)
                    proc = sp.Popen(runc)
                    sp.Popen.wait(proc)

    rslt = np.genfromtxt(timepath)
    vr = np.where(np.isnan(rslt))
    vr = vr[0][0]

    leng = len(div)*len(blx)

    arslt1 = rslt[vr+1:vr+leng+1,2]
    arslt2 = rslt[vr+leng+2:vr+2*leng+2,2]
    arslt3 = rslt[vr+2*leng+3:vr+3*leng+3,2]

    mx = np.max(np.ravel(np.concatenate((arslt1,arslt2,arslt3))))+.25


    brslt1 = np.reshape(arslt1,(len(blx),len(div)))
    brslt2 = np.reshape(arslt2,(len(blx),len(div)))
    brslt3 = np.reshape(arslt3,(len(blx),len(div)))
    leg = list()

    for k in range(len(blx)):
        leg.append('BlockDim: '+ str(blx[k]))
        plt.subplot(1,3,1)
        plt.semilogx(div,brslt1[k,:])
        plt.hold(True)
        plt.subplot(1,3,2)
        plt.semilogx(div,brslt2[k,:])
        plt.hold(True)
        plt.subplot(1,3,3)
        plt.semilogx(div,brslt3[k,:])
        plt.hold(True)

    plt.subplot(1,3,1)
    plt.title(Algor[0])
    plt.legend(leg,loc = 2)
    plt.grid()
    plt.xlabel('Number of spatial points')
    plt.ylabel('Seconds to completion')
    plt.ylim((0,mx))
    plt.subplot(1,3,2)
    plt.grid()
    plt.title(Algor[1])
    plt.legend(leg,loc = 2)
    plt.xlabel('Number of spatial points')
    plt.ylabel('Seconds to completion')
    plt.ylim((0,mx))
    plt.subplot(1,3,3)
    plt.title(Algor[2],)
    plt.grid()
    plt.legend(leg,loc = 2)
    plt.xlabel('Number of spatial points')
    plt.ylabel('Seconds to completion')
    plt.ylim((0,mx))

    plt.show()







#HOLD ON HERE
