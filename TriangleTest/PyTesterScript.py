# -*- coding: utf-8 -*-

# Just writing a plotting script for the Swept rule CUDA.
# Perhaps this will also be the calling script.

import matplotlib.pyplot as plt
import numpy as np
import subprocess as sp
import shlex
import os

timingfile = 'TriangleAlgTestTime.txt'
outputf = '1DHeatEQResult.dat'

Algor = ['Original','Justified_Left','Unified_IO']
sourcepath = os.path.dirname(__file__)
basepath = os.path.join(sourcepath,'Results')
timepath = os.path.abspath(os.path.join(basepath,timingfile))
rsltpath = os.path.abspath(os.path.join(basepath,outputf))

typic = int(raw_input("Enter 0 to test timing enter 1 to test output enter 2 for both:\n "))

if (typic > 3) or (typic < 0):
    os.exit(-1)

if os.path.isfile(timepath):
    os.remove(timepath)

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


		print "Percent Difference in integrals:"
		df = 100*abs(np.trapz(data[0][1:],xax)-np.trapz(data[1][1:],xax))/np.trapz(data[0][1:],xax)
		print df

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
    from scipy.interpolate import griddata
    #Just start the file with some parser information rather than human text.
	div = [2**k for k in range(10,20)]
	blx = [32,64,128,256,512,1024]
	for a in range(3):
		fn = open(timepath,'a+')
		fn.write(Algor[a] + "\t" str(len(div)) "\t" str(len(blx)) "\n")
		fn.close()
		for k in blx:
			for n in div:
				runcall = execut + ' {0} {1} {2} {3} {4}'.format(n,k,.01,1000,a)
				runc = shlex.split(runcall)
				proc = sp.Popen(runc)
				sp.Popen.wait(proc)

rslt = np.genfromtxt(timepath)
# OK We need to signal.  It's almost working.





#HOLD ON HERE
