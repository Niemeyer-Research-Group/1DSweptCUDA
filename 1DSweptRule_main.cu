/* This file is the current iteration of research being done to implement the
swept rule for Partial differential equations in one dimension.  This research
is a collaborative effort between teams at MIT, Oregon State University, and
Purdue University.

Copyright (C) 2015 Kyle Niemeyer, niemeyek@oregonstate.edu AND
Daniel Magee, mageed@oregonstate.edu

This program is free software: you can redistribute it and/or modify
it under the terms of the MIT license.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

You should have received a copy of the MIT license
along with this program.  If not, see <https://opensource.org/licenses/MIT>.
*/

/*
Note that this code alters the original scheme. Paper available here:
http://www.sciencedirect.com/science/article/pii/S0021999115007664
The nodes never calculate a full diamond in a single kernel call and the boundary
values only get passed one direction, right.  This is a slightly simpler
application that passes the shared values in each node to the GPU global memory
more often.  This cuts down on some of the logic required in the full scheme and
makes results easier to output at various points in the solution.
*/

#include <cuda.h>
#include "cuda_runtime_api.h"
#include "device_functions.h"

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <ostream>
#include <cstring>
#include <fstream>
#include "SwR_1DShared.cuh"

using namespace std;

// Define Given Parameters.  Material is aluminum.
#define DIVISIONS  1024.
#define LENX       5.
#define TS         .01
//#define ITERLIMIT  50000
#define REAL       float
#define TH_DIFF    8.418e-5
//#define THREADBLK  32

// Declare constant Fourier number that will go in Device constant memory.
__constant__ REAL fo;

//The host routine.
int main()
{
	// Choose the GPGPU.  This is device 0 in my machine which has 2 devices.
	cudaSetDevice(0);
	const int dv = int(DIVISIONS); //Setting it to an int helps with arrays
	const int bks = dv/THREADBLK; //The number of blocks since threads/block = 32.
	// Threads/block will be experimented on.
	const REAL ds = LENX/(DIVISIONS-1); //The x division length.
	REAL fou = TS*TH_DIFF/(ds*ds); //The Fourier number.

	// Initialize arrays.
	REAL IC[dv];
	REAL T_final[dv];
	REAL *d_IC, *d_right, *d_left;

	// Some initial condition for the bar temperature, an exponential decay
	// function.
	for (int k = 0; k<dv; k++)
	{
		IC[k] = 500.f*expf((-ds*k)/LENX);
	}

	cout << fou << endl;
	// Call out the file before the loop and write out the initial condition.
	ofstream fwr;
	fwr.open("1DHeatEQResult.dat",ios::trunc);
	// Write out x length and then delta x and then delta t.
	// First item of each line is timestamp.
	fwr << LENX << " " << DIVISIONS << " " << TS << " " << endl << 0 << " ";

	for (int k = 0; k<dv; k++)
	{
		fwr << IC[k] << " ";
	}

	fwr << endl;

	// Transfer data to GPU.

	// This puts the Fourier number in constant memory.
	cudaMemcpyToSymbol(fo,&fou,sizeof(REAL));

	// This initializes the device arrays on the device in global memory.
	// They're all the same size.  Conveniently.
	cudaMalloc((void **)&d_IC, sizeof(REAL)*dv);
	cudaMalloc((void **)&d_right, sizeof(REAL)*dv);
	cudaMalloc((void **)&d_left, sizeof(REAL)*dv);

	// Copy the initial conditions to the device array.
	cudaMemcpy(d_IC,IC,sizeof(REAL)*dv,cudaMemcpyHostToDevice);

	// Start the counter and start the clock.
	REAL t_eq = 0.;
	REAL t_fullstep = TS*(THREADBLK+1);
	double wall0 = clock();

	// Call the kernels until you reach the iteration limit.
	while(t_eq < 1e5)
	{

		upTriangle <<< bks,THREADBLK >>>(d_IC,d_right,d_left);

		downTriangle <<< bks,THREADBLK >>>(d_IC,d_right,d_left);


		t_eq += t_fullstep;

		/* Since the procedure does not store the temperature values, the user
		could input some time interval for which they want the temperature
		values and this loop could copy the values over from the device and
		write them out.  This way the user could see the progression of the
		solution over time, identify an area to be investigated and re-run a
		shorter version of the simulation starting with those intiial conditions.

		-------------------------------------
	 	if (true)
		{
		cudaMemcpy(T_final, d_IC, sizeof(REAL)*dv, cudaMemcpyDeviceToHost);
		fwr << t_eq << " ";

		for (int k = 0; k<dv; k++)
		{
				fwr << T_final[k] << " ";
			}
			fwr << endl;
		}
		-------------------------------------
		*/


	}

	// Show the time and write out the final condition.
	double wall1 = clock();
	double timed = (wall1-wall0)/CLOCKS_PER_SEC;

	cout << "That took: " << timed << " seconds" << endl;


	cudaMemcpy(T_final, d_IC, sizeof(REAL)*dv, cudaMemcpyDeviceToHost);
	fwr << t_eq << " ";
	for (int k = 0; k<dv; k++)
	{
		fwr << T_final[k] << " ";
	}

	fwr.close();

	// Free the memory and reset the device.
	cudaFree(d_IC);
	cudaFree(d_right);
	cudaFree(d_left);
	cudaDeviceReset();

	return 0;
}
