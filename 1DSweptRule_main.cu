/* This file is the current iteration of research being done to implement the
swept rule for Partial differential equations in one dimension.  This research
is a collaborative effort between teams at MIT, Oregon State University, and
Purdue University.

Copyright (C) 2015 Kyle Niemeyer, niemeyek@oregonstate.edu AND
Daniel Magee, mageed@oregonstate.edu

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
Note that this code alters the original scheme. Paper available here:
http://www.sciencedirect.com/science/article/pii/S0021999115007664
The nodes never calculate a full diamond in a single kernel call and the boundary
values only get passed one direction, right.  This is a slightly simpler
application that passes the shared values in each node to the GPU global memory
more often.  This cuts down on some of the logic required in the full scheme and
makes results easier to output at various pointsin the solution.
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

using namespace std;

// Define Given Parameters.  Material is aluminum.
#define DIVISIONS  1024.
#define LENX       50.
#define TS         .5
#define ITERLIMIT  50000
#define REAL       float
#define TH_DIFF    8.418e-5

// Declare constant Fourier number that will go in Device constant memory.
__constant__ REAL fo;

/*
First function, the right side up triangle.  It takes IC which is the array
of temperatures at a single time and returns left and right values that are
either passed to the neighborning node or kept by the home node.  Elements of
both functions were inspired by the nvidia presentation on reduction here:
https://docs.nvidia.com/cuda/samples/6_Advanced/reduction/doc/reduction.pdf
*/

__global__ void upTriangle(REAL *IC, REAL *right, REAL *left)
{
	/*
	Initialize shared variables.  Each node (warp) will store 32 values on the
	right and left sides of their triangle, 2 on each side for each timeslice.
	Since the base of the triangle is 32 numbers for each node, 16 timeslices
	are evaluated per kernel call.
	Temper stores the temperatures at each timeslice.  Since only the current
	and previous timeslice results need to be held at each iteration.  This
	variable has 64 values, or two rows of 32, linearized.  The current and
	previous value alternate rows at each timeslice.
	*/
	__shared__ REAL temper[64];
	__shared__ REAL sR[32];
	__shared__ REAL sL[32];

	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	int tid = threadIdx.x; //Warp or node ID
	int shft_wr; //Initialize the shift to the written row of temper.
	int shft_rd; //Initialize the shift to the read row (opposite of written)

	//Assign the initial values to the first row in temper, each warp (in this
	//case each block) has it's own version of temper shared among its threads.
	temper[tid] = IC[gid];
	__syncthreads(); // Then make sure each block of threads are synced.

	//This counter facilitates the transfer of the relevant values into the
	//right and left saved arrays.
	int itr = -1;

	//The initial conditions are timslice 0 so start k at 1.
	for (int k = 1; k<16; k++)
	{
		//Bitwise even odd. On even iterations write to first row.
		shft_wr = (k & 1);
		//On even iterations write to second row (starts at element 32)
		shft_rd = 32*((shft_wr+1) & 1);

		//Each iteration the triangle narrows.  When k = 1, 30 points are
		//computed, k = 2, 28 points.
		if (tid <= (31-k) && tid >= k)
		{
			temper[tid + (32*shft_wr)] = fo * (temper[tid+shft_rd-1] + temper[tid+shft_rd+1]) + (1-2.*fo) * temper[tid+shft_rd];
		}

		//Make sure the threads are synced
		__syncthreads();

		//Now thread 0 in each block (which never computes a value) is used to
		//fill the shared right and left arrays with the relevant values.
		//This grabs the top and bottom edges on the iteration when the top
		//row is written.
		if (shft_wr && tid == 0)
		{
			sL[k+itr] = temper[k-1];
			sL[k+itr+1] = temper[k];
			sL[k+itr+2] = temper[32+k];
			sL[k+itr+3] = temper[33+k];
			sR[k+itr] = temper[31-k];
			sR[k+itr+1] = temper[32-k];
			sR[k+itr+2] = temper[62-k];
			sR[k+itr+3] = temper[63-k];
			itr += 2;
		}

	}

	//After the triangle has been computed, the right and left shared arrays are
	//stored in global memory by the global thread ID since (conveniently),
	//they're the same size as a warp!
	right[gid] = sR[tid];
	left[gid] = sL[tid];

}

//The upside down triangle.  This function essentially takes right and left and
//returns IC.

//IMPORTANT note: k and tid were in sync in the first function, but here they're
//out of sync in the loop.  This is because we can't use tid = 33 or 32 and the
//shared temperature array is that long.  BUT in order to fill the arrays, these
//elements must be accessed.  So each element in each row is shifted by +1.
//For instance, thread tid = 16 refers to temper[17].  That being said, tid is
//unique and k is NOT so the index must be referenced by tid.

__global__ void downTriangle(REAL *IC, REAL *right, REAL *left)
{

	//Now temper needs to accommodate a longer row by 2, one on each side.
	//since it has two rows that's 4 extra floats.  The last row will still be
	//32 numbers long.
	__shared__ REAL temper[68];
	__shared__ REAL sR[32];
	__shared__ REAL sL[32];

	//Same as upTriangle
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	int shft_rd;
	int shft_wr;

	// Pass to the left so all checks are for block 0 (this reduces arithmetic).
	// The left ridge is always kept by the block.
	sR[tid] = left[gid];

	// The right ridge is passed, each block 1-end gets the right of 0-end-1
	// Block 0 gets the right of the last block.
	if (blockIdx.x > 0)
	{
		sL[tid] = right[gid-blockDim.x];
	}
	else
	{
		sL[tid] = right[blockDim.x*(gridDim.x-1) + tid];
	}

	__syncthreads();

	// Initialize temper. Kind of an unrolled for loop.  This is actually at
	// Timestep 0.
	temper[15] = sL[0];
	temper[16] = sL[1];
	temper[17] = sR[0];
	temper[18] = sR[1];

	//Now we need two counters since we need to use sL and sR EVERY iteration
	//instead of every other iteration and instead of growing smaller with every
	//iteration this grows larger.
	int itr = 2;
	int itr2 = 18;
	//k needs to insert the relevant left right values around the computed values
	//every timestep.  Since it grows larger the loop is reversed.
	for (int k = 17; k>1; k--)
	{
		// This tells you if the current row is the first or second.
		shft_wr = (k & 1);
		// Read and write are opposite rows.
		shft_rd = 34*((shft_wr+1) & 1);

		//Block 0 is split so it needs a different algorithm.  This algorithm
		//is slightly different than top triangle as described in the note above.
		if (blockIdx.x > 0)
		{
			if (tid <= (33-k) && tid >= (k-2))
			{
				temper[tid + 1 + (34*shft_wr)] = fo * (temper[tid+shft_rd] + temper[tid+shft_rd+2]) + (1-2.*fo) * temper[tid+shft_rd+1];
			}

		}

		//Split part.  This exhibits thread divergence and is suboptimal.
		//So it's ripe to be improved.

		else
		{
			if (tid <= (33-k) && tid >= (k-2))
			{
				if (tid == 15)
				{
					temper[tid + 1 + (34*shft_wr)] = 2. * fo * (temper[tid+shft_rd]-temper[tid+shft_rd+1]) + temper[tid+shft_rd+1];
				}
				else if (tid == 16)
				{
					temper[tid + 1 + (34*shft_wr)] = 2. * fo * (temper[tid+shft_rd+2]-temper[tid+shft_rd+1]) + temper[tid+shft_rd+1];
				}
				else
				{
					temper[tid + 1 + (34*shft_wr)] = fo * (temper[tid+shft_rd] + temper[tid+shft_rd+2]) + (1-2.*fo) * temper[tid+shft_rd+1];
				}
			}

		}

		//Fill edges.  Thread 0 never gets used for both operations so the
		//calculation and the filling are conceptually coincident.
		//Threads are synced afterward here because the next timestep is
		//reliant on the entire loop.
		if (k>2 && tid == 0)
		{
			temper[(k-3)+(34*shft_wr)] = sL[itr];
			temper[(k-2)+(34*shft_wr)] = sL[itr+1];
			temper[itr2+(34*shft_wr)] = sR[itr];
			itr2++;
			temper[itr2+(34*shft_wr)] = sR[itr+1];
			itr+=2;

		}
		__syncthreads();

	}

	//Now fill the global unified timestep variable with the final calculated
	//temperatures.

	//Blocks 1 to end hold values 16 to end-16.
	if (blockIdx.x > 0)
	{
		//True if it ends on the first row! The first and last of temper on the final row are empty.
		IC[gid - 16] = temper[tid+1];
	}
	//Block 0 holds values 0 to 15 and end-15 to end.  In that order.
	else
	{
		if (tid>15)
		{
			IC[gid - 16] = temper[tid+1];
		}
		else
		{
			IC[(blockDim.x * gridDim.x) + (tid - 16) ] = temper[tid+1];
		}
	}
}

//The host routine.
int main()
{
	//Choose the GPGPU.  This is device 0 in my machine which has 2 devices.
	cudaSetDevice(0);
	const int dv = int(DIVISIONS); //Setting it to an int helps with arrays
	const int bks = dv/32; //The number of blocks since threads/block = 32.
	//Threads/block will be experimented on.
	const REAL ds = LENX/(DIVISIONS-1); //The x division length.
	REAL fou = TS*TH_DIFF/(ds*ds); //The Fourier number.

	//Initialize arrays.
	REAL IC[dv];
	REAL T_final[dv];
	REAL *d_IC, *d_right, *d_left;

	// Some initial condition for the bar temperature, an exponential decay
	// function.
	for (int k = 0; k<dv; k++)
	{
		IC[k] = 500.f*expf((-ds*k)/LENX);
	}

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

	//Copy the initial conditions to the device array.
	cudaMemcpy(d_IC,IC,sizeof(REAL)*dv,cudaMemcpyHostToDevice);

	// Start the counter and start the clock.
	REAL t_eq = 0.;
	double wall0 = clock();

	// Call the kernels until you reach the iteration limit.
	for(unsigned int k = 0; k < ITERLIMIT; k++)
	{

		upTriangle <<< bks,32 >>>(d_IC,d_right,d_left);

		downTriangle <<< bks,32 >>>(d_IC,d_right,d_left);

		t_eq += (TS*17);

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

	//Show the time and write out the final condition.
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

	//Free the memory and reset the device.
	cudaFree(d_IC);
	cudaFree(d_right);
	cudaFree(d_left);
	cudaDeviceReset();

	return 0;
}
