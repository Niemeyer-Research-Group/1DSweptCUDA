/* This file is the current iteration of research being done to implement the
swept rule for Partial differential equations in one discion.  This research
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

//COMPILE LINE!
// nvcc -o ./bin/KSOut KS1D_SweptShared.cu -gencode arch=compute_35,code=sm_35 -lm -restrict -Xcompiler -fopenmp --ptxas-options=-v

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_functions.h>

#include <iostream>
#include <ostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>


#ifndef REAL
#define REAL        float
#define ONE         1.f
#define TWO         2.f
#define FOUR        4.f
#define SIX			6.f
#else
#define ONE         1.0
#define TWO         2.0
#define FOUR        4.0
#define SIX			6.0
#endif

#ifndef WIDTH
#define BASE	 	36
#define BASEHalf	18
#define WIDTH		32
#define WIDTHHalf	16
#endif


using namespace std;

const REAL dx = 0.5;

// I seriously suspect that in order to keep the main variables in registers, the 
// base - widthHalf variables will need to be #defines.

struct discConstants{

	REAL dxTimes4;
	REAL dx2;
	REAL dx4;
	REAL dt;
	REAL dt_half;
        int idxend;
};

__constant__ discConstants disc;

//Initial condition.
__host__
REAL initFun(REAL xnode)
{
	return TWO * cos(19.0*xnode*M_PI/128.0);
}

//Read in the data from the global right/left variables to the shared temper variable.
// __device__
// __forceinline__
// void
// readIn(REAL *temp, const REAL *rights, const REAL *lefts, int td, int gd)
// {
// 	int leftidx = disc.ht + (((td>>2) & 1) * disc.base) + (td & 3) - (4 + ((td>>2)<<1));
// 	int rightidx = disc.ht + (((td>>2) & 1) * disc.base) + ((td>>2)<<1) + (td & 3);

// 	temp[leftidx] = rights[gd];
// 	temp[rightidx] = lefts[gd];
// }

// __device__
// __forceinline__
// void
// writeOutRight(REAL *temp, REAL *rights, REAL *lefts, int td, int gd, int bd)
// {
// 	int gdskew = (gd + bd) & disc.idxend;
//     int leftidx = (((td>>2) & 1)  * disc.base) + ((td>>2)<<1) + (td & 3) + 2;
//     int rightidx = (BASE-6) + (((td>>2) & 1)  * BASE) + (td & 3) - ((td>>2)<<1);
// 	rights[gdskew] = temp[rightidx];
// 	lefts[gd] = temp[leftidx];
// }


// __device__
// __forceinline__
// void
// writeOutLeft(REAL *temp, REAL *rights, REAL *lefts, int td, int gd, int bd)
// {
// 	int gdskew = (gd - bd) & disc.idxend;
//     int leftidx = (((td>>2) & 1)  * BASE) + ((td>>2)<<1) + (td & 3) + 2;
//     int rightidx = (BASE-6) + (((td>>2) & 1)  * BASE) + (td & 3) - ((td>>2)<<1);
// 	rights[gd] = temp[rightidx];
// 	lefts[gdskew] = temp[leftidx];
// }

__device__
__forceinline__
REAL fourthDer(REAL tfarLeft, REAL tLeft, REAL tCenter, REAL tRight, REAL tfarRight)
{
	return (tfarLeft - FOUR*tLeft + SIX*tCenter - FOUR*tRight + tfarRight)*(disc.dx4);
}

__device__
__forceinline__
REAL secondDer(REAL tLeft, REAL tRight, REAL tCenter)
{
	return (tLeft + tRight - TWO*tCenter)*(disc.dx2);
}

__device__
__forceinline__
REAL convect(REAL tLeft, REAL tRight)
{
	return (tRight*tRight - tLeft*tLeft)*(disc.dxTimes4);
}


__device__
__forceinline__
REAL stutterStep(REAL tfarLeft, REAL tLeft, REAL tCenter, REAL tRight, REAL tfarRight)
{
	return tCenter - disc.dt_half * (convect(tLeft, tRight) + secondDer(tLeft, tRight, tCenter) +
		fourthDer(tfarLeft, tLeft, tCenter, tRight, tfarRight));
}

__device__
__forceinline__
REAL finalStep(REAL tfarLeft, REAL tLeft, REAL tCenter, REAL tRight, REAL tfarRight)
{
	return (-disc.dt * (convect(tLeft, tRight) + secondDer(tLeft, tRight, tCenter) +
			fourthDer(tfarLeft, tLeft, tCenter, tRight, tfarRight)));
}

__global__
void
upTriangle(const REAL *IC, REAL *outRight, REAL *outLeft)
{

	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	int gidout = (gid+1) & disc.idxend;

	REAL vel[2][BASE];

	#pragma unroll
	for (int k=0; k<(WIDTH); k++) vel[0][k+2] = IC[gid * WIDTH + k];

	#pragma unroll
	for (int i=4; i<(WIDTHHalf); i+=2)
	{
		int read = ((i>>1) & 1);
		int write = !read;

		#pragma unroll
		for (int j=i; j<(BASE-i); j++)
		{
			if (read)
			{ 
				vel[write][j] += finalStep(vel[read][j-2], vel[read][j-1], vel[read][j], vel[read][j+1], vel[read][j+2]); 
			}
			else
			{
				vel[write][j] = stutterStep(vel[read][j-2], vel[read][j-1], vel[read][j], vel[read][j+1], vel[read][j+2]); 
			}
		}
	}

	#pragma unroll
	for (int k=0; k<WIDTHHalf; k++) 
	{
		outLeft[k*disc.idxend + gid] = vel[0][k+2];
		outRight[k*disc.idxend + gidout] = vel[0][BASEHalf+k];
	}
	#pragma unroll
	for (int k=0; k<WIDTHHalf; k++) 
	{
		outLeft[(k+WIDTHHalf)*disc.idxend + gid] = vel[1][k+4];
		outRight[(k+WIDTHHalf)*disc.idxend + gidout] = vel[1][WIDTHHalf + k];
	}

}

__global__
void
downTriangle(REAL *IC, const REAL *inRight, const REAL *inLeft)
{
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	REAL vel[2][BASE];
	
	// We'll see what happens. 
	#pragma unroll
	for (int k=0; k<(WIDTHHalf); k++) 	
	{
		vel[0][k + BASEHalf] = inLeft[k*disc.idxend + gid]; //Goes to Right side
		vel[0][k + 2] = inRight[k*disc.idxend + gid];
		vel[1][k + BASEHalf + 2] = inLeft[(k+WIDTHHalf)*disc.idxend + gid]; //Goes to Right side
		vel[1][k] = inRight[(k+WIDTHHalf)*disc.idxend + gid];
	}

	#pragma unroll
	for (int i=WIDTHHalf; i>0; i-=2)
	{
		int read = ((i>>1) & 1);
		int write = !read;

		#pragma unroll
		for (int j=i; j<(BASE-i); j++)
		{
			if (read)
			{ 
				vel[write][j] += finalStep(vel[read][j-2], vel[read][j-1], vel[read][j], vel[read][j+1], vel[read][j+2]); 
			}
			else
			{
				vel[write][j] = stutterStep(vel[read][j-2], vel[read][j-1], vel[read][j], vel[read][j+1], vel[read][j+2]); 
			}
		}
	}

    #pragma unroll
    for (int k=0; k<(WIDTH); k++)  IC[gid * WIDTH + k] = vel[0][k+2];
}


__global__
void
wholeDiamond(REAL *inRight, REAL *inLeft, REAL *outRight, REAL *outLeft, const bool split)
{
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	REAL vel[2][BASE];
	
	// We'll see what happens. 
	#pragma unroll
	for (int k=0; k<(WIDTHHalf); k++) 	
	{
		vel[0][k + BASEHalf] = inLeft[k*disc.idxend + gid]; //Goes to Right side
		vel[0][k + 2] = inRight[k*disc.idxend + gid];
		vel[1][k + BASEHalf + 2] = inLeft[(k+WIDTHHalf)*disc.idxend + gid]; //Goes to Right side
		vel[1][k] = inRight[(k+WIDTHHalf)*disc.idxend + gid];
	}

	#pragma unroll
	for (int i=WIDTHHalf; i>0; i-=2)
	{
		int read = ((i>>1) & 1);
		int write = !read;

		#pragma unroll
		for (int j=i; j<(BASE-i); j++)
		{
			if (read)
			{ 
				vel[write][j] += finalStep(vel[read][j-2], vel[read][j-1], vel[read][j], vel[read][j+1], vel[read][j+2]); 
			}
			else
			{
				vel[write][j] = stutterStep(vel[read][j-2], vel[read][j-1], vel[read][j], vel[read][j+1], vel[read][j+2]); 
			}
		}
	}

    //-------------------TOP PART------------------------------------------
	#pragma unroll
	for (int i=4; i<(BASEHalf); i+=2)
	{
		int read = ((i>>1) & 1);
		int write = !read;

		#pragma unroll
		for (int j=i; j<(BASE-i); j++)
		{
			if (read)
			{ 
				vel[write][j] += finalStep(vel[read][j-2], vel[read][j-1], vel[read][j], vel[read][j+1], vel[read][j+2]); 
			}
			else
			{
				vel[write][j] = stutterStep(vel[read][j-2], vel[read][j-1], vel[read][j], vel[read][j+1], vel[read][j+2]); 
			}
		}
	}

	if (split)
	{
		int gidout = (gid-1) & disc.idxend;
		#pragma unroll
		for (int k=0; k<WIDTHHalf; k++) 
		{
			outLeft[k*disc.idxend + gidout] = vel[0][k+2];
			outRight[k*disc.idxend + gid] = vel[0][BASEHalf+k];
		}
		#pragma unroll
		for (int k=0; k<WIDTHHalf; k++) 
		{
			outLeft[(k+WIDTHHalf)*disc.idxend + gidout] = vel[1][k+4];
			outRight[(k+WIDTHHalf)*disc.idxend + gid] = vel[1][BASEHalf + (k-2)];
		}
	}
	else
	{
		int gidout = (gid+1) & disc.idxend;
		for (int k=0; k<WIDTHHalf; k++) 
		{
			outLeft[k*disc.idxend + gid] = vel[0][k+2];
			outRight[k*disc.idxend + gidout] = vel[0][BASEHalf+k];
		}
		#pragma unroll
		for (int k=0; k<WIDTHHalf; k++) 
		{
			outLeft[(k+WIDTHHalf)*disc.idxend + gid] = vel[1][k+4];
			outRight[(k+WIDTHHalf)*disc.idxend + gidout] = vel[1][BASEHalf + (k-2)];
		}
	}


}

//The host routine.
double
sweptWrapper(const int bks, int tpb, const int dv, REAL dt, const REAL t_end,
	REAL *IC, REAL *T_f, const REAL freq, ofstream &fwr)
{

	REAL *d_IC, *d0_right, *d0_left, *d2_right, *d2_left;

	cudaMalloc((void **)&d_IC, sizeof(REAL)*dv);
	cudaMalloc((void **)&d0_right, sizeof(REAL)*dv);
	cudaMalloc((void **)&d0_left, sizeof(REAL)*dv);
	cudaMalloc((void **)&d2_right, sizeof(REAL)*dv);
	cudaMalloc((void **)&d2_left, sizeof(REAL)*dv);

	// Copy the initial conditions to the device array.
	cudaMemcpy(d_IC,IC,sizeof(REAL)*dv,cudaMemcpyHostToDevice);
	//Start the counter and start the clock.
	//
	//Every other step is a full timestep and each cycle is half tpb steps.
	const double t_fullstep = 0.25 * dt * (double)WIDTH;
	double twrite = freq;

	upTriangle <<< bks,tpb >>> (d_IC,d0_right,d0_left);

	//Split
	wholeDiamond <<< bks,tpb >>> (d0_right,d0_left,d2_right,d2_left,true);

	double t_eq = t_fullstep;

	// Call the kernels until you reach the iteration limit.
	while(t_eq < t_end)
	{

		wholeDiamond <<< bks,tpb >>> (d2_right,d2_left,d0_right,d0_left,false);

		//So it always ends on a left pass since the down triangle is a right pass.

		//Split
		wholeDiamond <<< bks,tpb >>> (d0_right,d0_left,d2_right,d2_left,true);

		t_eq += t_fullstep;

	 	if (t_eq > twrite)
		{
			downTriangle <<< bks,tpb >>> (d_IC,d2_right,d2_left);

			cudaMemcpy(T_f, d_IC, sizeof(REAL)*dv, cudaMemcpyDeviceToHost);

			fwr << " Velocity " << t_eq << " ";

			for (int k = 0; k<dv; k++)	fwr << T_f[k] << " ";

			fwr << endl;

			upTriangle <<< bks,tpb >>> (d_IC,d0_right,d0_left);

			//Split
			wholeDiamond <<< bks,tpb >>> (d0_right,d0_left,d2_right,d2_left,true);

			t_eq += t_fullstep;

			twrite += freq;
		}

	}

	downTriangle <<< bks,tpb>>> (d_IC,d2_right,d2_left);

	cudaMemcpy(T_f, d_IC, sizeof(REAL)*dv, cudaMemcpyDeviceToHost);

	cudaFree(d_IC);
	cudaFree(d0_right);
	cudaFree(d0_left);
	cudaFree(d2_right);
	cudaFree(d2_left);

	return t_eq;

}

int main( int argc, char *argv[])
{

	if (argc < 9)
	{
		cout << "The Program takes 9 inputs, #Divisions, #Threads/block, deltat, finish time, output frequency..." << endl;
        cout << "Classic/Swept, CPU sharing Y/N (Ignored), Variable Output File, Timing Output File (optional)" << endl;
		exit(-1);
	}

	// Choose the GPGPU.  This is device 0 in my machine which has 2 devices.
	cudaSetDevice(0);
	if (sizeof(REAL)>6) cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

	const int dv = atoi(argv[1]); //Number of spatial points
	const int tpb = atoi(argv[2]); //Threads per Block
    	const REAL dt = atof(argv[3]); //delta T timestep
	const float tf = atof(argv[4]); //Finish time
    	const float freq = atof(argv[5]); //Output frequency
    	const int scheme = atoi(argv[6]); //1 for Swept 0 for classic
    
	const int bks = dv/(WIDTH*tpb); //The number of blocks
	cout << tpb << " " << WIDTH << " " << WIDTH*tpb << endl;
	const float lx = dv*dx;
	char const *prec;
	prec = (sizeof(REAL)<6) ? "Single": "Double";

	cout << "KS --- #Blocks: " << bks << " | Length: " << lx << " | Precision: " << prec << " | dt/dx: " << dt/dx << endl;

	//Conditions for main input.  Unit testing kinda.
	//dv and tpb must be powers of two.  dv must be larger than tpb and divisible by
	//tpb.

	if ((dv & (tpb-1) !=0) || (tpb&31) != 0)
    {
        cout << "INVALID NUMERIC INPUT!! "<< endl;
        cout << "2nd ARGUMENT MUST BE A POWER OF TWO >= 32 AND FIRST ARGUMENT MUST BE DIVISIBLE BY SECOND" << endl;
        exit(-1);
    }

	discConstants dsc = {
		ONE/(FOUR*dx), //dx
		ONE/(dx*dx), //dx^2
		ONE/(dx*dx*dx*dx), //dx^4
		dt, //dt
		dt*0.5, //dt half
		(dv/WIDTH)-1, //last global thread 
	};

	// Initialize arrays.
    REAL *IC, *T_final;

	cudaHostAlloc((void **) &IC, dv*sizeof(REAL), cudaHostAllocDefault);
	cudaHostAlloc((void **) &T_final, dv*sizeof(REAL), cudaHostAllocDefault);

    // IC = (REAL *) malloc(dv*sizeof(REAL));
    // T_final = (REAL *) malloc(dv*sizeof(REAL));

	// Inital condition
	for (int k = 0; k<dv; k++)
	{
		IC[k] = initFun((REAL)k*(REAL)dx);
	}

	// Call out the file before the loop and write out the initial condition.
	ofstream fwr;
	fwr.open(argv[8],ios::trunc);

	// Write out x length and then delta x and then delta t.
	// First item of each line is timestamp.
	fwr << lx << " " << dv << " " << dx << " " << endl << " Velocity " << 0 << " ";

	for (int k = 0; k<dv; k++) fwr << IC[k] << " ";

	fwr << endl;
	// Transfer data to GPU.

	// This puts the constant part of the equation in constant memory
	cudaMemcpyToSymbol(disc,&dsc,sizeof(dsc));

	// Start the counter and start the clock.
	cudaEvent_t start, stop;
	float timed;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
	cudaEventRecord( start, 0);

	// Call the kernels until you reach the iteration limit.
	double tfm;


	tfm = sweptWrapper(bks, tpb, dv, dsc.dt, tf, IC, T_final, freq, fwr);
	
	
	// Show the time and write out the final condition.
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime( &timed, start, stop);

	timed *= 1.e3;

	double n_timesteps = tfm/dt;

    double per_ts = timed/n_timesteps;

    cout << n_timesteps << " timesteps" << endl;
	cout << "Averaged " << per_ts << " microseconds (us) per timestep" << endl;
	
    cout << timed << endl;
    if (argc>8)
    {
        ofstream ftime;
        ftime.open(argv[9],ios::app);
    	ftime << dv << "\t" << tpb << "\t" << per_ts << endl;
    	ftime.close();
    }

	fwr << " Velocity " << tfm << " ";
	for (int k = 0; k<dv; k++) fwr << T_final[k] << " ";

    fwr << endl;

	fwr.close();

	cudaDeviceSynchronize();
	// Free the memory and reset the device.

	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	cudaDeviceReset();
	cudaFreeHost(IC);
    cudaFreeHost(T_final);

	return 0;

}
