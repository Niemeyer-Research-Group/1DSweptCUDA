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
#define REAL  float
#endif

using namespace std;

const REAL dx = 0.5;

struct discConstants{

	REAL dx;
	REAL dx2;
	REAL dx4;
	REAL dt;
	REAL dt_half;
};

__constant__ discConstants disc;

__host__
REAL initFun(REAL xnode)
{
	return 2.0 * cos(19.0*xnode*M_PI/128.0);
}

__device__
__forceinline__
REAL fourthDer(REAL tfarLeft, REAL tLeft, REAL tCenter, REAL tRight, REAL tfarRight)
{
	return (tfarLeft - 4.f*tLeft + 6.f*tCenter - 4.f*tRight + tfarRight)/(disc.dx4);
}

__device__
__forceinline__
REAL secondDer(REAL tLeft, REAL tRight, REAL tCenter)
{
	return (tLeft + tRight - 2.f*tCenter)/(disc.dx2);
}

__device__
__forceinline__
REAL convect(REAL tLeft, REAL tRight)
{
	return (tRight*tRight - tLeft*tLeft)/(4.f*disc.dx);
}

__device__
REAL stutterStep(REAL tfarLeft, REAL tLeft, REAL tCenter, REAL tRight, REAL tfarRight)
{
	return tCenter - disc.dt_half * (convect(tLeft, tRight) + secondDer(tLeft, tRight, tCenter) +
		fourthDer(tfarLeft, tLeft, tCenter, tRight, tfarRight));
}

__device__
REAL finalStep(REAL tfarLeft, REAL tLeft, REAL tCenter, REAL tRight, REAL tfarRight)
{
	return (-disc.dt * (convect(tLeft, tRight) + secondDer(tLeft, tRight, tCenter) +
			fourthDer(tfarLeft, tLeft, tCenter, tRight, tfarRight)));
}

__global__
void
swapKernel(const REAL *passing_side, REAL *bin, int direction)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
    int lastidx = ((blockDim.x*gridDim.x)-1);
    int gidout = (gid + direction*blockDim.x) & lastidx;

    bin[gidout] = passing_side[gid];

}

//Classic
__global__
void
classicKS(const REAL *ks_in, REAL *ks_out, bool final)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
    int lastidx = ((blockDim.x*gridDim.x)-1);

	if (final)
	{
		ks_out[gid] += finalStep(ks_in[(gid-2)&lastidx],ks_in[(gid-1)&lastidx],ks_in[gid],ks_in[(gid+1)&lastidx],ks_in[(gid+2)&lastidx]);
	}
	else
	{
		ks_out[gid] = stutterStep(ks_in[(gid-2)&lastidx],ks_in[(gid-1)&lastidx],ks_in[gid],ks_in[(gid+1)&lastidx],ks_in[(gid+2)&lastidx]);
	}
}

__global__
void
upTriangle(const REAL *IC, REAL *right, REAL *left)
{
	extern __shared__ REAL temper[];

	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	int tid = threadIdx.x; //Block Thread ID

	int tid_top = tid + blockDim.x;

	int leftidx = ((tid/4 & 1) * blockDim.x) + (tid/4)*2 + (tid & 3);
	int rightidx = (blockDim.x - 4) + ((tid/4 & 1) * blockDim.x) + (tid & 3) - (tid/4)*2;

	int step2;

    //Assign the initial values to the first row in temper, each block
    //has it's own version of temper shared among its threads.
	temper[tid] = IC[gid];

	__syncthreads();

	if (tid > 1 && tid <(blockDim.x-2))
	{
		temper[tid_top] = stutterStep(temper[tid - 2], temper[tid - 1], temper[tid],
			temper[tid + 1], temper[tid + 2]);
	}

	__syncthreads();

	//The initial conditions are timslice 0 so start k at 1.
	for (int k = 4; k<(blockDim.x/2); k+=4)
	{
		if (tid < (blockDim.x-k) && tid >= k)
		{
			temper[tid] += finalStep(temper[tid_top - 2], temper[tid_top - 1], temper[tid_top],
				temper[tid_top + 1], temper[tid_top + 2]);
		}

		step2 = k + 2;
		__syncthreads();

		if (tid < (blockDim.x-step2) && tid >= step2)
		{
			temper[tid_top] = stutterStep(temper[tid - 2], temper[tid - 1], temper[tid],
				temper[tid + 1], temper[tid + 2]);
		}

		//Make sure the threads are synced
		__syncthreads();

	}
	//After the triangle has been computed, the right and left shared arrays are
	//stored in global memory by the global thread ID since (conveniently),
	//they're the same size as a warp!
	right[gid] = temper[rightidx];
	left[gid] = temper[leftidx];
}

__global__
void
downTriangle(REAL *IC, const REAL *right, const REAL *left)
{
	extern __shared__ REAL temper[];

	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	int tididx = tid + 2;
	int base = blockDim.x + 4;
	int height = base/2;
	int step2;

	int tid_top[5], tid_bottom[5];
	#pragma unroll
	for (int k = -2; k<3; k++)
	{
		tid_top[k+2] = tididx + k + base;
		tid_bottom[k+2] = tididx + k;
	}

	int leftidx = height + ((tid/4 & 1) * base) + (tid & 3) - (4 + (tid/4) * 2);
	int rightidx = height + ((tid/4 & 1) * base) + (tid/4)*2 + (tid & 3);

	temper[leftidx] = right[gid];
	temper[rightidx] = left[gid];

	__syncthreads();

	for (int k = (height-2); k>0; k-=4)
	{
		if (tididx < (base-k) && tididx >= k)
		{
			temper[tid_top[2]] = stutterStep(temper[tid_bottom[0]], temper[tid_bottom[1]], temper[tid_bottom[2]],
				temper[tid_bottom[3]], temper[tid_bottom[4]]);

		}

		step2 = k-2;
		__syncthreads();

		if (tididx < (base-step2) && tididx >= step2)
		{
			temper[tididx] += finalStep(temper[tid_top[0]], temper[tid_top[1]], temper[tid_top[2]],
				temper[tid_top[3]], temper[tid_top[4]]);
		}

		//Make sure the threads are synced
		__syncthreads();
	}

    IC[gid] = temper[tididx];
}


__global__
void
wholeDiamond(REAL *right, REAL *left)
{
	extern __shared__ REAL temper[];

	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	int tididx = tid + 2;
	int base = blockDim.x + 4;
	int height = base/2;
	int step2;

	int tid_top[5], tid_bottom[5];
	#pragma unroll
	for (int k = -2; k<3; k++)
	{
		tid_top[k+2] = tididx + k + base;
		tid_bottom[k+2] = tididx + k;
	}

	int leftidx = height + ((tid/4 & 1) * base) + (tid & 3) - (4 + (tid/4) * 2);
	int rightidx = height + ((tid/4 & 1) * base) + (tid/4)*2 + (tid & 3);

	temper[leftidx] = right[gid];
	temper[rightidx] = left[gid];

	__syncthreads();

	for (int k = (height-2); k>0; k-=4)
	{
		if (tididx < (base-k) && tididx >= k)
		{
			temper[tid_top[2]] = stutterStep(temper[tid_bottom[0]], temper[tid_bottom[1]], temper[tid_bottom[2]],
				temper[tid_bottom[3]], temper[tid_bottom[4]]);
		}

		step2 = k-2;
		__syncthreads();

		if (tididx < (base-step2) && tididx >= step2)
		{
			temper[tididx] += finalStep(temper[tid_top[0]], temper[tid_top[1]], temper[tid_top[2]],
				temper[tid_top[3]], temper[tid_top[4]]);
		}

		//Make sure the threads are synced
		__syncthreads();
	}

	//Shift the last row to justify it at 0.
	REAL trade = temper[tididx];
	__syncthreads();
	temper[tid] = trade;
	__syncthreads();
    //-------------------TOP PART------------------------------------------

	leftidx = ((tid/4 & 1) * blockDim.x) + (tid/4)*2 + (tid & 3);
	rightidx = (blockDim.x - 4) + ((tid/4 & 1) * blockDim.x) + (tid & 3) - (tid/4)*2;

	#pragma unroll
	for (int k = -2; k<3; k++)
	{
		tid_top[k+2] = tid + k + blockDim.x;
		tid_bottom[k+2] = tid + k;
	}

	__syncthreads();

	if (tid > 1 && tid <(blockDim.x-2))
	{
		temper[tid_top[2]] = stutterStep(temper[tid_bottom[0]], temper[tid_bottom[1]], temper[tid_bottom[2]],
			temper[tid_bottom[3]], temper[tid_bottom[4]]);
	}

	__syncthreads();

	//The initial conditions are timslice 0 so start k at 1.
	for (int k = 4; k<(blockDim.x/2); k+=4)
	{
		if (tid < (blockDim.x-k) && tid >= k)
		{
			temper[tid] += finalStep(temper[tid_top[0]], temper[tid_top[1]], temper[tid_top[2]],
				temper[tid_top[3]], temper[tid_top[4]]);
		}

		step2 = k+2;
		__syncthreads();

		if (tid < (blockDim.x-step2) && tid >= step2)
		{
			temper[tid_top[2]] = stutterStep(temper[tid_bottom[0]], temper[tid_bottom[1]], temper[tid_bottom[2]],
				temper[tid_bottom[3]], temper[tid_bottom[4]]);
		}

		//Make sure the threads are synced
		__syncthreads();

	}

	//After the triangle has been computed, the right and left shared arrays are
	//stored in global memory by the global thread ID since (conveniently),
	//they're the same size as a warp!
	right[gid] = temper[rightidx];
	left[gid] = temper[leftidx];

}

double
classicWrapper(const int bks, int tpb, const int dv, const REAL dt, const REAL t_end,
    REAL *IC, REAL *T_f, const REAL freq, ofstream &fwr)
{
    REAL *dks_in, *dks_out, *dks_orig;

    cudaMalloc((void **)&dks_in, sizeof(REAL)*dv);
    cudaMalloc((void **)&dks_out, sizeof(REAL)*dv);

    // Copy the initial conditions to the device array.
    cudaMemcpy(dks_in,IC,sizeof(REAL)*dv,cudaMemcpyHostToDevice);

    double t_eq = 0.0;
    double twrite = freq;

    while (t_eq < t_end)
    {
        classicKS <<< bks,tpb >>> (dks_in, dks_out, false)
        classicKS <<< bks,tpb >>> (dks_out, dks_in, true)
        t_eq += dt;

        if (t_eq > twrite)
        {
            cudaMemcpy(T_f, dks_in, sizeof(REAL)*dv, cudaMemcpyDeviceToHost);

			fwr << " Velocity " << t_eq << " ";
            for (int k = 0; k<dv; k++)
            {
                fwr << T_f[k] << " ";
            }
            fwr << endl;

            twrite += freq;
        }
    }

    cudaMemcpy(T_f, dks_in, sizeof(REAL)*dv, cudaMemcpyDeviceToHost);

    cudaFree(dks_in);
    cudaFree(dks_out);

    return t_eq;
}

//The host routine.
double
sweptWrapper(const int bks, int tpb, const int dv, REAL dt, const REAL t_end,
	REAL *IC, REAL *T_f, const REAL freq, ofstream &fwr)
{

	REAL *d_IC, *d_right, *d_left, *d_bin;
	cudaMalloc((void **)&d_IC, sizeof(REAL)*dv);
	cudaMalloc((void **)&d_right, sizeof(REAL)*dv);
	cudaMalloc((void **)&d_left, sizeof(REAL)*dv);
	cudaMalloc((void **)&d_bin, sizeof(REAL)*dv);

	// Copy the initial conditions to the device array.
	cudaMemcpy(d_IC,IC,sizeof(REAL)*dv,cudaMemcpyHostToDevice);
	//Start the counter and start the clock.
	//
	//Every other step is a full timestep and each cycle is half tpb steps.
	const double t_fullstep = 0.25 * dt * (double)tpb;
	double twrite = freq;

	const size_t smem1 = 2*tpb*sizeof(REAL);
	const size_t smem2 = (2*tpb+8)*sizeof(REAL);

	upTriangle <<< bks,tpb,smem1 >>> (d_IC,d_right,d_left);

	swapKernel <<< bks,tpb >>> (d_right, d_bin, 1);
	swapKernel <<< bks,tpb >>> (d_bin, d_right, 0);

	//Split
	wholeDiamond <<< bks,tpb,smem2 >>> (d_right,d_left);

	swapKernel <<< bks,tpb >>> (d_left, d_bin, -1);
	swapKernel <<< bks,tpb >>> (d_bin, d_left, 0);

	double t_eq = t_fullstep;

	// Call the kernels until you reach the iteration limit.
	while(t_eq < t_end)
	{

		wholeDiamond <<< bks,tpb,smem2 >>> (d_right,d_left);

		swapKernel <<< bks,tpb >>> (d_right, d_bin, 1);
		swapKernel <<< bks,tpb >>> (d_bin, d_right, 0);

		//So it always ends on a left pass since the down triangle is a right pass.

		//Split
		wholeDiamond <<< bks,tpb,smem2 >>> (d_right,d_left);

		swapKernel <<< bks,tpb >>> (d_left, d_bin, -1);
		swapKernel <<< bks,tpb >>> (d_bin, d_left, 0);

		t_eq += t_fullstep;


	 	if (t_eq > twrite)
		{
			downTriangle <<< bks,tpb,smem2 >>>(d_IC,d_right,d_left);

			cudaMemcpy(T_f, d_IC, sizeof(REAL)*dv, cudaMemcpyDeviceToHost);

			fwr << " Velocity " << t_eq << " ";

			for (int k = 0; k<dv; k++)	fwr << T_f[k] << " ";

			fwr << endl;

			upTriangle <<< bks,tpb,smem1 >>>(d_IC,d_right,d_left);

			swapKernel <<< bks,tpb >>> (d_right, d_bin, 1);
			swapKernel <<< bks,tpb >>> (d_bin, d_right, 0);

			//Split
			wholeDiamond <<< bks,tpb,smem2 >>>(d_right,d_left);

			swapKernel <<< bks,tpb >>> (d_left, d_bin, -1);
			swapKernel <<< bks,tpb >>> (d_bin, d_left, 0);

			t_eq += t_fullstep;

			twrite += freq;
		}

	}

	downTriangle <<< bks,tpb,smem2 >>>(d_IC,d_right,d_left);

	cudaMemcpy(T_f, d_IC, sizeof(REAL)*dv, cudaMemcpyDeviceToHost);

	cudaFree(d_IC);
	cudaFree(d_right);
	cudaFree(d_left);
	cudaFree(d_bin);

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

	const int dv = atoi(argv[1]); //Number of spatial points
	const int tpb = atoi(argv[2]); //Threads per Block
    const REAL dt = atof(argv[3]); //delta T timestep
	const float tf = atof(argv[4]); //Finish time
    const float freq = atof(argv[5]); //Output frequency
    const int scheme = atoi(argv[6]); //1 for Swept 0 for classic
    // const int tst = atoi(argv[7]); CPU/GPU share
    const int bks = dv/tpb; //The number of blocks
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
		dx, //dx
		dx*dx, //dx^2
		dx*dx*dx*dx, //dx^4
		dt, //dt
		dt*.05, //dt half
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
		IC[k] = initFun((float)k*dsc.dx);
	}

	// Call out the file before the loop and write out the initial condition.
	ofstream fwr;
	fwr.open(argv[8],ios::trunc);

	// Write out x length and then delta x and then delta t.
	// First item of each line is timestamp.
	fwr << lx << " " << dv << " " << dsc.dx << " " << endl << " Velocity " << 0 << " ";

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
	if (scheme)
    {
		tfm = sweptWrapper(bks, tpb, dv, dsc.dt, tf, IC, T_final, freq, fwr);
	}
	else
	{
		tfm = classicWrapper(bks, tpb, dv, dsc.dt, tf, IC, T_final, freq, fwr);
	}

	// Show the time and write out the final condition.
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime( &timed, start, stop);

	timed *= 1.e3;

	double n_timesteps = tfm/dt;

    double per_ts = timed/n_timesteps;

    cout << n_timesteps << " timesteps" << endl;
	cout << "Averaged " << per_ts << " microseconds (us) per timestep" << endl;

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
	// free(IC);
	// free(T_final);

	return 0;

}
