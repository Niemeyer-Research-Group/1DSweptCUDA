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
    #define REAL        float
    #define TWO         2.f
	#define FOUR        4.f
	#define SIX			6.f
#else
    #define TWO         2.0
	#define FOUR        4.0
	#define SIX			6.0
#endif

#define BASE            36
#define HEIGHT          18
#define WARPSIZE        32
#define TPB             256
#define WPB             8
#define TWOBASE         72

using namespace std;

const REAL dx = 0.5;

struct discConstants{

	REAL dx_i4; // 1/(4*dx)
	REAL dx2_i; // 1/(dx^2)
	REAL dx4_i; // 1/(dx^4)
	REAL dt; // dt
	REAL dt_half; // dt/2
};

__constant__ discConstants disc;

__host__
REAL initFun(REAL xnode)
{
	return TWO * cos(19.0*xnode*M_PI/128.0);
}

__device__
__forceinline__
REAL fourthDer(REAL tfarLeft, REAL tLeft, REAL tCenter, REAL tRight, REAL tfarRight)
{
	return disc.dx4_i * (tfarLeft - FOUR*tLeft + SIX*tCenter - FOUR*tRight + tfarRight);
}

__device__
__forceinline__
REAL secondDer(REAL tLeft, REAL tRight, REAL tCenter)
{
	return disc.dx2_i * (tLeft + tRight - TWO*tCenter);
}

__device__
__forceinline__
REAL convect(REAL tLeft, REAL tRight)
{
	return disc.dx_i4 * (tRight*tRight - tLeft*tLeft);
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
		ks_out[gid] += finalStep(ks_in[(gid-2)&lastidx], ks_in[(gid-1)&lastidx],
			ks_in[gid], ks_in[(gid+1)&lastidx], ks_in[(gid+2)&lastidx]);
	}
	else
	{
		ks_out[gid] = stutterStep(ks_in[(gid-2)&lastidx], ks_in[(gid-1)&lastidx], ks_in[gid],
			ks_in[(gid+1)&lastidx], ks_in[(gid+2)&lastidx]);
	}
}

__global__
void
upTriangle(const REAL *IC, REAL *right, REAL *left)
{
	__shared__ REAL temper[WPB][TWOBASE];

	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	int wid = threadIdx.x & 31; //Thread id in warp.
    int wtag = threadIdx.x/TPB; //Warp id in block.
    int widx = wid + 2;
    int widTop = widx+BASE;

	int leftidx = (((wid>>2) & 1) * BASE) + ((wid>>2)<<1) + (wid & 3) + 2;
	int rightidx = 30 + (((wid>>2) & 1) * BASE) + (wid & 3) - ((tid>>2)<<1);

    REAL vel[2];
    //Assign the initial values to the first row in temper, each block
    //has it's own version of temper shared among its threads.
	vel[0] = IC[gid];

	__syncthreads();

    vel[1] = stutterStep(__shfl_up(vel[0],2),__shfl_up(vel[0],1),vel[0],
        __shfl_down(vel[0],1),__shfl_down(vel[0],2));

    temper[wtag][widx] = vel[0];
    temper[wtag][widTop] = vel[1];

	__syncthreads();

    vel[0] += finalStep(__shfl_up(vel[1],2),__shfl_up(vel[1],1),vel[1],
        __shfl_down(vel[1],1),__shfl_down(vel[1],2));

    if (wid > 3 && wid < 28) temper[wtag][widx] = vel[0];

    vel[1] = stutterStep(__shfl_up(vel[0],2),__shfl_up(vel[0],1),vel[0],
        __shfl_down(vel[0],1),__shfl_down(vel[0],2));

    if (wid > 5 && wid < 26) temper[wtag][widTop] = vel[1];

    vel[0] += finalStep(__shfl_up(vel[1],2),__shfl_up(vel[1],1),vel[1],
        __shfl_down(vel[1],1),__shfl_down(vel[1],2));

    if (wid > 7 && wid < 24) temper[wtag][widx] = vel[0];

    vel[1] = stutterStep(__shfl_up(vel[0],2),__shfl_up(vel[0],1),vel[0],
        __shfl_down(vel[0],1),__shfl_down(vel[0],2));

    if (wid > 9 && wid < 22) temper[wtag][widTop] = vel[1];

    vel[0] += finalStep(__shfl_up(vel[1],2),__shfl_up(vel[1],1),vel[1],
        __shfl_down(vel[1],1),__shfl_down(vel[1],2));

    if (wid > 11 && wid < 20) temper[wtag][widx] = vel[0];

    vel[1] = stutterStep(__shfl_up(vel[0],2),__shfl_up(vel[0],1),vel[0],
        __shfl_down(vel[0],1),__shfl_down(vel[0],2));

    if (wid > 13 && wid < 18) temper[wtag][widTop] = vel[1];

	//Make sure the threads are synced
	__syncthreads();

	//After the triangle has been computed, the right and left shared arrays are
	//stored in global memory by the global thread ID since (conveniently),
	//they're the same size as a warp!
	right[gid] = temper[wtag][rightidx];
	left[gid] = temper[wtag][leftidx];
}

__global__
void
downTriangle(REAL *IC, const REAL *right, const REAL *left)
{
    __shared__ REAL temper[WPB][TWOBASE];

	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	int wid = threadIdx.x & 31; //Thread id in warp.
    int wtag = threadIdx.x/TPB; //Warp id in block.
    int widx = wid + 2;
    int widTop = wid+BASE;

    int leftidx = HEIGHT + (((wid>>2) & 1) * BASE) + (wid & 3) - (4 + ((wid>>2) << 1));
	int rightidx = HEIGHT + (((wid>>2) & 1) * BASE) + ((wid>>2)<<1) + (wid & 3);

	temper[wtag][leftidx] = right[gid];
	temper[wtag][rightidx] = left[gid];

    //stutter first
    vel[0] = temper[wtag][widx];

    __syncthreads();

    vel[1] = stutterStep(__shfl_up(vel[0],2),__shfl_up(vel[0],1),vel[0],
        __shfl_down(vel[0],1),__shfl_down(vel[0],2));

    if (wid < 14 || wid > 17) vel[1] = temper[wtag][widTop];

    __syncthreads();

    vel[0] += finalStep(__shfl_up(vel[1],2),__shfl_up(vel[1],1),vel[1],
        __shfl_down(vel[1],1),__shfl_down(vel[1],2));

    if (wid < 12 || wid > 19) vel[0] = temper[wtag][widx];

    __syncthreads();

    vel[1] = stutterStep(__shfl_up(vel[0],2),__shfl_up(vel[0],1),vel[0],
        __shfl_down(vel[0],1),__shfl_down(vel[0],2));

    if (wid < 10 || wid > 21) vel[1] = temper[wtag][widTop];

    __syncthreads();

    vel[0] += finalStep(__shfl_up(vel[1],2),__shfl_up(vel[1],1),vel[1],
        __shfl_down(vel[1],1),__shfl_down(vel[1],2));

    if (wid < 8 || wid > 23) vel[0] = temper[wtag][widx];

    __syncthreads();

    vel[1] = stutterStep(__shfl_up(vel[0],2),__shfl_up(vel[0],1),vel[0],
        __shfl_down(vel[0],1),__shfl_down(vel[0],2));

    if (wid < 6 || wid > 25) vel[1] = temper[wtag][widTop];

    __syncthreads();

    vel[0] += finalStep(__shfl_up(vel[1],2),__shfl_up(vel[1],1),vel[1],
        __shfl_down(vel[1],1),__shfl_down(vel[1],2));

    if (wid < 4 || wid > 27) vel[0] = temper[wtag][widx];

    __syncthreads();

    vel[1] = stutterStep(__shfl_up(vel[0],2),__shfl_up(vel[0],1),vel[0],
        __shfl_down(vel[0],1),__shfl_down(vel[0],2));

    if (wid < 1 || wid > 30) temper[wtag][widTop] = vel[1];

    __syncthreads();

    //This is where to do it with shared mem.
    vel[0] += finalStep(temper[wtag][widTop-2],temper[wtag][widTop-1],temper[wtag][widTop],
        temper[wtag][widTop+1],temper[wtag][widTop+2]);

    IC[gid] = vel[0];
}


__global__
void
wholeDiamond(REAL *right, REAL *left)
{
    __shared__ REAL temper[WPB][TWOBASE];

	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	int wid = threadIdx.x & 31; //Thread id in warp.
    int wtag = threadIdx.x/TPB; //Warp id in block.
    int widx = wid+2;
    int widxTop = widx+BASE;

    int leftidx = HEIGHT + (((wid>>2) & 1) * BASE) + (wid & 3) - (4 + ((wid>>2) << 1));
	int rightidx = HEIGHT + (((wid>>2) & 1) * BASE) + ((wid>>2)<<1) + (wid & 3);

	temper[wtag][leftidx] = right[gid];
	temper[wtag][rightidx] = left[gid];

    //stutter first
    vel[0] = temper[wtag][widx];

    __syncthreads();

    vel[1] = stutterStep(__shfl_up(vel[0],2),__shfl_up(vel[0],1),vel[0],
        __shfl_down(vel[0],1),__shfl_down(vel[0],2));

    if (wid < 14 || wid > 17) vel[1] = temper[wtag][widTop];

    __syncthreads();

    vel[0] += finalStep(__shfl_up(vel[1],2),__shfl_up(vel[1],1),vel[1],
        __shfl_down(vel[1],1),__shfl_down(vel[1],2));

    if (wid < 12 || wid > 19) vel[0] = temper[wtag][widx];

    __syncthreads();

    vel[1] = stutterStep(__shfl_up(vel[0],2),__shfl_up(vel[0],1),vel[0],
        __shfl_down(vel[0],1),__shfl_down(vel[0],2));

    if (wid < 10 || wid > 21) vel[1] = temper[wtag][widTop];

    __syncthreads();

    vel[0] += finalStep(__shfl_up(vel[1],2),__shfl_up(vel[1],1),vel[1],
        __shfl_down(vel[1],1),__shfl_down(vel[1],2));

    if (wid < 8 || wid > 23) vel[0] = temper[wtag][widx];

    __syncthreads();

    vel[1] = stutterStep(__shfl_up(vel[0],2),__shfl_up(vel[0],1),vel[0],
        __shfl_down(vel[0],1),__shfl_down(vel[0],2));

    if (wid < 6 || wid > 25) vel[1] = temper[wtag][widTop];

    __syncthreads();

    vel[0] += finalStep(__shfl_up(vel[1],2),__shfl_up(vel[1],1),vel[1],
        __shfl_down(vel[1],1),__shfl_down(vel[1],2));

    if (wid < 4 || wid > 27) vel[0] = temper[wtag][widx];

    __syncthreads();

    vel[1] = stutterStep(__shfl_up(vel[0],2),__shfl_up(vel[0],1),vel[0],
        __shfl_down(vel[0],1),__shfl_down(vel[0],2));

    if (wid < 1 || wid > 30) temper[wtag][widTop] = vel[1];

    __syncthreads();

    //This is where to do it with shared mem.
    vel[0] += finalStep(temper[wtag][widTop-2],temper[wtag][widTop-1],temper[wtag][widTop],
        temper[wtag][widTop+1],temper[wtag][widTop+2]);

    leftidx = (((wid>>2) & 1) * BASE) + ((wid>>2)<<1) + (wid & 3) + 2;
    rightidx = 30 + (((wid>>2) & 1) * BASE) + (wid & 3) - ((tid>>2)<<1);

    __syncthreads();

    vel[1] = stutterStep(__shfl_up(vel[0],2),__shfl_up(vel[0],1),vel[0],
        __shfl_down(vel[0],1),__shfl_down(vel[0],2));

    temper[wtag][widx] = vel[0];
    temper[wtag][widTop] = vel[1];

	__syncthreads();

    vel[0] += finalStep(__shfl_up(vel[1],2),__shfl_up(vel[1],1),vel[1],
        __shfl_down(vel[1],1),__shfl_down(vel[1],2));

    if (wid > 3 && wid < 28) temper[wtag][widx] = vel[0];

    vel[1] = stutterStep(__shfl_up(vel[0],2),__shfl_up(vel[0],1),vel[0],
        __shfl_down(vel[0],1),__shfl_down(vel[0],2));

    if (wid > 5 && wid < 26) temper[wtag][widTop] = vel[1];

    vel[0] += finalStep(__shfl_up(vel[1],2),__shfl_up(vel[1],1),vel[1],
        __shfl_down(vel[1],1),__shfl_down(vel[1],2));

    if (wid > 7 && wid < 24) temper[wtag][widx] = vel[0];

    vel[1] = stutterStep(__shfl_up(vel[0],2),__shfl_up(vel[0],1),vel[0],
        __shfl_down(vel[0],1),__shfl_down(vel[0],2));

    if (wid > 9 && wid < 22) temper[wtag][widTop] = vel[1];

    vel[0] += finalStep(__shfl_up(vel[1],2),__shfl_up(vel[1],1),vel[1],
        __shfl_down(vel[1],1),__shfl_down(vel[1],2));

    if (wid > 11 && wid < 20) temper[wtag][widx] = vel[0];

    vel[1] = stutterStep(__shfl_up(vel[0],2),__shfl_up(vel[0],1),vel[0],
        __shfl_down(vel[0],1),__shfl_down(vel[0],2));

    if (wid > 13 && wid < 18) temper[wtag][widTop] = vel[1];

	//Make sure the threads are synced
	__syncthreads();

	//After the triangle has been computed, the right and left shared arrays are
	//stored in global memory by the global thread ID since (conveniently),
	//they're the same size as a warp!
	right[gid] = temper[wtag][rightidx];
	left[gid] = temper[wtag][leftidx];
}

double
classicWrapper(const int bks, int tpb, const int dv, const REAL dt, const REAL t_end,
    REAL *IC, REAL *T_f, const REAL freq, ofstream &fwr)
{
    REAL *dks_in, *dks_out;

    cudaMalloc((void **)&dks_in, sizeof(REAL)*dv);
    cudaMalloc((void **)&dks_out, sizeof(REAL)*dv);

    // Copy the initial conditions to the device array.
    cudaMemcpy(dks_in,IC,sizeof(REAL)*dv,cudaMemcpyHostToDevice);

    double t_eq = 0.0;
    double twrite = freq;

    while (t_eq <= t_end)
    {
        classicKS <<< bks,tpb >>> (dks_in, dks_out, false);
        classicKS <<< bks,tpb >>> (dks_out, dks_in, true);
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
	if (sizeof(REAL)>6) cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

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
		1.0/(FOUR*dx),
		1.0/(dx*dx),
		1.0/(dx*dx*dx*dx),
		dt,
		dt*0.5
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
		IC[k] = initFun((float)k*dx);
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
	if (scheme)
    {
		cout << "Swept" << endl;
		tfm = sweptWrapper(bks, tpb, dv, dsc.dt, tf, IC, T_final, freq, fwr);
	}
	else
	{
		cout << "Classic" << endl;
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
