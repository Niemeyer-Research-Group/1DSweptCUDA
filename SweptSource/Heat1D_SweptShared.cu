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

//COMPILE LINE:
// nvcc -o ./bin/HeatOut Heat1D_SweptShared.cu -gencode arch=compute_35,code=sm_35 -lm -restrict -Xcompiler -fopenmp


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include "myVectorTypes.h" //For clamp.

#include <ostream>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <omp.h>

#ifndef REAL
    #define REAL        float
    #define HALF        0.5f
    #define ONE         1.f
    #define TWO         2.f
#else
    #define HALF        0.5
    #define ONE         1.0
    #define TWO         2.0
#endif

using namespace std;

struct heatConstants{

    REAL fourier;
    REAL fourierTwo;
    int base;
	int ht;
    int idxend;
};

heatConstants hostC;

__constant__ heatConstants gpuC;

const REAL th_diff = 8.418e-5;

const REAL ds = 0.001;

__host__ __device__
__forceinline__
REAL initFun(int xnode, REAL ds, REAL lx)
{
    REAL a = ((REAL)xnode*ds);
    return 100.f*a*(ONE-a/lx);
}

//Read in the data from the global right/left variables to the shared temper variable.
__host__ __device__
__forceinline__
void
readIn(REAL *temp, const REAL *rights, const REAL *lefts, int td, int gd)
{
    #ifdef __CUDA_ARCH__    
	int leftidx = gpuC.ht - (td>>1) + (((td>>1) & 1) * gpuC.base) + (td & 1) - 2;
	int rightidx = gpuC.ht + (td>>1) + (((td>>1) & 1) * gpuC.base) + (td & 1);
    #else
	int leftidx = hostC.ht - (td>>1) + (((td>>1) & 1) * hostC.base) + (td & 1) - 2;
	int rightidx = hostC.ht + (td>>1) + (((td>>1) & 1) * hostC.base) + (td & 1);
    #endif
	temp[leftidx] = rights[gd];
	temp[rightidx] = lefts[gd];
}

__host__ __device__
__forceinline__
void
writeOutRight(REAL *temp, REAL *rights, REAL *lefts, int td, int gd, int bd)
{
    #ifdef __CUDA_ARCH__
	int gdskew = (gd + bd) & gpuC.idxend;
    int leftidx = (td>>1) + (((td>>1) & 1) * gpuC.base) + (td & 1) + 1;
	int rightidx = (gpuC.base - 3) + (((td>>1) & 1) * gpuC.base) + (td & 1) -  (td>>1);
    #else
	int gdskew = gd;
    int leftidx = (td>>1) + (((td>>1) & 1) * hostC.base) + (td & 1) + 1;
	int rightidx = (hostC.base - 3) + (((td>>1) & 1) * hostC.base) + (td & 1) -  (td>>1);
    #endif
	rights[gdskew] = temp[rightidx];
	lefts[gd] = temp[leftidx];
}

__host__ __device__
__forceinline__
void
writeOutLeft(REAL *temp, REAL *rights, REAL *lefts, int td, int gd, int bd)
{
    #ifdef __CUDA_ARCH__
	int gdskew = (gd - bd) & gpuC.idxend;
    int leftidx = (td>>1) + (((td>>1) & 1) * gpuC.base) + (td & 1) + 1;
	int rightidx = (gpuC.base - 3) + (((td>>1) & 1) * gpuC.base) + (td & 1) -  (td>>1);
    #else
	int gdskew = gd;
    int leftidx = (td>>1) + (((td>>1) & 1) * hostC.base) + (td & 1) + 1;
	int rightidx = (hostC.base - 3) + (((td>>1) & 1) * hostC.base) + (td & 1) -  (td>>1);
    #endif
	rights[gd] = temp[rightidx];
	lefts[gdskew] = temp[leftidx];
}

__host__ __device__
__forceinline__
REAL execFunc(const REAL *heat, int idx[3])
{
    #ifdef __CUDA_ARCH__
    return gpuC.fourier*(heat[idx[0]] + heat[idx[2]]) + gpuC.fourierTwo * heat[idx[1]];
    #else 
    return hostC.fourier*(heat[idx[0]] + heat[idx[2]]) + hostC.fourierTwo * heat[idx[1]];
    #endif
}

__global__
void
classicHeat(const REAL *heat_in, REAL *heat_out)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
    int gidz[3];

    //This doesn't work just do it manually
    gidz[0] = (gid == 0) ? (gid + 1) : (gid - 1);
    gidz[1] = gid;
    gidz[2] = (gid == gpuC.idxend) ? (gid - 1) : (gid + 1);

    heat_out[gid] =  execFunc(heat_in, gidz);
}

__global__
void
upTriangle(const REAL *IC, REAL *outRight, REAL *outLeft)
{
	extern __shared__ REAL temper[];

	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tididx = threadIdx.x + 1;

	int tid_top[3], tid_bottom[3];
    int k = blockDim.x-1;

    #pragma unroll
    for (int a=-1; a<2; a++)
    {
	    tid_bottom[a+1] = tididx + a;
        tid_top[a+1] = tididx + a + gpuC.base;
    }

	temper[tididx] = IC[gid];

    __syncthreads();

    //upTriangle
	while (k > gpuC.ht)
	{
        if (tididx > (blockDim.x-k) && tididx <= k)
        {
			temper[tid_top[1]] = execFunc(temper, tid_bottom);
		}
        k--;
        __syncthreads();

        if (tididx > (blockDim.x-k) && tididx <= k)
		{
			temper[tididx] = execFunc(temper, tid_top);
		}
        k--;
        __syncthreads();      
	}

    if (tididx > (blockDim.x-k) && tididx <= k)
    {
        temper[tid_top[1]] = execFunc(temper, tid_bottom);
    }

    __syncthreads(); 

	writeOutRight(temper, outRight, outLeft, threadIdx.x, gid, blockDim.x);
}

// Down triangle is only called at the end when data is passed left.  It's never split.
// It returns IC which is a full 1D result at a certain time.
__global__
void
downTriangle(REAL *IC, const REAL *inRight, const REAL *inLeft)
{
    extern __shared__ REAL temper[];

	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tididx = threadIdx.x + 1;

	int tid_top[3], tid_bottom[3];

    int k = gpuC.ht;
	tid_bottom[1] = tididx;
    tid_top[1] = tididx + gpuC.base;

    tid_bottom[0] = (gid) ? (tididx - 1) : (tididx + 1);
    tid_bottom[2] = (gid == gpuC.idxend) ? (tididx - 1) : (tididx + 1);

    readIn(temper, inRight, inLeft, threadIdx.x, gid);
 
    tid_top[0] = tid_bottom[0] + gpuC.base;
    tid_top[2] = tid_bottom[2] + gpuC.base;

    //downTriangle
    __syncthreads();

	while (k < blockDim.x)
	{
        if (tididx > (blockDim.x-k) && tididx <= k)
        {
			temper[tid_top[1]] = execFunc(temper, tid_bottom);
		}
        k++;
        __syncthreads();

        if (tididx > (blockDim.x-k) && tididx <= k)
		{
			temper[tididx] = execFunc(temper, tid_top);
		}
        k++;
        __syncthreads();      
	}

    IC[gid] = temper[tididx];
}

//Full refers to whether or not there is a node run on the CPU.
__global__
void
wholeDiamond(const REAL *inRight, const REAL *inLeft, REAL *outRight, REAL *outLeft, const bool split)
{
    extern __shared__ REAL temper[];

	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tididx = threadIdx.x + 1;

	int tid_top[3], tid_bottom[3];

    int k = gpuC.ht;
	tid_bottom[1] = tididx;
    tid_top[1] = tididx + gpuC.base;

    if (split)
    {
        gid += blockDim.x;
        tid_bottom[0] = tididx - 1;
        tid_bottom[2] = tididx + 1;
    }
    else
    {
        tid_bottom[0] = (gid) ? (tididx - 1) : (tididx + 1);
        tid_bottom[2] = (gid == gpuC.idxend) ? (tididx - 1) : (tididx + 1);
    }

    readIn(temper, inRight, inLeft, threadIdx.x, gid);
 
    tid_top[0] = tid_bottom[0] + gpuC.base;
    tid_top[2] = tid_bottom[2] + gpuC.base;

    //downTriangle
    __syncthreads();

	while (k < blockDim.x)
	{
        if (tididx > (blockDim.x-k) && tididx <= k)
        {
			temper[tid_top[1]] = execFunc(temper, tid_bottom);
		}
        k++;
        __syncthreads();

        if (tididx > (blockDim.x-k) && tididx <= k)
		{
			temper[tididx] = execFunc(temper, tid_top);
		}
        k++;
        __syncthreads();      
	}

    k -= 2; 

    //upTriangle
	while (k > gpuC.ht)
	{
        if (tididx > (blockDim.x-k) && tididx <= k)
        {
			temper[tid_top[1]] = execFunc(temper, tid_bottom);
		}
        k--;
        __syncthreads();

        if (tididx > (blockDim.x-k) && tididx <= k)
		{
			temper[tididx] = execFunc(temper, tid_top);
		}
        k--;
        __syncthreads();      
	}

    if (tididx > (blockDim.x-k) && tididx <= k)
    {
        temper[tid_top[1]] = execFunc(temper, tid_bottom);
    }
    __syncthreads();  

	if (split)
	{
		writeOutLeft(temper, outRight, outLeft, threadIdx.x, gid, blockDim.x);
	}
	else
	{
		writeOutRight(temper, outRight, outLeft, threadIdx.x, gid, blockDim.x);
	}
}

//Split one is always first.  Passing left like the downTriangle.  downTriangle
//should be rewritten so it isn't split.  Only write on a non split pass.
__global__
void
splitDiamond(const REAL *inRight, const REAL *inLeft, REAL *outRight, REAL *outLeft)
{
    extern __shared__ REAL temper[];

	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tididx = threadIdx.x + 1;

	int tid_top[3], tid_bottom[3];

    int k = gpuC.ht;
    tid_bottom[1] = tididx;
    tid_top[1] = tididx + gpuC.base;

    tid_bottom[2] = (gid == (gpuC.ht-2)) ? (tididx-1) : (tididx+1);
    tid_bottom[0] = (gid == (gpuC.ht-1)) ? (tididx+1) : (tididx-1);

    readIn(temper, inRight, inLeft, threadIdx.x, gid);
 
    tid_top[0] = tid_bottom[0] + gpuC.base;
    tid_top[2] = tid_bottom[2] + gpuC.base;

    //upTriangle
    __syncthreads();

	while (k < blockDim.x)
	{
        if (tididx > (blockDim.x-k) && tididx <= k)
        {
			temper[tid_top[1]] = execFunc(temper, tid_bottom);
		}
        k++;
        __syncthreads();

        if (tididx > (blockDim.x-k) && tididx <= k)
		{
			temper[tididx] = execFunc(temper, tid_top);
		}
        k++;
        __syncthreads();      
	}

    k -= 2; 

    //downTriangle
	while (k > gpuC.ht)
	{
        if (tididx > (blockDim.x-k) && tididx <= k)
        {
			temper[tid_top[1]] = execFunc(temper, tid_bottom);
		}
        k--;
        __syncthreads();

        if (tididx > (blockDim.x-k) && tididx <= k)
		{
			temper[tididx] = execFunc(temper, tid_top);
		}
        k--;
        __syncthreads();      
	}

    if (tididx > (blockDim.x-k) && tididx <= k)
    {
        temper[tid_top[1]] = execFunc(temper, tid_bottom);
    }
    __syncthreads();  

	writeOutLeft(temper, outRight, outLeft, threadIdx.x, gid, blockDim.x);

}

__host__
void
CPU_diamond(REAL *temper, const int tpb)
{
    int stencil[hostC.base][3], stencil_top[hostC.base][3];
    int k = hostC.ht;
    int htL = hostC.ht-1;
    int tpbo = tpb+1;

    for (int a=1; a<(tpb+2); a++)
    {
        stencil[a][0] = (a == hostC.ht) ? (a+1) : (a-1);
        stencil[a][1] = a;
        stencil[a][2] = (a == htL) ? (a-1) : (a+1);
    }

    for (int a=1; a<(tpb+2); a++)
    {
        for(int b=0; b<3; b++) stencil_top[a][b] = stencil[a][b] + hostC.base;
    }

    while (k < tpb)
    {
        for(int n=(tpbo-k); n<=k; n++)
        {
            temper[n + hostC.base] = execFunc(temper, stencil[n]); 
        }

        k++;
        for(int n=(tpbo-k); n<=k; n++)
        {
            temper[n] = execFunc(temper, stencil_top[n]);  
        }
        k++;
    }

    k -= 2; 

    while (k > hostC.ht)
    {
        for(int n=(tpbo-k); n<=k; n++)
        {
            temper[n + hostC.base] = execFunc(temper, stencil[n]); 
        }

        k--;
        for(int n=(tpbo-k); n<=k; n++)
        {
            temper[n] = execFunc(temper, stencil_top[n]);  
        }
        k--;
    }

    for(int n=(tpbo-k); n<=k; n++)
    {
        temper[n + hostC.base] = execFunc(temper, stencil[n]); 
    }

}

//Classic Discretization wrapper.
double
classicWrapper(const int bks, int tpb, const int dv, const double dt, const double t_end,
    REAL *IC, REAL *T_f, const double freq, ofstream &fwr)
{
    REAL *dheat_in, *dheat_out;

    cudaMalloc((void **)&dheat_in, sizeof(REAL)*dv);
    cudaMalloc((void **)&dheat_out, sizeof(REAL)*dv);

    // Copy the initial conditions to the device array.
    cudaMemcpy(dheat_in,IC, sizeof(REAL)*dv, cudaMemcpyHostToDevice);

    const double t_fullstep = dt+dt;
    double twrite = freq;
    classicHeat <<< bks,tpb >>> (dheat_in, dheat_out);
    classicHeat <<< bks,tpb >>> (dheat_out, dheat_in);

    double t_eq = t_fullstep;

    while (t_eq < t_end)
    {
        classicHeat <<< bks,tpb >>> (dheat_in, dheat_out);
        classicHeat <<< bks,tpb >>> (dheat_out, dheat_in);
        t_eq += t_fullstep;

        if (t_eq > twrite)
        {
            cudaMemcpy(T_f, dheat_in, sizeof(REAL)*dv, cudaMemcpyDeviceToHost);
            fwr << " Temperature " << t_eq << " ";

            for (int k = 0; k<dv; k++)   fwr << T_f[k] << " ";

            fwr << endl;

            t_eq += t_fullstep;

            twrite += freq;
        }
    }

    cout << t_eq << " " << t_end << " " << t_fullstep << endl;

    cudaMemcpy(T_f, dheat_in, sizeof(REAL)*dv, cudaMemcpyDeviceToHost);

    cudaFree(dheat_in);
    cudaFree(dheat_out);

    return t_eq;
}

//The Swept Rule wrapper.
double
sweptWrapper(const int bks, int tpb, const int dv, const double dt, const double t_end, const int cpu,
    REAL *IC, REAL *T_f, const double freq, ofstream &fwr)
{
    const size_t smem = (2*hostC.base) * sizeof(REAL);
    const int cpuLoc = dv - tpb;

	REAL *d_IC, *d0_right, *d0_left, *d2_right, *d2_left;

	cudaMalloc((void **)&d_IC, sizeof(REAL)*dv);
	cudaMalloc((void **)&d0_right, sizeof(REAL)*dv);
	cudaMalloc((void **)&d0_left, sizeof(REAL)*dv);
	cudaMalloc((void **)&d2_right, sizeof(REAL)*dv);
	cudaMalloc((void **)&d2_left, sizeof(REAL)*dv);

	cudaMemcpy(d_IC,IC,sizeof(REAL)*dv,cudaMemcpyHostToDevice);

	// Start the counter and start the clock.
	const double t_fullstep = dt*(double)tpb;

	upTriangle <<<bks, tpb, smem>>> (d_IC, d0_right, d0_left);

    double t_eq;
    double twrite = freq;

	// Call the kernels until you reach the iteration limit.

    if (cpu)
    {
        REAL *h_right, *h_left;
        REAL *tmpr = (REAL*)malloc(smem);
        cudaHostAlloc((void **) &h_right, tpb*sizeof(REAL), cudaHostAllocDefault);
        cudaHostAlloc((void **) &h_left, tpb*sizeof(REAL), cudaHostAllocDefault);

        t_eq = t_fullstep;

        cudaStream_t st1, st2, st3;
        cudaStreamCreate(&st1);
        cudaStreamCreate(&st2);
        cudaStreamCreate(&st3);

        //Split Diamond Begin------

        wholeDiamond <<<bks-1, tpb, smem, st1>>>(d0_right, d0_left, d2_right, d2_left, true);

        cudaMemcpyAsync(h_left, d0_left, tpb*sizeof(REAL), cudaMemcpyDeviceToHost, st2);
        cudaMemcpyAsync(h_right, d0_right, tpb*sizeof(REAL), cudaMemcpyDeviceToHost, st3);

        cudaStreamSynchronize(st2);
        cudaStreamSynchronize(st3);

        // CPU Part Start -----

        for (int k=0; k<tpb; k++) readIn(tmpr, h_right, h_left, k, k);

        CPU_diamond(tmpr, tpb);

        for (int k=0; k<tpb; k++) writeOutLeft(tmpr, h_right, h_left, k, k, 0);
       
        cudaMemcpyAsync(d2_right, h_right, tpb*sizeof(REAL), cudaMemcpyHostToDevice, st2);
        cudaMemcpyAsync(d2_left+cpuLoc, h_left, tpb*sizeof(REAL), cudaMemcpyHostToDevice, st3);

        //Split Diamond End------

        while(t_eq < t_end)
        {
            wholeDiamond <<<bks, tpb, smem>>> (d2_right, d2_left, d0_right, d0_left, false);

            //Split Diamond Begin------

            wholeDiamond <<<bks-1, tpb, smem, st1>>> (d0_right, d0_left, d2_right, d2_left, true);

            cudaMemcpyAsync(h_left, d0_left, tpb*sizeof(REAL), cudaMemcpyDeviceToHost, st2);
            cudaMemcpyAsync(h_right, d0_right, tpb*sizeof(REAL), cudaMemcpyDeviceToHost, st3);

            cudaStreamSynchronize(st2);
            cudaStreamSynchronize(st3);

            // CPU Part Start -----

            for (int k=0; k<tpb; k++)  readIn(tmpr, h_right, h_left, k, k);

            CPU_diamond(tmpr, tpb);

            for (int k=0; k<tpb; k++)  writeOutLeft(tmpr, h_right, h_left, k, k, 0);
            
            cudaMemcpyAsync(d2_right, h_right, tpb*sizeof(REAL), cudaMemcpyHostToDevice,st2);
            cudaMemcpyAsync(d2_left+cpuLoc, h_left, tpb*sizeof(REAL), cudaMemcpyHostToDevice,st3);

            //Split Diamond End------

		    //So it always ends on a left pass since the down triangle is a right pass.

		    t_eq += t_fullstep;

            if (t_eq > twrite)
    		{
    			downTriangle <<<bks, tpb, smem>>> (d_IC, d2_right, d2_left);

    			cudaMemcpy(T_f, d_IC, sizeof(REAL)*dv, cudaMemcpyDeviceToHost);

    			fwr << "Temperature " << t_eq << " ";

    			for (int k = 0; k<dv; k++)	fwr << T_f[k] << " ";

    			fwr << endl;

                upTriangle <<< bks,tpb,smem >>>(d_IC,d0_right,d0_left);

    			splitDiamond <<< bks,tpb,smem >>>(d0_right,d0_left,d2_right,d2_left);

                t_eq += t_fullstep;

    			twrite += freq;
    		}
        }
        cudaFreeHost(h_right);
        cudaFreeHost(h_left);
        free(tmpr);
	}
    else
    {
        splitDiamond <<< bks, tpb, smem>>> (d0_right, d0_left, d2_right, d2_left);
        t_eq = t_fullstep;

        while(t_eq < t_end)
        {
            wholeDiamond <<<bks, tpb, smem>>> (d2_right, d2_left, d0_right, d0_left, false);

            splitDiamond <<<bks, tpb, smem>>> (d0_right, d0_left, d2_right, d2_left);

            //So it always ends on a left pass since the down triangle is a right pass.
            t_eq += t_fullstep;

            if (t_eq > twrite)
    		{
    			downTriangle <<< bks,tpb,smem >>>(d_IC,d2_right,d2_left);

    			cudaMemcpy(T_f, d_IC, sizeof(REAL)*dv, cudaMemcpyDeviceToHost);
    			fwr << "Temperature " << t_eq << " ";

    			for (int k = 0; k<dv; k++)	fwr << T_f[k] << " ";

    			fwr << endl;

    			upTriangle <<< bks,tpb,smem >>>(d_IC,d0_right,d0_left);

    			splitDiamond <<< bks,tpb,smem >>>(d0_right,d0_left,d2_right,d2_left);

                t_eq += t_fullstep;

    			twrite += freq;
    		}
        }
    }

	downTriangle <<<bks, tpb, smem>>> (d_IC, d2_right, d2_left);

    cout << t_eq << " " << t_end << " " << t_fullstep << endl;

	cudaMemcpy(T_f, d_IC, sizeof(REAL)*dv, cudaMemcpyDeviceToHost);

	cudaFree(d_IC);
	cudaFree(d0_right);
	cudaFree(d0_left);
    cudaFree(d2_right);
	cudaFree(d2_left);

    return t_eq;
}

int main(int argc, char *argv[])
{
    //That is: there are less than 8 arguments.
    if (argc < 9)
    {
    	cout << "The Program takes 9 inputs, #Divisions, #Threads/block, deltat, finish time, output frequency..." << endl;
        cout << "Classic/Swept, CPU sharing Y/N, Variable Output File, Timing Output File (optional)" << endl;
    	exit(-1);
    }

    cout.precision(10);
	// Choose the GPGPU.  This is device 0 in my machine which has 2 devices.
	cudaSetDevice(0);
    if (sizeof(REAL)>6) cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    int dv = atoi(argv[1]); //Number of spatial points
	const int tpb = atoi(argv[2]); //Threads per Blocks
    const double dt =  atof(argv[3]);
	const double tf = atof(argv[4]) - 0.25*dt; //Finish time
    const double freq = atof(argv[5]);
    const int scheme = atoi(argv[6]); //1 for Swept 0 for classic
    const int share = atoi(argv[7]);
	const int bks = dv/tpb; //The number of blocks
    const double lx = ds * ((double)dv - 1.0);
    double fou = th_diff*dt/(ds*ds);  //Fourier number
    char const *prec;
    prec = (sizeof(REAL)<6) ? "Single": "Double";

    hostC.fourier = fou;
    hostC.fourierTwo = ONE - TWO*fou;
    hostC.base = tpb+2;
    hostC.ht = tpb/2 + 1;
    hostC.idxend = dv-1;

    cout << "Heat --- #Blocks: " << bks << " | Length: " << lx << " | Precision: " << prec << " | Fo: " << fou << endl;

	//dv and tpb must be powers of two.  dv must be larger than tpb and divisible by
	//tpb.

	if ((dv & (tpb-1) !=0) || (tpb&31) != 0)
    {
        cout << "INVALID NUMERIC INPUT!! "<< endl;
        cout << "2nd ARGUMENT MUST BE A POWER OF TWO >= 32 AND FIRST ARGUMENT MUST BE DIVISIBLE BY SECOND" << endl;
        exit(-1);
    }

	// Initialize arrays.
    REAL *IC, *T_final;

	cudaHostAlloc((void **) &IC, dv*sizeof(REAL), cudaHostAllocDefault);
	cudaHostAlloc((void **) &T_final, dv*sizeof(REAL), cudaHostAllocDefault);

    // IC = (REAL *) malloc(dv*sizeof(REAL));
    // T_final = (REAL *) malloc(dv*sizeof(REAL));

	for (int k = 0; k<dv; k++)
	{
		IC[k] = initFun(k, ds, lx);
	}

	// Call out the file before the loop and write out the initial condition.
	ofstream fwr;
	fwr.open(argv[8], ios::trunc);
    fwr.precision(10);

	// Write out x length and then delta x and then delta t.
	// First item of each line is timestamp.
	fwr << lx << " " << dv << " " << ds << " " << endl << "Temperature " << 0 << " ";

	for (int k = 0; k<dv; k++) fwr << IC[k] << " ";

	fwr << endl;

    //Transfer data to GPU.
	// This puts the Fourier number in constant memory.
	cudaMemcpyToSymbol(gpuC, &hostC, sizeof(heatConstants));

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
        tfm = sweptWrapper(bks, tpb, dv, dt, tf, share, IC, T_final, freq, fwr);
    }
    else
    {
        cout << "Classic" << endl;
        tfm = classicWrapper(bks, tpb, dv, dt, tf, IC, T_final, freq, fwr);
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

	fwr << "Temperature " << tfm << " ";
	for (int k = 0; k<dv; k++)	fwr << T_final[k] << " ";

	fwr.close();

	// Free the memory and reset the device.

	cudaEventDestroy( start );
	cudaEventDestroy( stop );
    cudaDeviceReset();
    cudaFreeHost(IC);
    cudaFreeHost(T_final);

	return 0;
}
