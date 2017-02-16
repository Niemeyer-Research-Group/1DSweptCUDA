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

__host__
__device__
REAL initFun(int xnode, REAL ds, REAL lx)
{
    REAL a = ((REAL)xnode*ds);
    return 100.f*a*(ONE-a/lx);
}

//Read in the data from the global right/left variables to the shared temper variable.
__device__
__forceinline__
void
readIn(REAL *temp, const REAL *rights, const REAL *lefts, int td, int gd)
{
	int leftidx = gpuC.ht - (tid>>1) + (((tid>>1) & 1) * gpuC.base) + (tid & 1) - 2;
	int rightidx = gpuC.ht + (tid>>1) + (((tid>>1) & 1) * gpuC.base) + (tid & 1);

	temp[leftidx] = rights[gd];
	temp[rightidx] = lefts[gd];
}

__device__
__forceinline__
void
writeOutRight(REAL *temp, REAL *rights, REAL *lefts, int td, int gd, int bd)
{
	int gdskew = (gd + bd) & disc.idxend;
    int leftidx = (td>>1) + (((tid>>1) & 1) * gpuC.base) + (tid & 1) + 1;
	int rightidx = (bd - 1) + (((tid>>1) & 1) * gpuC.base) + (tid & 1) -  (tid>>1);
	rights[gdskew] = temp[rightidx];
	lefts[gd] = temp[leftidx];
}

__device__
__forceinline__
void
writeOutLeft(REAL *temp, REAL *rights, REAL *lefts, int td, int gd, int bd)
{
	int gdskew = (gd - bd) & disc.idxend;
    int leftidx = (td>>1) + (((tid>>1) & 1) * gpuC.base) + (tid & 1) + 1;
	int rightidx = (bd - 1) + (((tid>>1) & 1) * gpuC.base + (tid & 1) -  (tid>>1);
	rights[gdskew] = temp[rightidx];
	lefts[gd] = temp[leftidx];
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

    #pragma unroll
    for (int k=-1; k<2; k++) gidz[k+1] = clamp(gid+k, 0, gpuC.lastidx);

    heat_out[gid] =  execFunc(heat_in, gidz);

}

__global__
void
upTriangle(const REAL *IC, REAL *outRight, REAL *outLeft)
{
	extern __shared__ REAL temper[];

	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	int tididx = threadIdx.x + 1; //Block Thread ID
	int shft_wr; //Initialize the shift to the written row of temper.
	int shft_rd; //Initialize the shift to the read row (opposite of written)
	int leftidx = (tid>>1) + (((tid>>1) & 1) * blockDim.x) + (tid & 1);
	int rightidx = (blockDim.x - 2) + (((tid>>1) & 1) * blockDim.x) + (tid & 1) -  (tid>>1);
    int gidout = (gid + blockDim.x) & lastidx;

    //Assign the initial values to the first row in temper, each warp (in this
	//case each block) has it's own version of temper shared among its threads.
	temper[tid] = IC[gid];

    __syncthreads();

	//The initial conditions are timslice 0 so start k at 1.

	for (int k = 1; k<(blockDim.x>>1); k++)
	{
		//Bitwise even odd. On even iterations write to first row.
		shft_wr = blockDim.x * (k & 1);
		//On even iterations write to second row (starts at element 32)
		shft_rd = blockDim.x * ((k + 1) & 1);

		//Each iteration the triangle narrows.  When k = 1, 30 points are
		//computed, k = 2, 28 points.
		if (tid < (blockDim.x-k) && tid >= k)
		{
			temper[tid + shft_wr] = execFunc(temper[tidm+shft_rd], temper[tidp+shft_rd], temper[tid+shft_rd]);
		}

		//Make sure the threads are synced
		__syncthreads(); 

	}

	//After the triangle has been computed, the right and left shared arrays are
	//stored in global memory by the global thread ID since (conveniently),
	//they're the same size as a warp!
	writeOutRight(temper, outRight, outLeft, threadIdx.x, gid, blockDim.x);

}

// Down triangle is only called at the end when data is passed left.  It's never split.
// It returns IC which is a full 1D result at a certain time.
__global__
void
downTriangle(REAL *IC, const REAL *inRight, const REAL *inLeft)
{
	extern __shared__ REAL temper[];

	//Same as upTriangle
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	int tid1 = tid + 1;
	int tid2 = tid + 2;
	int base = blockDim.x + 2;
	int height = base>>1;
	int shft_rd;
	int shft_wr;
	int leftidx = height - (tid>>1) + (((tid>>1) & 1) * base) + (tid & 1) - 2;
	int rightidx = height + (tid>>1) + (((tid>>1) & 1) * base) + (tid & 1);
    int lastidx = ((blockDim.x*gridDim.x)-1);

	// Initialize temper. Kind of an unrolled for loop.  This is actually at
	// Timestep 0.

	temper[leftidx] = inRight[gid];
	temper[rightidx] = inLeft[gid];

    __syncthreads();
    //k needs to insert the relevant left right values around the computed values
	//every timestep.  Since it grows larger the loop is reversed.

	for (int k = (height-1); k>1; k--)
	{
		// This tells you if the current row is the first or second.
		shft_wr = base * ((k+1) & 1);
		// Read and write are opposite rows.
		shft_rd = base * (k & 1);

		if (tid1 < (base-k) && tid1 >= k)
		{
			temper[tid1 + shft_wr] = execFunc(temper[tid+shft_rd], temper[tid2+shft_rd], temper[tid1+shft_rd]);
		}
        __syncthreads();
	}

    if (gid == 0)
    {
        temper[tid] = execFunc(temper[tid2+base], temper[tid2+base], temper[tid1+base]);
    }
    else if (gid == lastidx)
    {
        temper[tid] = execFunc(temper[tid+base], temper[tid+base], temper[tid1+base]);
    }
    else
    {
        temper[tid] = execFunc(temper[tid+base], temper[tid2+base], temper[tid1+base]);
    }
    __syncthreads();

    IC[gid] = temper[tid];
}

//Full refers to whether or not there is a node run on the CPU.
__global__
void
wholeDiamond(const REAL *inRight, const REAL *inLeft, REAL *outRight, REAL *outLeft, const bool split)
{
    extern __shared__ REAL temper[];

	//Same as upTriangle
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tididx = threadIdx.x + 1;

	int tid_top[3], tid_bottom[3];

    int k = gpuC.ht;
	tid_bottom[1] = tididx;
    tid_top[1] = tididx + gpuC.base;

	// Initialize temper.
    if (split)
    {
        gid += blockDim.x;
        tid_bottom[0] = tididx - 1;
        tid_bottom[2] = tididx + 1;
    }
    else
    {
        tid_bottom[0] = (gid == 0) ? (tididx + 1 : tididx - 1);
        tid_bottom[2] = (gid == gpuC.lastidx) ? (tididx - 1 : tididx + 1);
    }

    readIn(temper, inRight, inLeft, threadIdx.x, gid);
 
    tid_top[0] = tid_bottom[0] + gpuC.base;
    tid_top[2] = tid_bottom[2] + gpuC.base;

    __syncthreads();

	while (k<blockdim.x)
	{
        if (tididx > (blockDim.x-k) && tididx <= k)
        {
			temper[tid_top[1]] = execFunc(temper, tid_bottom);
		}
        k++
        __syncthreads();

        if (tididx > (blockDim.x-k) && tididx <= k)
		{
			temper[tididx] = execFunc(temper, tid_top);
		}
        k++
        __syncthreads();      
	}

    //-------------------TOP PART------------------------------------------
    k -= 2 

	while (k<0)
	{
        if (tididx > (blockDim.x-k) && tididx <= k)
        {
			temper[tid_top[1]] = execFunc(temper, tid_bottom);
		}
        k--
        __syncthreads();

        if (tididx > (blockDim.x-k) && tididx <= k)
		{
			temper[tididx] = execFunc(temper, tid_top);
		}
        k--
        __syncthreads();      
	}

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
//Needs to be changed to not justify temper between triangles.
__global__
void
splitDiamond(const REAL *inRight, const REAL *inLeft, REAL *outRight, REAL *outLeft)
{
    extern __shared__ REAL temper[];

	//Same as upTriangle
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
    int lastidx = ((blockDim.x*gridDim.x)-1);
	int base = blockDim.x + 2;
	int height = base>>1;
    int ht1 = height-1;
	int shft_rd;
	int shft_wr;
	int leftidx = height - (tid>>1) + (((tid>>1) & 1) * base) + (tid & 1) - 2;
	int rightidx = height + (tid>>1) + (((tid>>1) & 1) * base) + (tid & 1);
    int tid1 = tid + 1;
    int tid2 = ((gid == ht1) ? tid : tid+2);
    int tid0 = ((gid == height) ? tid+2 : tid);
    int gidout = (gid - blockDim.x) & lastidx;

	// Initialize temper.

    temper[leftidx] = inRight[gid];
	temper[rightidx] = inLeft[gid];

    //Wind it up!

    __syncthreads();

    for (int k = ht1; k>0; k--)
    {
        // This tells you if the current row is the first or second.
        shft_wr = base * ((k+1) & 1);
        // Read and write are opposite rows.
        shft_rd = base * (k & 1);

        if (tid1 < (base-k) && tid1 >= k)
        {
            temper[tid1 + shft_wr] = execFunc(temper[tid0+shft_rd], temper[tid2+shft_rd], temper[tid1+shft_rd]);
        }

        __syncthreads();
    }

    REAL trade = temper[tid1];
    __syncthreads();
    temper[tid] = trade;
    __syncthreads();

    //-------------------TOP PART------------------------------------------
    leftidx = (tid>>1) + (((tid>>1) & 1) * blockDim.x) + (tid & 1);
    rightidx = (blockDim.x - 2) + (((tid>>1) & 1) * blockDim.x) + (tid & 1) -  (tid>>1);

    tid0--;
    tid2--;

	for (int k = 1; k<ht1; k++)
	{
		//Bitwise even odd. On even iterations write to first row.
		shft_wr = blockDim.x * (k & 1);
		//On even iterations write to second row (starts at element 32)
		shft_rd = blockDim.x * ((k + 1) & 1);


        if (tid < (blockDim.x-k) && tid >= k)
        {
            temper[tid + shft_wr] = execFunc(temper[tid0+shft_rd], temper[tid2+shft_rd], temper[tid+shft_rd]);
        }

		//Make sure the threads are synced
		__syncthreads();
    }

	outRight[gid] = temper[rightidx];
	outLeft[gidout] = temper[leftidx];
}

__host__
void
CPU_diamond(REAL *temper, int tpb)
{
    int bck, fwd, shft_rd, shft_wr;
    int base = tpb + 2;
    int ht = tpb/2;

    //Splitting it is the whole point!
    for (int k = ht; k>0; k--)
    {
        // This tells you if the current row is the first or second.
        shft_wr = base * ((k+1) & 1);
        // Read and write are opposite rows.
        shft_rd = base * (k & 1);

        for(int n = k; n<(base-k); n++)
        {
            bck = n - 1;
            fwd = n + 1;
            //Double trailing index.
            if(n == ht)
            {
                temper[n + shft_wr] = execFuncHost(temper[bck+shft_rd], temper[bck+shft_rd], temper[n+shft_rd]);
            }
            //Double leading index.
            else if(n == ht+1)
            {
                temper[n + shft_wr] = execFuncHost(temper[fwd+shft_rd], temper[fwd+shft_rd], temper[n+shft_rd]);
            }
            else
            {
                temper[n + shft_wr] = execFuncHost(temper[bck+shft_rd], temper[fwd+shft_rd], temper[n+shft_rd]);
            }
        }
    }

    for (int k = 0; k<tpb; k++) temper[k] = temper[k+1];
    //Top part.
    ht--;
    for (int k = 1; k<ht; k++)
    {
        // This tells you if the current row is the first or second.
        shft_wr = tpb * (k & 1);
        // Read and write are opposite rows.
        shft_rd = tpb * ((k+1) & 1);

        for(int n = k; n<(tpb-k); n++)
        {
            bck = n - 1;
            fwd = n + 1;
            //Double trailing index.
            if(n == ht)
            {
                temper[n + shft_wr] = execFuncHost(temper[bck+shft_rd], temper[bck+shft_rd], temper[n+shft_rd]);
            }
            //Double leading index.
            else if(n == ht+1)
            {
                temper[n + shft_wr] = execFuncHost(temper[fwd+shft_rd], temper[fwd+shft_rd], temper[n+shft_rd]);
            }
            else
            {
                temper[n + shft_wr] = execFuncHost(temper[bck+shft_rd], temper[fwd+shft_rd], temper[n+shft_rd]);
            }
        }
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
    const int base = (tpb + 2);
    const int ht = base/2;
    const size_t smem = (base*2)*sizeof(REAL);
    const int cpuLoc = dv-tpb;

    int indices[4][tpb];
    for (int k = 0; k<tpb; k++)
    {
        indices[0][k] = ht - k/2 + ((k/2 & 1) * base) + (k & 1) - 2; //left
        indices[1][k] = ht + k/2 + ((k/2 & 1) * base) + (k & 1); //right

        indices[2][k] = k/2 + ((k/2 & 1) * tpb) + (k & 1); //left
        indices[3][k] = (tpb - 2) + ((k/2 & 1) * tpb) + (k & 1) -  k/2; //right
    }

	REAL *d_IC, *d0_right, *d0_left, *d2_right, *d2_left;

	cudaMalloc((void **)&d_IC, sizeof(REAL)*dv);
	cudaMalloc((void **)&d0_right, sizeof(REAL)*dv);
	cudaMalloc((void **)&d0_left, sizeof(REAL)*dv);
	cudaMalloc((void **)&d2_right, sizeof(REAL)*dv);
	cudaMalloc((void **)&d2_left, sizeof(REAL)*dv);

	// Copy the initial conditions to the device array.
	cudaMemcpy(d_IC,IC,sizeof(REAL)*dv,cudaMemcpyHostToDevice);
	// Start the counter and start the clock.
	const double t_fullstep = dt*(double)tpb;

	upTriangle <<< bks,tpb,smem >>>(d_IC,d0_right,d0_left);

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

        wholeDiamond <<< bks-1, tpb, smem, st1 >>>(d0_right, d0_left, d2_right, d2_left, false);

        cudaMemcpyAsync(h_left, d0_left, tpb*sizeof(REAL), cudaMemcpyDeviceToHost, st2);
        cudaMemcpyAsync(h_right, d0_right, tpb*sizeof(REAL), cudaMemcpyDeviceToHost, st3);

        cudaStreamSynchronize(st2);
        cudaStreamSynchronize(st3);

        for (int k = 0; k<tpb; k++) 
        {		
            tmpr[indices[0][k]] = h_right[k];		
            tmpr[indices[1][k]] = h_left[k];			
        }

        CPU_diamond(tmpr, tpb);

        for (int k = 0; k<tpb; k++) 
        {		
            h_left[k] = tmpr[indices[2][k]];		
            h_right[k] = tmpr[indices[3][k]];		
        }
        
        cudaMemcpyAsync(d2_right, h_right, tpb*sizeof(REAL), cudaMemcpyHostToDevice,st2);
        cudaMemcpyAsync(d2_left+cpuLoc, h_left, tpb*sizeof(REAL), cudaMemcpyHostToDevice,st3);

        //Split Diamond End------

    	while(t_eq < t_end)
    	{

            wholeDiamond <<< bks,tpb,smem >>>(d2_right,d2_left,d0_right,d0_left,true);

            //Split Diamond Begin------

            wholeDiamond <<< bks-1, tpb, smem, st1 >>>(d0_right, d0_left, d2_right, d2_left, false);

            cudaMemcpyAsync(h_left, d0_left, tpb*sizeof(REAL), cudaMemcpyDeviceToHost, st2);
            cudaMemcpyAsync(h_right, d0_right, tpb*sizeof(REAL), cudaMemcpyDeviceToHost, st3);

            cudaStreamSynchronize(st2);
            cudaStreamSynchronize(st3);

            for (int k = 0; k<tpb; k++) 
            {		
                tmpr[indices[0][k]] = h_right[k];		
                tmpr[indices[1][k]] = h_left[k];		
            }

            CPU_diamond(tmpr, tpb);

            for (int k = 0; k<tpb; k++) 
            {		
                h_left[k] = tmpr[indices[2][k]];		
                h_right[k] = tmpr[indices[3][k]];		
            }

            cudaMemcpyAsync(d2_right, h_right, tpb*sizeof(REAL), cudaMemcpyHostToDevice,st2);
            cudaMemcpyAsync(d2_left+cpuLoc, h_left, tpb*sizeof(REAL), cudaMemcpyHostToDevice,st3);

            //Split Diamond End------

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
        cudaFreeHost(h_right);
        cudaFreeHost(h_left);
        free(tmpr);
	}
    else
    {
        splitDiamond <<< bks,tpb,smem >>>(d0_right,d0_left,d2_right,d2_left);
        t_eq = t_fullstep;

        while(t_eq < t_end)
        {
            wholeDiamond <<< bks,tpb,smem >>>(d2_right,d2_left,d0_right,d0_left,true);

            splitDiamond <<< bks,tpb,smem >>>(d0_right,d0_left,d2_right,d2_left);

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

	downTriangle <<< bks,tpb,smem >>>(d_IC,d2_right,d2_left);

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
    //That is there are less than 8 arguments.

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
	const double tf = atof(argv[4]) - 0.5*dt; //Finish time
    const double freq = atof(argv[5]);
    const int scheme = atoi(argv[6]); //1 for Swept 0 for classic
    const int share = atoi(argv[7]);
	const int bks = dv/tpb; //The number of blocks
    const double lx = ds * ((double)dv - 1.0);
    fou = th_diff*dt/(ds*ds);  //Fourier number
    char const *prec;
    prec = (sizeof(REAL)<6) ? "Single": "Double";

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

	for (int k = 0; k<dv; k++)
	{
		fwr << IC[k] << " ";
	}
	fwr << endl;

    //Transfer data to GPU.
	// This puts the Fourier number in constant memory.
	cudaMemcpyToSymbol(fo,&fou,sizeof(REAL));

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
