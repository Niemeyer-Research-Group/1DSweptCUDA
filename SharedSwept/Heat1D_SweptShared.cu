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

#include <ostream>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <omp.h>

#ifndef REAL
#define REAL  float
#endif

using namespace std;

__constant__ REAL fo;

REAL fou;

const REAL th_diff = 8.418e-5;

const REAL ds = .001;

__host__ __device__ REAL initFun(int xnode, REAL ds, REAL lx)
{
    REAL a = ((REAL)xnode*ds);
    return 100.f*a*(1.f-a/lx);
}

__device__ REAL execFunc(REAL tLeft, REAL tRight, REAL tCenter)
{
    return fo*(tLeft+tRight) + (1.f-2.f*fo)*tCenter;
}

__host__ REAL execFuncHost(REAL tLeft, REAL tRight, REAL tCenter)
{
    return fou*(tLeft+tRight) + (1.f-2.f*fou)*tCenter;
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

__global__
void
classicHeat(REAL *heat_in, REAL *heat_out)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
    int lastidx = ((blockDim.x*gridDim.x)-1);
    if (gid == 0)
    {
        heat_out[gid] = execFunc(heat_in[gid+1],heat_in[gid+1],heat_in[gid]);
    }
    else if (gid == lastidx)
    {
        heat_out[gid] = execFunc(heat_in[gid-1],heat_in[gid-1],heat_in[gid]);
    }
    else
    {
        heat_out[gid] = execFunc(heat_in[gid-1],heat_in[gid+1],heat_in[gid]);
    }
}

__global__
void
upTriangle(const REAL *IC, REAL *right, REAL *left)
{

	extern __shared__ REAL temper[];

	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	int tid = threadIdx.x; //Block Thread ID
    int tidp = tid + 1;
	int tidm = tid - 1;
	int shft_wr; //Initialize the shift to the written row of temper.
	int shft_rd; //Initialize the shift to the read row (opposite of written)
	int leftidx = tid/2 + ((tid/2 & 1) * blockDim.x) + (tid & 1);
	int rightidx = (blockDim.x - 2) + ((tid/2 & 1) * blockDim.x) + (tid & 1) -  tid/2;

    //Assign the initial values to the first row in temper, each warp (in this
	//case each block) has it's own version of temper shared among its threads.
	temper[tid] = IC[gid];

    __syncthreads();

	//The initial conditions are timslice 0 so start k at 1.

	for (int k = 1; k<(blockDim.x/2); k++)
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
	right[gid] = temper[rightidx];
	left[gid] = temper[leftidx];

}

// Down triangle is only called at the end when data is passed left.  It's never split.
// It returns IC which is a full 1D result at a certain time.
__global__
void
downTriangle(REAL *IC, const REAL *right, const REAL *left)
{
	extern __shared__ REAL temper[];

	//Same as upTriangle
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	int tid1 = tid + 1;
	int tid2 = tid + 2;
	int base = blockDim.x + 2;
	int height = base/2;
	int shft_rd;
	int shft_wr;

	int leftidx = height - tid/2 + ((tid/2 & 1) * base) + (tid & 1) - 2;
	int rightidx = height + tid/2 + ((tid/2 & 1) * base) + (tid & 1);
    int lastidx = ((blockDim.x*gridDim.x)-1);

	// Initialize temper. Kind of an unrolled for loop.  This is actually at
	// Timestep 0.

	temper[leftidx] = right[gid];
	temper[rightidx] = left[gid];

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
wholeDiamond(REAL *right, REAL *left, bool full)
{
    extern __shared__ REAL temper[];

	//Same as upTriangle
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
    int lastidx = ((blockDim.x*gridDim.x)-1);
	int tid1 = tid + 1;
	int tid2 = tid + 2;
	int base = blockDim.x + 2;
	int height = base/2;
	int shft_rd;
	int shft_wr;
	int leftidx = height - tid/2 + ((tid/2 & 1) * base) + (tid & 1) - 2;
	int rightidx = height + tid/2 + ((tid/2 & 1) * base) + (tid & 1);


    //if (blockIdx.x > (gridDim.x-3)) printf("gid: %i, gidin: %i \n",gid,gidin);
	// Initialize temper.

    if (full)
    {
        temper[leftidx] = right[gid];
        temper[rightidx] = left[gid];
    }
    else
    {
        gid += blockDim.x;
        temper[leftidx] = right[gid];
        temper[rightidx] = left[gid];
    }

    __syncthreads();

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

    //Boundary Conditions!
    if (full)
    {
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
    }
    else
    {
        temper[tid] = execFunc(temper[tid+base], temper[tid2+base], temper[tid1+base]);
    }

    __syncthreads();

    // Then make sure each block of threads are synced.

    //-------------------TOP PART------------------------------------------

    leftidx = tid/2 + ((tid/2 & 1) * blockDim.x) + (tid & 1);
    rightidx = (blockDim.x - 2) + ((tid/2 & 1) * blockDim.x) + (tid & 1) -  tid/2;

    int tidm = tid - 1;

    height -= 1;
	//The initial conditions are timeslice 0 so start k at 1.

    for (int k = 1; k<height; k++)
	{
		//Bitwise even odd. On even iterations write to first row.
		shft_wr = blockDim.x * (k & 1);
		//On even iterations write to second row (starts at element 32)
		shft_rd = blockDim.x * ((k + 1) & 1);

		//Each iteration the triangle narrows.  When k = 1, 30 points are
		//computed, k = 2, 28 points.
		if (tid < (blockDim.x-k) && tid >= k)
		{
			temper[tid + shft_wr] = execFunc(temper[tidm+shft_rd], temper[tid1+shft_rd], temper[tid+shft_rd]);
		}

		//Make sure the threads are synced
		__syncthreads();

	}
    right[gid] = temper[rightidx];
	left[gid] = temper[leftidx];
}

//Split one is always first.  Passing left like the downTriangle.  downTriangle
//should be rewritten so it isn't split.  Only write on a non split pass.
__global__
void
splitDiamond(REAL *right, REAL *left)
{
    extern __shared__ REAL temper[];

	//Same as upTriangle
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	int tid1 = tid + 1;
	int tid2 = tid + 2;
	int base = blockDim.x + 2;
	int height = base/2;
	int shft_rd;
	int shft_wr;
	int leftidx = height - tid/2 + ((tid/2 & 1) * base) + (tid & 1) - 2;
	int rightidx = height + tid/2 + ((tid/2 & 1) * base) + (tid & 1);

	// Initialize temper.

    temper[leftidx] = right[gid];
	temper[rightidx] = left[gid];

    //Wind it up!

    __syncthreads();

    for (int k = (height-1); k>0; k--)
    {
        // This tells you if the current row is the first or second.
        shft_wr = base * ((k+1) & 1);
        // Read and write are opposite rows.
        shft_rd = base * (k & 1);

        //Block 0 is split so it needs a different algorithm.  This algorithm
        //is slightly different than top triangle as described in the note above.
        if (blockIdx.x > 0)
        {
            if (tid1 < (base-k) && tid1 >= k)
            {
                temper[tid1 + shft_wr] = execFunc(temper[tid+shft_rd], temper[tid2+shft_rd], temper[tid1+shft_rd]);
            }

        }

        else
        {
            if (tid1 < (base-k) && tid1 >= k)
            {
                if (tid1 == (height-1))
                {
                    temper[tid1 + shft_wr] = execFunc(temper[tid+shft_rd], temper[tid+shft_rd], temper[tid1+shft_rd]);
                }
                else if (tid1 == height)
                {
                    temper[tid1 + shft_wr] = execFunc(temper[tid2+shft_rd], temper[tid2+shft_rd], temper[tid1+shft_rd]);
                }
                else
                {
                    temper[tid1 + shft_wr] = execFunc(temper[tid+shft_rd], temper[tid2+shft_rd], temper[tid1+shft_rd]);
                }
            }

        }

        __syncthreads();
    }

    REAL trade = temper[tid1];
    __syncthreads();
    temper[tid] = trade;
    __syncthreads();

    //-------------------TOP PART------------------------------------------
    leftidx = tid/2 + ((tid/2 & 1) * blockDim.x) + (tid & 1);
    rightidx = (blockDim.x - 2) + ((tid/2 & 1) * blockDim.x) + (tid & 1) -  tid/2;

    int tidm = tid - 1;

    height--;

	for (int k = 1; k<height; k++)
	{
		//Bitwise even odd. On even iterations write to first row.
		shft_wr = blockDim.x * (k & 1);
		//On even iterations write to second row (starts at element 32)
		shft_rd = blockDim.x * ((k + 1) & 1);

		//Each iteration the triangle narrows.  When k = 1, 30 points are
		//computed, k = 2, 28 points.
        if (blockIdx.x > 0)
        {
            if (tid < (blockDim.x-k) && tid >= k)
    		{
    			temper[tid + shft_wr] = execFunc(temper[tidm + shft_rd], temper[tid1 + shft_rd], temper[tid + shft_rd]);
    		}
        }
        else
        {
            if (tid < (blockDim.x-k) && tid >= k)
            {
                if (tid == (height-1))
                {
                    temper[tid + shft_wr] = execFunc(temper[tidm + shft_rd], temper[tidm + shft_rd], temper[tid + shft_rd]);

                }
                else if (tid == height)
                {
                    temper[tid + shft_wr] = execFunc(temper[tid1 + shft_rd], temper[tid1 + shft_rd], temper[tid + shft_rd]);
                }
                else
                {
                    temper[tid + shft_wr] = execFunc(temper[tidm + shft_rd], temper[tid1 + shft_rd], temper[tid + shft_rd]);
                }
            }
        }

		//Make sure the threads are synced
		__syncthreads();
    }

	right[gid] = temper[rightidx];
	left[gid] = temper[leftidx];
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
classicWrapper(const int bks, int tpb, const int dv, const REAL dt, const float t_end,
    REAL *IC, REAL *T_f, const float freq, ofstream &fwr)
{
    REAL *dheat_in, *dheat_out;

    cudaMalloc((void **)&dheat_in, sizeof(REAL)*dv);
    cudaMalloc((void **)&dheat_out, sizeof(REAL)*dv);

    // Copy the initial conditions to the device array.
    cudaMemcpy(dheat_in,IC,sizeof(REAL)*dv,cudaMemcpyHostToDevice);

    double t_eq = 0.0;
    double twrite = freq;

    while (t_eq < t_end)
    {
        classicHeat <<< bks,tpb >>> (dheat_in, dheat_out);
        classicHeat <<< bks,tpb >>> (dheat_out, dheat_in);
        t_eq += 2*dt;

        if (t_eq > twrite)
        {
            cudaMemcpy(T_f, dheat_in, sizeof(REAL)*dv, cudaMemcpyDeviceToHost);
            fwr << " Temperature " << t_eq << " ";

            for (int k = 0; k<dv; k++)
            {
                fwr << T_f[k] << " ";
            }
            fwr << endl;

            twrite += freq;
        }
    }

    cudaMemcpy(T_f, dheat_in, sizeof(REAL)*dv, cudaMemcpyDeviceToHost);

    cudaFree(dheat_in);
    cudaFree(dheat_out);

    return t_eq;

}

//The Swept Rule wrapper.
double
sweptWrapper(const int bks, int tpb, const int dv, const REAL dt, const float t_end, const int cpu,
    REAL *IC, REAL *T_f, const float freq, ofstream &fwr)
{
    const int base = (tpb + 2);
    const int ht = base/2;
    const size_t smem1 = 2*tpb*sizeof(REAL);
    const size_t smem2 = (base*2)*sizeof(REAL);

    int indices[4][tpb];
    for (int k = 0; k<tpb; k++)
    {
        indices[0][k] = ht - k/2 + ((k/2 & 1) * base) + (k & 1) - 2; //left
        indices[1][k] = ht + k/2 + ((k/2 & 1) * base) + (k & 1); //right

        indices[2][k] = k/2 + ((k/2 & 1) * tpb) + (k & 1); //left
        indices[3][k] = (tpb - 2) + ((k/2 & 1) * tpb) + (k & 1) -  k/2; //right
    }


	REAL *d_IC, *d_right, *d_left, *d_bin;

	cudaMalloc((void **)&d_IC, sizeof(REAL)*dv);
	cudaMalloc((void **)&d_right, sizeof(REAL)*dv);
	cudaMalloc((void **)&d_left, sizeof(REAL)*dv);
    cudaMalloc((void **)&d_bin, sizeof(REAL)*dv);

	// Copy the initial conditions to the device array.
	cudaMemcpy(d_IC,IC,sizeof(REAL)*dv,cudaMemcpyHostToDevice);
	// Start the counter and start the clock.
	const double t_fullstep = dt*(double)tpb;

	upTriangle <<< bks,tpb,smem1 >>>(d_IC,d_right,d_left);

    swapKernel <<< bks,tpb >>> (d_right, d_bin, 1);
    swapKernel <<< bks,tpb >>> (d_bin, d_right, 0);

    double t_eq;
    double twrite = freq;


	// Call the kernels until you reach the iteration limit.

    if (cpu)
    {
        REAL *tmpr = (REAL*)malloc(smem2);
        REAL *h_right, *h_left;
        cudaHostAlloc((void **) &h_right, tpb*sizeof(REAL), cudaHostAllocDefault);
        cudaHostAlloc((void **) &h_left, tpb*sizeof(REAL), cudaHostAllocDefault);
        // h_right = (REALfour *) malloc(tpb*sizeof(REALfour));
        // h_left = (REALfour *) malloc(tpb*sizeof(REALfour));

        t_eq = t_fullstep;
        omp_set_num_threads( 2 );

        //Split Diamond Begin------

        cudaMemcpy(h_right, d_left, tpb*sizeof(REAL), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_left, d_right, tpb*sizeof(REAL), cudaMemcpyDeviceToHost);

        #pragma omp parallel sections
        {
        #pragma omp section
        {
            for (int k = 0; k<tpb; k++)
            {
                tmpr[indices[0][k]] = h_left[k];
                tmpr[indices[1][k]] = h_right[k];
            }

            CPU_diamond(tmpr, tpb);

            for (int k = 0; k<tpb; k++)
            {
                h_left[k] = tmpr[indices[2][k]];
                h_right[k] = tmpr[indices[3][k]];
            }
        }
        #pragma omp section
        {
            wholeDiamond <<< bks-1,tpb,smem2 >>>(d_right,d_left,false);
        }
        }

        cudaMemcpy(d_right, h_right, tpb*sizeof(REAL), cudaMemcpyHostToDevice);
        cudaMemcpy(d_left, h_left, tpb*sizeof(REAL), cudaMemcpyHostToDevice);

        swapKernel <<< bks,tpb >>> (d_left, d_bin, -1);
        swapKernel <<< bks,tpb >>> (d_bin, d_left, 0);

        //Split Diamond End------

    	while(t_eq < t_end)
    	{

            wholeDiamond <<< bks,tpb,smem2 >>>(d_right,d_left,true);

            swapKernel <<< bks,tpb >>> (d_right, d_bin, 1);
            swapKernel <<< bks,tpb >>> (d_bin, d_right, 0);

            //Split Diamond Begin------

            cudaMemcpy(h_right,d_left, tpb*sizeof(REAL), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_left, d_right, tpb*sizeof(REAL), cudaMemcpyDeviceToHost);
            #pragma omp parallel sections
            {
            #pragma omp section
            {


                for (int k = 0; k<tpb; k++)
                {
                    tmpr[indices[0][k]] = h_left[k];
                    tmpr[indices[1][k]] = h_right[k];
                }

                CPU_diamond(tmpr, tpb);

                for (int k = 0; k<tpb; k++)
                {
                    h_left[k] = tmpr[indices[2][k]];
                    h_right[k] = tmpr[indices[3][k]];
                }
            }
            #pragma omp section
            {
                wholeDiamond <<< bks-1,tpb,smem2 >>>(d_right,d_left,false);

            }
            }

            cudaMemcpy(d_right, h_right, tpb*sizeof(REAL), cudaMemcpyHostToDevice);
            cudaMemcpy(d_left, h_left, tpb*sizeof(REAL), cudaMemcpyHostToDevice);

            swapKernel <<< bks,tpb >>> (d_left, d_bin, -1);
            swapKernel <<< bks,tpb >>> (d_bin, d_left, 0);

            //Split Diamond End------

		    //So it always ends on a left pass since the down triangle is a right pass.

		    t_eq += t_fullstep;

    		/* Since the procedure does not store the temperature values, the user
    		could input some time interval for which they want the temperature
    		values and this loop could copy the values over from the device and
    		write them out.  This way the user could see the progression of the
    		solution over time, identify an area to be investigated and re-run a
    		shorter version of the simulation starting with those intiial conditions.
            */

            if (t_eq > twrite)
    		{
    			downTriangle <<< bks,tpb,smem2 >>>(d_IC,d_right,d_left);

    			cudaMemcpy(T_f, d_IC, sizeof(REAL)*dv, cudaMemcpyDeviceToHost);

    			fwr << "Temperature " << t_eq << " ";

    			for (int k = 0; k<dv; k++)
    			{
    				fwr << T_f[k] << " ";
    			}
    			fwr << endl;

                upTriangle <<< bks,tpb,smem1 >>>(d_IC,d_right,d_left);

                swapKernel <<< bks,tpb >>> (d_right, d_bin, 1);
                swapKernel <<< bks,tpb >>> (d_bin, d_right, 0);

    			splitDiamond <<< bks,tpb,smem2 >>>(d_right,d_left);

                swapKernel <<< bks,tpb >>> (d_left, d_bin, -1);
                swapKernel <<< bks,tpb >>> (d_bin, d_left, 0);

                t_eq += t_fullstep;

    			twrite += freq;
    		}
        }
        cudaFreeHost(h_right);
        cudaFreeHost(h_left);
        // free(h_right);
        // free(h_left);
        free(tmpr);
	}
    else
    {
        splitDiamond <<< bks,tpb,smem2 >>>(d_right,d_left);
        t_eq = t_fullstep;
        swapKernel <<< bks,tpb >>> (d_left, d_bin, -1);
        swapKernel <<< bks,tpb >>> (d_bin, d_left, 0);

        while(t_eq < t_end)
        {

            wholeDiamond <<< bks,tpb,smem2 >>>(d_right,d_left,true);

            swapKernel <<< bks,tpb >>> (d_right, d_bin, 1);
            swapKernel <<< bks,tpb >>> (d_bin, d_right, 0);

            splitDiamond <<< bks,tpb,smem2 >>>(d_right,d_left);

            swapKernel <<< bks,tpb >>> (d_left, d_bin, -1);
            swapKernel <<< bks,tpb >>> (d_bin, d_left, 0);

            //So it always ends on a left pass since the down triangle is a right pass.
            t_eq += t_fullstep;

            if (t_eq > twrite)
    		{
    			downTriangle <<< bks,tpb,smem2 >>>(d_IC,d_right,d_left);

    			cudaMemcpy(T_f, d_IC, sizeof(REAL)*dv, cudaMemcpyDeviceToHost);
    			fwr << "Temperature " << t_eq << " ";

    			for (int k = 0; k<dv; k++)
    			{
    				fwr << T_f[k] << " ";
    			}
    			fwr << endl;

    			upTriangle <<< bks,tpb,smem1 >>>(d_IC,d_right,d_left);

                swapKernel <<< bks,tpb >>> (d_right, d_bin, 1);
                swapKernel <<< bks,tpb >>> (d_bin, d_right, 0);

    			splitDiamond <<< bks,tpb,smem2 >>>(d_right,d_left);

                swapKernel <<< bks,tpb >>> (d_left, d_bin, -1);
                swapKernel <<< bks,tpb >>> (d_bin, d_left, 0);

                t_eq += t_fullstep;

    			twrite += freq;
    		}
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

int main( int argc, char *argv[] )
{
    //That is there are less than 8 arguments.
    if (argc < 9)
	{
		cout << "The Program takes 9 inputs, #Divisions, #Threads/block, deltat, finish time, output frequency..." << endl;
        cout << "Classic/Swept, CPU sharing Y/N, Variable Output File, Timing Output File (optional)" << endl;
		exit(-1);
	}

	// Choose the GPGPU.  This is device 0 in my machine which has 2 devices.
	cudaSetDevice(0);

    int dv = atoi(argv[1]); //Number of spatial points
	const int tpb = atoi(argv[2]); //Threads per Blocks
    const float dt =  atof(argv[3]);
	const float tf = atof(argv[4]); //Finish time
    const float freq = atof(argv[5]);
    const int scheme = atoi(argv[6]); //1 for Swept 0 for classic
    const int share = atoi(argv[7]);
	const int bks = dv/tpb; //The number of blocks
    const REAL lx = ds * ((REAL)dv - 1.f);
    fou = th_diff*dt/(ds*ds);  //Fourier number

    cout << bks << " Blocks " << lx << " Length" << " Type (float = 4, double = 8) " << sizeof(REAL) << endl;

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

	// Some initial condition for the bar temperature, an exponential decay
	// function.
	for (int k = 0; k<dv; k++)
	{
		IC[k] = initFun(k, ds, lx);
	}

	// Call out the file before the loop and write out the initial condition.
	ofstream fwr;
	fwr.open(argv[8],ios::trunc);
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

	// This initializes the device arrays on the device in global memory.
	// They're all the same size.  Conveniently.

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
        tfm = sweptWrapper(bks, tpb, dv, dt, tf, share, IC, T_final, freq, fwr);
    }
    else
    {
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

    if (argc>7)
    {
        ofstream ftime;
        ftime.open(argv[9],ios::app);
    	ftime << dv << "\t" << tpb << "\t" << per_ts << endl;
    	ftime.close();
    }
	fwr << "Temperature " << tfm << " ";
	for (int k = 0; k<dv; k++)
	{
		fwr << T_final[k] << " ";
	}

	fwr.close();

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

//END
