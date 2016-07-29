//Based on
//https://en.wikipedia.org/wiki/Sod_shock_tube
//http://wonka.physics.ncsu.edu/pub/VH-1/bproblems.php
//http://www.astro.sunysb.edu/mzingale/codes.html
// http://cococubed.asu.edu/code_pages/exact_riemann.shtml


//COMPILE LINE:
// nvcc -o ./bin/EulerOut Euler1D_SweptShared.cu -gencode arch=compute_35,code=sm_35 -lm -w -std=c++11

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <vector_types.h>
#include <helper_math.h>
#include <math_functions.h>

#include <ostream>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <omp.h>

//#include "SwR_1DShared.h"


#define REAL        float
#define REALfour    float4
#define REALthree   float3

const REAL gam = 1.4f;
const REAL m_gamma = .4;
const REAL dx = .5;

__constant__ REALfour dbd[2];
__constant__ REALthree dimens;

__device__
__forceinline__
void
pressure(REALfour current)
{
    current.w = dimens.z * (current.z - (0.5 * current.y * current.y/current.x));
}

//This will need to return the ratio to the execFunc

__device__
__forceinline__
REAL
pressureRatio(REAL cvLeft, REAL cvCenter, REAL cvRight)
{
    return (cvRight- cvCenter)/(cvCenter- cvLeft);
}


__device__
REALfour
limitor(REALthree cvCurrent, REALthree cvOther, REAL pRatio)
{
    if (isfinite(pRatio) && pRatio > 0) //If it's finite and positive
    {
        REAL fact = (pRatio < 1) ? pRatio : 1.f;
        return make_float4(cvCurrent + 0.5* fact * (cvOther - cvCurrent));

    }
    else //If it's nan, inf, negative or zero.
    {
        return make_float4(cvCurrent);
    }

}

//Left and Center then Left and right.
__device__
void
eulerFlux(REALfour cvLeft, REALfour cvRight, REALthree flux)
{
    REAL uLeft = cvLeft.y/cvLeft.x;
    REAL uRight = cvRight.y/cvRight.x;
    REAL eLeft = cvLeft.z/cvLeft.x;
    REAL eRight = cvRight.z/cvRight.x;

    flux.x = 0.5 * (cvLeft.x*uLeft + cvRight.x*uRight);
    flux.y = 0.5 * (cvLeft.x*uLeft*uLeft + cvRight.x*uRight*uRight + cvLeft.w + cvRight.w);
    flux.z = 0.5 * (cvLeft.x*uLeft*eLeft + cvRight.x*uRight*eRight + uLeft*cvLeft.w + uRight*cvRight.w);

    printf("FluxL: %.8f %.8f %.8f \n",flux.x,flux.y, flux.z);

    REALfour halfState;
    REAL rhoLeftsqrt = sqrtf(cvLeft.x); REAL rhoRightsqrt = sqrtf(cvRight.x);
    halfState.x = rhoLeftsqrt * rhoRightsqrt;
    halfState.y = (rhoLeftsqrt*uLeft + rhoRightsqrt*uRight)/(rhoLeftsqrt+rhoRightsqrt);
    halfState.z = (rhoLeftsqrt*eLeft + rhoRightsqrt*eRight)/(rhoLeftsqrt+rhoRightsqrt);
    pressure(halfState);

    REAL spectreRadius = sqrtf(dimens.y * halfState.w/halfState.x) + fabs(halfState.y);

    flux += 0.5 * spectreRadius * (make_float3(cvLeft) - make_float3(cvRight));

}


__device__
REALfour
eulerStutterStep(REAL pfarLeft, REALfour stateLeft, REALfour stateCenter, REALfour stateRight, REAL pfarRight)
{
    REALthree fluxL, fluxR, pR;
    REALfour tempStateLeft, tempStateRight;

    pR = make_float3(pressureRatio(pfarLeft,stateLeft.w,stateCenter.w),
        pressureRatio(stateLeft.w,stateCenter.w,stateRight.w),
        pressureRatio(stateCenter.w,stateRight.w,pfarRight));

    tempStateLeft = limitor(make_float3(stateLeft), make_float3(stateCenter), pR.x);
    tempStateRight = limitor(make_float3(stateCenter), make_float3(stateLeft), 1.0/pR.y);
    pressure(tempStateLeft);
    pressure(tempStateRight);
    eulerFlux(tempStateLeft,tempStateRight,fluxL);

    tempStateLeft = limitor(make_float3(stateCenter), make_float3(stateRight), pR.y);
    tempStateRight = limitor(make_float3(stateRight), make_float3(stateCenter), 1.0/pR.z);
    pressure(tempStateLeft);
    pressure(tempStateRight);
    eulerFlux(tempStateLeft,tempStateRight,fluxR);

    stateCenter += make_float4(0.5 * dimens.x * (fluxL-fluxR));
    return pressure(stateCenter);

}

eulerFinalStep(REAL pfarLeft, REALfour stateLeft, REALfour stateCenter, REAL stateCenter_orig REALfour stateRight, REAL pfarRight)
{
    REALthree fluxL, fluxR, pR;
    REALfour tempStateLeft, tempStateRight;

    pR = make_float3(pressureRatio(pfarLeft,stateLeft.w,stateCenter.w),
        pressureRatio(stateLeft.w,stateCenter.w,stateRight.w),
        pressureRatio(stateCenter.w,stateRight.w,pfarRight));

    tempStateLeft = limitor(make_float3(stateLeft), make_float3(stateCenter), pR.x);
    tempStateRight = limitor(make_float3(stateCenter), make_float3(stateLeft), 1.0/pR.y);
    pressure(tempStateLeft);
    pressure(tempStateRight);
    eulerFlux(tempStateLeft,tempStateRight,fluxL);

    tempStateLeft = limitor(make_float3(stateCenter), make_float3(stateRight), pR.y);
    tempStateRight = limitor(make_float3(stateRight), make_float3(stateCenter), 1.0/pR.z);
    pressure(tempStateLeft);
    pressure(tempStateRight);
    eulerFlux(tempStateLeft,tempStateRight,fluxR);

    stateCenter_orig += make_float4(dimens.x * (fluxL-fluxR));
    return pressure(stateCenter_orig);

}


__global__
void
classicDisc(REALfour *IC, REALfour *temp)
{

    int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
    int lastidx = ((blockDim.x*gridDim.x)-1);
    int gidp = gid + 1;
    int gidm = gid - 1;

    if (gid == 0)
    {
        temp[gid] = execFunc(IC[gidp], IC[gidp], IC[gid]);
        printf("IM HERE!\n");
    }
    else if (gid == lastidx)
    {

        temp[gid] = execFunc(IC[gidm], IC[gidm], IC[gid]);
    }
    else
    {
        temp[gid] = execFunc(IC[gidm], IC[gidp], IC[gid]);
    }

    IC[gid] = temp[gid];
}

__global__
void
upTriangle(REALfour *IC, REALfour *right, REALfour *left)
{

	extern __shared__ REALfour temper[];

	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	int tid = threadIdx.x; //Block Thread ID

    int tid_top[5], tid_bottom[5];
	#pragma unroll
	for (int k = -2; k<3; k++)
	{
		tid_top[k+2] = tid + k + blockDim.x;
		tid_bottom[k+2] = tid + k;
	}

	int leftidx = ((tid/4 & 1) * blockDim.x) + (tid/4)*2 + (tid & 3);
	int rightidx = (blockDim.x - 4) + ((tid/4 & 1) * blockDim.x) + (tid & 3) - (tid/4)*2;

	int step2;

    //Assign the initial values to the first row in temper, each block
    //has it's own version of temper shared among its threads.
	temper[tid] = IC[gid];

	if (tid > 1 && tid <(blockDim.x-2))
	{
		temper[tid_top[2]] = eulerStutterStep(temper[tid_bottom[0]].w, temper[tid_bottom[1]], temper[tid_bottom[2]],
			temper[tid_bottom[3]], temper[tid_bottom[4]].w);
	}

	__syncthreads();

	//The initial conditions are timslice 0 so start k at 1.
	for (int k = 4; k<(blockDim.x/2); k+=4)
	{
		if (tid < (blockDim.x-k) && tid >= k)
		{
			temper[tid] = eulerFinalStep(temper[tid_top[0]].w, temper[tid_top[1]], temper[tid_top[2]],
				temper[tid], temper[tid_top[3]], temper[tid_top[4]].w);

		}

		step2 = k + 2;
		__syncthreads();

		if (tid < (blockDim.x-step2) && tid >= step2)
		{
			temper[tid_top[2]] = eulerStutterStep(temper[tid_bottom[0]].w, temper[tid_bottom[1]], temper[tid_bottom[2]],
				temper[tid_bottom[3]], temper[tid_bottom[4]].w);
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
downTriangle(REALfour *IC, REALfour *right, REALfour *left)
{
	extern __shared__ REALfour temper[];

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
	int gidin = (gid + blockDim.x) & ((blockDim.x*gridDim.x)-1);

	temper[leftidx] = right[gid];
	temper[rightidx] = left[gidin];

	for (int k = (height-2); k>0; k-=4)
	{
		if (tididx < (base-k) && tididx >= k)
		{
			temper[tid_top[2]] = stutterStep(temper[tid_bottom[0]].x, temper[tid_bottom[1]], temper[tid_bottom[2]],
				temper[tid_bottom[3]], temper[tid_bottom[4]].x);

		}

		step2 = k-2;

		if (tididx < (base-step2) && tididx >= step2)
		{
			temper[tididx] = finalStep(temper[tid_top[0]].x, temper[tid_top[1]], temper[tid_top[2]],
				temper[tididx], temper[tid_top[3]], temper[tid_top[4]].x);
		}

		//Make sure the threads are synced
		__syncthreads();
	}

    IC[gid] = temper[tididx];
}


//Full refers to whether or not there is a node run on the CPU.
__global__
void
wholeDiamond(REALfour *right, REALfour *left, bool full)
{

    extern __shared__ REALfour temper[];

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
	// Initialize temper. Kind of an unrolled for loop.  This is actually at
	// Timestep 0.

    if (full)
    {
        int gidin = (gid + blockDim.x) & lastidx;
        temper[leftidx] = right[gid];
        temper[rightidx] = left[gidin];
    }
    else
    {
        int gidin = (gid - blockDim.x) & lastidx;
        temper[leftidx] = right[gidin];
        temper[rightidx] = left[gid];
    }

    for (int k = (height-2); k>0; k-=4)
	{
		if (tididx < (base-k) && tididx >= k)
		{
			temper[tid_top[2]] = stutterStep(temper[tid_bottom[0]].x, temper[tid_bottom[1]], temper[tid_bottom[2]],
				temper[tid_bottom[3]], temper[tid_bottom[4]].x);

		}

		step2 = k-2;

		if (tididx < (base-step2) && tididx >= step2)
		{
			temper[tididx] = finalStep(temper[tid_top[0]].x, temper[tid_top[1]], temper[tid_top[2]],
				temper[tididx], temper[tid_top[3]], temper[tid_top[4]].x);
		}

		//Make sure the threads are synced
		__syncthreads();
	}

    //Boundary Conditions! This justifies it.
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

    // -------------------TOP PART------------------------------------------



    int leftidx = ((tid/4 & 1) * blockDim.x) + (tid/4)*2 + (tid & 3);
	int rightidx = (blockDim.x - 4) + ((tid/4 & 1) * blockDim.x) + (tid & 3) - (tid/4)*2;

    #pragma unroll
    for (int k = -2; k<3; k++)
    {
        tid_top[k+2] = tid + k + blockDim.x;
        tid_bottom[k+2] = tid + k;
    }


	//The initial conditions are timeslice 0 so start k at 1.

    for (int k = 1; k<(height-1); k++)
	{
		//Bitwise even odd. On even iterations write to first row.
		shft_wr = base * (k & 1);
		//On even iterations write to second row (starts at element 32)
		shft_rd = base * ((k + 1) & 1);

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
splitDiamond(REALfour *right, REALfour *left)
{

    extern __shared__ REALfour temper[];

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
	int leftidx = base/2 - tid/2 + ((tid/2 & 1) * base) + (tid & 1) - 2;
	int rightidx = base/2 + tid/2 + ((tid/2 & 1) * base) + (tid & 1);
    int gidin = (gid - blockDim.x) & lastidx;
	// Initialize temper. Kind of an unrolled for loop.  This is actually at
	// Timestep 0.

    temper[leftidx] = right[gidin];
	temper[rightidx] = left[gid];

    //Wind it up!
    //k needs to insert the relevant left right values around the computed values
    //every timestep.  Since it grows larger the loop is reversed.

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
                    temper[tid1 + shft_wr] =execFunc(temper[tid+shft_rd], temper[tid+shft_rd], temper[tid1+shft_rd]);
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

    temper[tid] = temper[tid1];

    //-------------------TOP PART------------------------------------------
    leftidx = tid/2 + ((tid/2 & 1) * blockDim.x) + (tid & 1);
    rightidx = (blockDim.x - 2) + ((tid/2 & 1) * blockDim.x) + (tid & 1) -  tid/2;

    int tidm = tid - 1;

    //The initial conditions are timslice 0 so start k at 1.

	for (int k = 1; k<(height-1); k++)
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
                if (tid == (height - 2))
                {
                    temper[tid + shft_wr] = execFunc(temper[tidm + shft_rd], temper[tidm + shft_rd], temper[tid + shft_rd]);
                }
                else if (tid == (height - 1))
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

    //After the triangle has been computed, the right and left shared arrays are
	//stored in global memory by the global thread ID since (conveniently),
	//they're the same size as a warp!
	right[gid] = temper[rightidx];
	left[gid] = temper[leftidx];
}

//Do the split diamond on the CPU?
// What's the idea?  Say malloc the pointers in the wrapper.
// Calculate left and right idxs in wrapper too, why continually recalculate.
//

// __host__
// void
// CPU_diamond(REALfour *temper, int tpb)
// {
//     int bck, fwd, shft_rd, shft_wr;
//     int base = tpb + 2;
//     int ht = tpb/2;
//
//     //Splitting it is the whole point!
//     for (int k = ht; k>0; k--)
//     {
//         // This tells you if the current row is the first or second.
//         shft_wr = base * ((k+1) & 1);
//         // Read and write are opposite rows.
//         shft_rd = base * (k & 1);
//
//         for(int n = k; n<(base-k); n++)
//         {
//             bck = n - 1;
//             fwd = n + 1;
//             //Double trailing index.
//             if(n == ht)
//             {
//                 temper[n + shft_wr] = execFunc(temper[bck+shft_rd], temper[bck+shft_rd], temper[n+shft_rd]);
//             }
//             //Double leading index.
//             else if(n == ht+1)
//             {
//                 temper[n + shft_wr] = execFunc(temper[fwd+shft_rd], temper[fwd+shft_rd], temper[n+shft_rd]);
//             }
//             else
//             {
//                 temper[n + shft_wr] = execFunc(temper[bck+shft_rd], temper[fwd+shft_rd], temper[n+shft_rd]);
//             }
//         }
//     }
//
//     for (int k = 0; k<tpb; k++) temper[k] = temper[k+1];
//     //Top part.
//     for (int k = 1; k>ht; k++)
//     {
//         // This tells you if the current row is the first or second.
//         shft_wr = base * (k & 1);
//         // Read and write are opposite rows.
//         shft_rd = base * ((k+1) & 1);
//
//         for(int n = k; n<(tpb-k); n++)
//         {
//             bck = n - 1;
//             fwd = n + 1;
//             //Double trailing index.
//             if(n == ht)
//             {
//                 temper[n + shft_wr] = execFunc(temper[bck+shft_rd], temper[bck+shft_rd], temper[n+shft_rd]);
//             }
//             //Double leading index.
//             else if(n == ht+1)
//             {
//                 temper[n + shft_wr] = execFunc(temper[fwd+shft_rd], temper[fwd+shft_rd], temper[n+shft_rd]);
//             }
//             else
//             {
//                 temper[n + shft_wr] = execFunc(temper[bck+shft_rd], temper[fwd+shft_rd], temper[n+shft_rd]);
//             }
//         }
//     }
// }
//
// //The host routine.
double
sweptWrapper(const int bks, int tpb, const int dv, REAL dt, const int t_end,
    const int cpu, REALfour *IC, REALfour *T_f)
{

    const size_t smem1 = 2*tpb*sizeof(REALfour);
    const size_t smem2 = (2*tpb+8)*sizeof(REALfour);

    int indices[4][tpb];
    for (int k = 0; k<tpb; k++)
    {
        indices[0][k] = k/2 + ((k/2 & 1) * tpb) + (k & 1);
        indices[1][k] = (tpb - 2) + ((k/2 & 1) * tpb) + (k & 1) -  k/2;
        indices[2][k] = k/2 + ((k/2 & 1) * tpb) + (k & 1);
        indices[3][k] = (tpb - 1) + ((k/2 & 1) * tpb) + (k & 1) -  k/2;
    }

    REALfour *tmpr;
    tmpr = (REALfour*)malloc(smem2);
	REALfour *d_IC, *d_right, *d_left;
    REALfour right[tpb], left[tpb];

	cudaMalloc((void **)&d_IC, sizeof(REALfour)*dv);
	cudaMalloc((void **)&d_right, sizeof(REALfour)*dv);
	cudaMalloc((void **)&d_left, sizeof(REALfour)*dv);

	// Copy the initial conditions to the device array.
	cudaMemcpy(d_IC,IC,sizeof(REALfour)*dv,cudaMemcpyHostToDevice);
	// Start the counter and start the clock.
	const double t_fullstep = dt*(double)tpb;

	upTriangle <<< bks,tpb,smem1 >>>(d_IC,d_right,d_left);

    double t_eq;

	// Call the kernels until you reach the iteration limit.
    // Done now juse use streams or omp to optimize.

    // if (cpu)
    // {
    //     t_eq = t_fullstep/2;
    //     omp_set_num_threads( 2 );
    //
    // 	while(t_eq < t_end)
    // 	{
    //
    //         #pragma omp parallel sections
    //         {
    //         #pragma omp section
    //         {
    //             cudaMemcpy(right,d_left,tpb*sizeof(REAL),cudaMemcpyDeviceToHost);
    //             cudaMemcpy(left,d_right+dv-tpb,tpb*sizeof(REAL),cudaMemcpyDeviceToHost);
    //
    //             for (int k = 0; k<tpb; k++)
    //             {
    //                 tmpr[indices[0][k]] = right[k];
    //                 tmpr[indices[1][k]] = left[k];
    //             }
    //
    //             CPU_diamond(tmpr, tpb);
    //
    //             for (int k = 0; k<tpb; k++)
    //             {
    //                 right[k] = tmpr[indices[2][k]];
    //                 left[k] = tmpr[indices[3][k]];
    //             }
    //         }
    //         #pragma omp section
    //         {
    //             wholeDiamond <<< bks-1,tpb,smem2 >>>(d_right,d_left,false);
    //             cudaMemcpy(d_right, right, tpb*sizeof(REAL), cudaMemcpyHostToDevice);
    //             cudaMemcpy(d_left, left, tpb*sizeof(REAL), cudaMemcpyHostToDevice);
    //         }
    //         }
    //
    //         wholeDiamond <<< bks,tpb,smem2 >>>(d_right,d_left,true);
    //
	// 	    //So it always ends on a left pass since the down triangle is a right pass.
    //
	// 	    t_eq += t_fullstep;
    //
    // 		/* Since the procedure does not store the temperature values, the user
    // 		could input some time interval for which they want the temperature
    // 		values and this loop could copy the values over from the device and
    // 		write them out.  This way the user could see the progression of the
    // 		solution over time, identify an area to be investigated and re-run a
    // 		shorter version of the simulation starting with those intiial conditions.
    //
    // 		-------------------------------------
    // 	 	if (true)
    // 		{
    // 			downTriangle <<< bks,tpb,smem2 >>>(d_IC,d_right,d_left);
    // 			cudaMemcpy(T_final, d_IC, sizeof(REAL)*dv, cudaMemcpyDeviceToHost);
    // 			fwr << t_eq << " ";
    //
    // 			for (int k = 0; k<dv; k++)
    // 			{
    // 					fwr << T_final.x[k] << " ";
    // 			}
    // 				fwr << endl;
    //
    // 			upTriangle <<< bks,tpb,smem1 >>>(d_IC,d_right,d_left);
    // 			wholeDiamond <<< bks,tpb,smem2 >>>(d_right,d_left,-1);
    // 		}
    // 		-------------------------------------
    // 		*/
    //     }
	// }
    // else
    // {
        splitDiamond <<< bks,tpb,smem2 >>>(d_right,d_left);
        t_eq = t_fullstep;

        while(t_eq < t_end)
        {

            wholeDiamond <<< bks,tpb,smem2 >>>(d_right,d_left,true);

            splitDiamond <<< bks,tpb,smem2 >>>(d_right,d_left);
            //So it always ends on a left pass since the down triangle is a right pass.

            t_eq += t_fullstep;

            /*
            if (true)
            {
                downTriangle <<< bks,tpb,smem2 >>>(d_IC,d_right,d_left);
                cudaMemcpy(T_final, d_IC, sizeof(REAL)*dv, cudaMemcpyDeviceToHost);
                fwr << t_eq << " ";

                for (int k = 0; k<dv; k++)
                {
                        fwr << T_final.x[k] << " ";
                }
                    fwr << endl;

                upTriangle <<< bks,tpb,smem1 >>>(d_IC,d_right,d_left);
                wholeDiamond <<< bks,tpb,smem2 >>>(d_right,d_left,-1);
            }
            -------------------------------------
            */
        }
    //}

	downTriangle <<< bks,tpb,smem2 >>>(d_IC,d_right,d_left);

	cudaMemcpy(T_f, d_IC, sizeof(REALfour)*dv, cudaMemcpyDeviceToHost);

	cudaFree(d_IC);
	cudaFree(d_right);
	cudaFree(d_left);

    return t_eq;
}

int main( int argc, char *argv[] )
{
    using namespace std;
	if (argc != 6)
	{
		cout << "The Program takes five inputs: #Divisions, #Threads/block, dt, finish time, and GPU/CPU or all GPU" << endl;
		exit(-1);
	}
	// Choose the GPGPU.  This is device 0 in my machine which has 2 devices.
	cudaSetDevice(0);

    REALfour bd[2];
    bd[0].x = 1.0; //Density
    bd[1].x = 0.125;
    bd[0].y = 0.0; //Velocity
    bd[1].y = 0.0;
    bd[0].w = 1.0; //Pressure
    bd[1].w = 0.1;
    bd[0].z = bd[0].w/m_gamma; //Energy
    bd[1].z = bd[1].w/m_gamma;

    //Declare the dimensions in constant memory.

    const REAL dt = atof(argv[3]);
    const int dv = atoi(argv[1]); //Number of spatial points
	const int tpb = atoi(argv[2]); //Threads per Blocks
	const float tf = atof(argv[4]); //Finish time
	const int bks = dv/tpb; //The number of blocks
	const int tst = atoi(argv[5]);
    REAL lx = dx*((float)dv-1.f);

    REALthree dimz;
    dimz.x = dt/dx; // dt/dx
    dimz.y = gam; dimz.z = m_gamma;

	//Conditions for main input.  Unit testing kinda.
	//dv and tpb must be powers of two.  dv must be larger than tpb and divisible by
	//tpb.

	if ((dv & (tpb-1) !=0) || (tpb&31) != 0)
    {
        cout << "INVALID NUMERIC INPUT!! "<< endl;
        cout << "2nd ARGUMENT MUST BE A POWER OF TWO >= 32 AND FIRST ARGUMENT MUST BE DIVISIBLE BY SECOND" << endl;
        exit(-1);
    }

	// Initialize arrays.
    REALfour *IC = (REALfour*)malloc(dv*sizeof(float4));
	REALfour *T_final = (REALfour*)malloc(dv*sizeof(float4));

	// Some initial condition for the bar temperature, an exponential decay
	// function.
	for (int k = 0; k<dv; k++)
	{
        if (k<dv/2)
        {
            IC[k] = bd[0];
        }
        else
        {
            IC[k] = bd[1];
        }
	}

	// Call out the file before the loop and write out the initial condition.
	ofstream fwr, ftime;
	fwr.open("Results/Euler1D_Result.dat",ios::trunc);
	ftime.open("Results/Euler1D_Timing.txt",ios::app);
	// Write out x length and then delta x and then delta t.
	// First item of each line is timestamp.
	fwr << lx << " " << dv << " " << dx << " " << endl << 0 << " ";

	for (int k = 0; k<dv; k++)
	{
		fwr << IC[k].x << " ";

	}

	fwr << endl;

    //Transfer data to GPU.
	// This puts the Fourier number in constant memory.
	cudaMemcpyToSymbol(dimens,&dimz,sizeof(REALthree));
    cudaMemcpyToSymbol(dbd,&bd,2*sizeof(REALfour));

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

    //--------TEST-----------

     REALfour *d_IC, *d_temp;

    cudaMalloc((void **)&d_IC, sizeof(REALfour)*dv);
	cudaMalloc((void **)&d_temp, sizeof(REALfour)*dv);

	// Copy the initial conditions to the device array.
	cudaMemcpy(d_IC,IC,sizeof(REALfour)*dv,cudaMemcpyHostToDevice);

    tfm = 0.0;

    while (tfm < tf)
    {

        classicDisc <<< bks,tpb >>> (d_IC,d_temp);
        tfm += dt;

    }

    cudaMemcpy(T_final, d_IC, sizeof(REALfour)*dv, cudaMemcpyDeviceToHost);
    cudaFree(d_IC);
    cudaFree(d_temp);

    //--------TEST-----------

	//tfm = sweptWrapper(bks,tpb,dv,dt,tf,tst,IC,T_final);

	// Show the time and write out the final condition.
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime( &timed, start, stop);

	timed = timed * 1.e-3;

	cout << "That took: " << timed << " seconds" << endl;

	ftime << dv << " " << tpb << " " << timed << endl;

	ftime.close();

	fwr << tfm << " ";
	for (int k = 0; k<dv; k++)
	{
		fwr << T_final[k].x << " ";
	}

    fwr << endl;

    fwr << tfm << " ";
	for (int k = 0; k<dv; k++)
	{
		fwr << T_final[k].w << " ";
	}


	fwr.close();

	// Free the memory and reset the device.
	cudaDeviceSynchronize();

	cudaEventDestroy( start );
	cudaEventDestroy( stop );
    cudaDeviceReset();
    free(IC);
    free(T_final);

	return 0;

}
