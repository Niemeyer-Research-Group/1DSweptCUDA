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

You should have received a copy of the MIT license along with this program.
If not, see <https://opensource.org/licenses/MIT>.
*/

//COMPILE LINE:
// nvcc -o ./bin/EulerOut Euler1D_SweptShared.cu -gencode arch=compute_35,code=sm_35 -lm -restrict -Xcompiler -fopenmp

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include "myVectorTypes.h"

#include <ostream>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <omp.h>

#ifndef REAL
    #define REAL        float
    #define REALfour    float4
    #define REALthree   float3
    #define THREEVEC( ... ) make_float3(__VA_ARGS__)
    #define FOURVEC( ... )  make_float4(__VA_ARGS__)
#else
    #define THREEVEC( ... ) make_double3(__VA_ARGS__)
    #define FOURVEC( ... )  make_double4(__VA_ARGS__)
#endif

const REAL gam = 1.4;
const REAL m_gamma = 0.4;
const REAL lx = 1.0;

REALfour bd[2];
REALthree dimz;
//dbd is the boundary condition
__constant__ REALfour dbd[2]; //0 is left 1 is right.
//dimens has three fields x is dt/dx, y is gamma, z is gamma-1
__constant__ REALthree dimens;

//Calculates the pressure at the current node with the rho, u, e state variables.
__device__ __host__
__forceinline__
REAL
pressure(REALfour current)
{
    #ifdef __CUDA_ARCH__
    return dimens.z * (current.z - (0.5 * current.y * current.y/current.x));
    #else
    return dimz.z * (current.z - (0.5 * current.y * current.y/current.x));
    #endif
}

//Calculates the pressure ratio between the right and left side pressure differences.
//(pRight-pCurrent)/(pCurrent-pLeft)
__device__ __host__
__forceinline__
REAL
pressureRatio(REAL cvLeft, REAL cvCenter, REAL cvRight)
{
    return (cvRight- cvCenter)/(cvCenter- cvLeft);
}

//Reconstructs the state variables if the pressure ratio is finite and positive.
//I think it's that internal boundary condition.
__device__ __host__
__forceinline__
REALfour
limitor(REALthree cvCurrent, REALthree cvOther, REAL pRatio)
{
    #ifdef __CUDA_ARCH__
    if (isfinite(pRatio) && pRatio > 0) //If it's finite and positive
    {
        REAL fact = ((pRatio < 1) ? pRatio : 1.0);
        return FOURVEC(cvCurrent + 0.5 * fact * (cvOther - cvCurrent));

    }
    #else
    if (std::isfinite(pRatio) && pRatio > 0) //If it's finite and positive
    {
        REAL fact = ((pRatio < 1) ? pRatio : 1.0);
        return FOURVEC(cvCurrent + 0.5 * fact * (cvOther - cvCurrent));

    }
    #endif

    return FOURVEC(cvCurrent);
}


//Left and Center then Left and right.
//This is the meat of the flux calculation.  Fields: x is rho, y is u, z is e, w is p.
__device__ __host__
REALthree
eulerFlux(REALfour cvLeft, REALfour cvRight)
{
    #ifndef __CUDA_ARCH__
    using namespace std;
    #endif
    //For the first calculation rho and p remain the same.
    REALthree flux;
    REAL spectreRadius;

    REAL uLeft = cvLeft.y/cvLeft.x;
    REAL uRight = cvRight.y/cvRight.x;
    REAL eLeft = cvLeft.z/cvLeft.x;
    REAL eRight = cvRight.z/cvRight.x;

    flux.x = 0.5 * (cvLeft.x*uLeft + cvRight.x*uRight);
    flux.y = 0.5 * (cvLeft.x*uLeft*uLeft + cvRight.x*uRight*uRight + cvLeft.w + cvRight.w);
    flux.z = 0.5 * (cvLeft.x*uLeft*eLeft + cvRight.x*uRight*eRight + uLeft*cvLeft.w + uRight*cvRight.w);

    REALfour halfState;
    REAL rhoLeftsqrt = sqrt(cvLeft.x); REAL rhoRightsqrt = sqrt(cvRight.x);
    halfState.x = rhoLeftsqrt * rhoRightsqrt;
    halfState.y = halfState.x * (rhoLeftsqrt*uLeft + rhoRightsqrt*uRight)/(rhoLeftsqrt + rhoRightsqrt);
    halfState.z = halfState.x * (rhoLeftsqrt*eLeft + rhoRightsqrt*eRight)/(rhoLeftsqrt + rhoRightsqrt);
    halfState.w = pressure(halfState);

    halfState.y = halfState.y/halfState.x;

    #ifdef __CUDA_ARCH__
    spectreRadius = sqrt(dimens.y * halfState.w/halfState.x) + fabs(halfState.y);
    #else
    spectreRadius = sqrt(dimz.y * halfState.w/halfState.x) + fabs(halfState.y);
    #endif

    flux += 0.5 * spectreRadius * (THREEVEC(cvLeft) - THREEVEC(cvRight));

    return flux;
}

//This is the predictor step of the finite volume scheme.
__device__ __host__
REALfour
eulerStutterClassicStep(REAL pfarLeft, REALfour stateLeft, REALfour stateCenter, REALfour stateRight, REAL pfarRight)
{
    REALthree fluxL, fluxR, pR;
    REALfour tempStateLeft, tempStateRight;

    //Get the pressure ratios as a structure.
    pR = THREEVEC(pressureRatio(pfarLeft,stateLeft.w,stateCenter.w),
        pressureRatio(stateLeft.w,stateCenter.w,stateRight.w),
        pressureRatio(stateCenter.w,stateRight.w,pfarRight));

    //This is the temporary state bounded by the limitor function.
    tempStateLeft = limitor(THREEVEC(stateLeft), THREEVEC(stateCenter), pR.x);
    tempStateRight = limitor(THREEVEC(stateCenter), THREEVEC(stateLeft), 1.0/pR.y);

    //Pressure needs to be recalculated for the new limited state variables.
    tempStateLeft.w = pressure(tempStateLeft);
    tempStateRight.w = pressure(tempStateRight);
    fluxL = eulerFlux(tempStateLeft,tempStateRight);

    //Do the same thing with the right side.
    tempStateLeft = limitor(THREEVEC(stateCenter), THREEVEC(stateRight), pR.y);
    tempStateRight = limitor(THREEVEC(stateRight), THREEVEC(stateCenter), 1.0/pR.z);
    tempStateLeft.w = pressure(tempStateLeft);
    tempStateRight.w = pressure(tempStateRight);
    fluxR = eulerFlux(tempStateLeft,tempStateRight);

    //Add the change back to the node in question.
    #ifdef __CUDA_ARCH__
    stateCenter += FOURVEC(0.5 * dimens.x * (fluxL-fluxR));
    #else
    stateCenter += FOURVEC(0.5 * dimz.x * (fluxL-fluxR));
    #endif
    stateCenter.w = pressure(stateCenter);

    return stateCenter;
}

__device__ __host__
REALfour
eulerFinalClassicStep(REAL pfarLeft, REALfour stateLeft, REALfour stateCenter, REALfour stateRight, REAL pfarRight)
{
    REALthree fluxL, fluxR, pR;
    REALfour tempStateLeft, tempStateRight;

    pR = THREEVEC(pressureRatio(pfarLeft,stateLeft.w,stateCenter.w),
        pressureRatio(stateLeft.w,stateCenter.w,stateRight.w),
        pressureRatio(stateCenter.w,stateRight.w,pfarRight));

    tempStateLeft = limitor(THREEVEC(stateLeft), THREEVEC(stateCenter), pR.x);
    tempStateRight = limitor(THREEVEC(stateCenter), THREEVEC(stateLeft), 1.0/pR.y);
    tempStateLeft.w = pressure(tempStateLeft);
    tempStateRight.w = pressure(tempStateRight);
    fluxL = eulerFlux(tempStateLeft,tempStateRight);

    tempStateLeft = limitor(THREEVEC(stateCenter), THREEVEC(stateRight), pR.y);
    tempStateRight = limitor(THREEVEC(stateRight), THREEVEC(stateCenter), 1.0/pR.z);
    tempStateLeft.w = pressure(tempStateLeft);
    tempStateRight.w = pressure(tempStateRight);
    fluxR = eulerFlux(tempStateLeft,tempStateRight);

    #ifdef __CUDA_ARCH__
    return FOURVEC(dimens.x * (fluxL-fluxR));
    #else
    return FOURVEC(dimz.x * (fluxL-fluxR));
    #endif

}

//This is the predictor step of the finite volume scheme.
__device__ __host__
REALfour
eulerStutterStep(REALfour *state, int tr[5])
{
    REALthree fluxL, fluxR, pR;
    REALfour tempStateLeft, tempStateRight;

    //Get the pressure ratios as a structure.
    pR = THREEVEC(pressureRatio(state[tr[0]].w, state[tr[1]].w, state[tr[2]].w),
        pressureRatio(state[tr[1]].w, state[tr[2]].w, state[tr[3]].w),
        pressureRatio(state[tr[2]].w, state[tr[3]].w, state[tr[4]].w));

    //This is the temporary state bounded by the limitor function.
    tempStateLeft = limitor(THREEVEC(state[tr[1]]), THREEVEC(state[tr[2]]), pR.x);
    tempStateRight = limitor(THREEVEC(state[tr[2]]), THREEVEC(state[tr[1]]), 1.0/pR.y);

    //Pressure needs to be recalculated for the new limited state variables.
    tempStateLeft.w = pressure(tempStateLeft);
    tempStateRight.w = pressure(tempStateRight);
    fluxL = eulerFlux(tempStateLeft,tempStateRight);

    //Do the same thing with the right side.
    tempStateLeft = limitor(THREEVEC(state[tr[2]]), THREEVEC(state[tr[3]]), pR.y);
    tempStateRight = limitor(THREEVEC(state[tr[3]]), THREEVEC(state[tr[2]]), 1.0/pR.z);
    tempStateLeft.w = pressure(tempStateLeft);
    tempStateRight.w = pressure(tempStateRight);
    fluxR = eulerFlux(tempStateLeft,tempStateRight);

    //Add the change back to the node in question.
    #ifdef __CUDA_ARCH__
    tempStateRight = state[tr[2]] + FOURVEC(0.5 * dimens.x * (fluxL-fluxR));
    #else
    tempStateRight = state[tr[2]] + FOURVEC(0.5 * dimz.x * (fluxL-fluxR));
    #endif

    tempStateRight.w = pressure(tempStateRight);

    return tempStateRight;

}

//Same thing as the predictor step, but this final step adds the result to the original state variables to advance to the next timestep.
//But the predictor variables to find the fluxes.
__device__ __host__
REALfour
eulerFinalStep(REALfour *state, int tr[5])
{
    REALthree fluxL, fluxR, pR;
    REALfour tempStateLeft, tempStateRight;

    //Get the pressure ratios as a structure.
    pR = THREEVEC(pressureRatio(state[tr[0]].w, state[tr[1]].w, state[tr[2]].w),
        pressureRatio(state[tr[1]].w, state[tr[2]].w, state[tr[3]].w),
        pressureRatio(state[tr[2]].w, state[tr[3]].w, state[tr[4]].w));

    //This is the temporary state bounded by the limitor function.
    tempStateLeft = limitor(THREEVEC(state[tr[1]]), THREEVEC(state[tr[2]]), pR.x);
    tempStateRight = limitor(THREEVEC(state[tr[2]]), THREEVEC(state[tr[1]]), 1.0/pR.y);

    //Pressure needs to be recalculated for the new limited state variables.
    tempStateLeft.w = pressure(tempStateLeft);
    tempStateRight.w = pressure(tempStateRight);
    fluxL = eulerFlux(tempStateLeft,tempStateRight);

    //Do the same thing with the right side.
    tempStateLeft = limitor(THREEVEC(state[tr[2]]), THREEVEC(state[tr[3]]), pR.y);
    tempStateRight = limitor(THREEVEC(state[tr[3]]), THREEVEC(state[tr[2]]), 1.0/pR.z);
    tempStateLeft.w = pressure(tempStateLeft);
    tempStateRight.w = pressure(tempStateRight);
    fluxR = eulerFlux(tempStateLeft,tempStateRight);

    #ifdef __CUDA_ARCH__
    return FOURVEC(dimens.x * (fluxL-fluxR));
    #else
    return FOURVEC(dimz.x * (fluxL-fluxR));
    #endif

}

__global__
void
swapKernel(const REALfour *passing_side, REALfour *bin, int direction)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
    int lastidx = ((blockDim.x*gridDim.x)-1);
    int gidout = (gid + direction*blockDim.x) & lastidx;

    bin[gidout] = passing_side[gid];

}

//Simple scheme with dirchlet boundary condition.
__global__
void
classicEuler(const REALfour *euler_in, REALfour *euler_out, bool final)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
    int lastidx = ((blockDim.x*gridDim.x)-1);

    if (final)
    {
        if (gid == 0)
        {
            euler_out[gid] = dbd[0];
        }
        else if (gid == lastidx)
        {
            euler_out[gid] = dbd[1];
        }
        else if (gid == 1)
        {
            euler_out[gid] += eulerFinalClassicStep(dbd[0].w,dbd[0],euler_in[gid],
                euler_in[(gid+1)],euler_in[(gid+2)].w);

            euler_out[gid].w = pressure(euler_out[gid]);
        }
        else if (gid == (lastidx-1))
        {
            euler_out[gid] += eulerFinalClassicStep(euler_in[(gid-2)].w,euler_in[(gid-1)],euler_in[gid],
                dbd[1],dbd[1].w);
            euler_out[gid].w = pressure(euler_out[gid]);
        }
        else
        {
            euler_out[gid] += eulerFinalClassicStep(euler_in[(gid-2)].w,euler_in[(gid-1)],euler_in[gid],
                euler_in[(gid+1)],euler_in[(gid+2)].w);
            euler_out[gid].w = pressure(euler_out[gid]);
        }
    }
    else
    {
        if (gid == 0)
        {
            euler_out[gid] = dbd[0];
        }
        else if (gid == lastidx)
        {
            euler_out[gid] = dbd[1];
        }
        else if (gid == 1)
        {
            euler_out[gid] = eulerStutterClassicStep(dbd[0].w,dbd[0],euler_in[gid],euler_in[(gid+1)],euler_in[(gid+2)].w);
        }
        else if (gid == (lastidx-1))
        {
            euler_out[gid] = eulerStutterClassicStep(euler_in[(gid-2)].w,euler_in[(gid-1)],euler_in[gid],dbd[1],dbd[1].w);
        }
        else
        {
            euler_out[gid] = eulerStutterClassicStep(euler_in[(gid-2)].w,euler_in[(gid-1)],euler_in[gid],euler_in[(gid+1)],euler_in[(gid+2)].w);
        }
    }
}

__global__
void
upTriangle(const REALfour *IC, REALfour *right, REALfour *left)
{

	extern __shared__ REALfour temper[];

	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	int tid = threadIdx.x; //Block Thread ID
    int step2;

    int tid_top[5], tid_bottom[5];
	#pragma unroll
	for (int k = -2; k<3; k++)
    {
        tid_bottom[k+2] = tid + k;
        tid_top[k+2] = tid + k + blockDim.x;
    }

	int leftidx = ((tid/4 & 1) * blockDim.x) + (tid/4)*2 + (tid & 3);
	int rightidx = (blockDim.x - 4) + ((tid/4 & 1) * blockDim.x) + (tid & 3) - (tid/4)*2;

    //Assign the initial values to the first row in temper, each block
    //has it's own version of temper shared among its threads.
	temper[tid] = IC[gid];

    __syncthreads();

	if (tid > 1 && tid <(blockDim.x-2))
	{
		temper[tid_top[2]] = eulerStutterStep(temper, tid_bottom);
	}

	__syncthreads();

	//The initial conditions are timslice 0 so start k at 1.
	for (int k = 4; k<(blockDim.x/2); k+=4)
	{
		if (tid < (blockDim.x-k) && tid >= k)
		{
            temper[tid] += eulerFinalStep(temper, tid_top);

            temper[tid].w = pressure(temper[tid]);
		}
        step2 = k+2;
		__syncthreads();

		if (tid < (blockDim.x-step2) && tid >= step2)
		{
            temper[tid_top[2]] = eulerStutterStep(temper, tid_bottom);
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
downTriangle(REALfour *IC, const REALfour *right, const REALfour *left)
{
	extern __shared__ REALfour temper[];

	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	int tididx = tid + 2;
	int base = blockDim.x + 4;
	int height = base/2;
    int lastidx = ((blockDim.x*gridDim.x)-1);
    int step2;

    int tid_top[5], tid_bottom[5];
    #pragma unroll
    for (int k = -2; k<3; k++)
    {
        tid_bottom[k+2] = tididx + k;
        tid_top[k+2] = tididx + k + base;
    }

	int leftidx = height + ((tid/4 & 1) * base) + (tid & 3) - (4 + (tid/4) * 2);
	int rightidx = height + ((tid/4 & 1) * base) + (tid/4)*2 + (tid & 3);

	temper[leftidx] = right[gid];
	temper[rightidx] = left[gid];

    __syncthreads();

    if (gid == 0)
    {
        temper[tididx] = dbd[0];
        temper[tididx+base] = dbd[0];
        IC[gid] = temper[tididx];
        return;
    }
    else if (gid == lastidx)
    {
        temper[tididx] = dbd[1];
        temper[tididx+base] = dbd[1];
        IC[gid] = temper[tididx];
        return;
    }
    else if (gid == lastidx-1)
    {
        tid_top[4] = tid_top[3];
    }
    else if (gid == 1)
    {
        tid_top[0] = tid_top[1];
    }

    __syncthreads();

	for (int k = (height-2); k>1; k-=4)
	{
		if (tididx < (base-k) && tididx >= k)
		{
            temper[tid_top[2]] = eulerStutterStep(temper, tid_bottom);
		}

        step2 = k-2;
        __syncthreads();

        if (tididx < (base-step2) && tididx >= step2)
        {
            temper[tididx] += eulerFinalStep(temper, tid_top);

            temper[tididx].w = pressure(temper[tididx]);
        }
		//Make sure the threads are synced
		__syncthreads();
	}

    //__syncthreads();
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
    int lastidx = ((blockDim.x*gridDim.x)-1);
    int step2;

    int tid_top[5], tid_bottom[5];
    #pragma unroll
    for (int k = -2; k<3; k++)
    {
        tid_bottom[k+2] = tididx + k;
        tid_top[k+2] = tididx + k + base;
    }

    int2 idx;
	idx.x = height + ((tid/4 & 1) * base) + (tid & 3) - (4 + (tid/4) * 2); //left set
	idx.y = height + ((tid/4 & 1) * base) + (tid/4)*2 + (tid & 3); //right set


	// Initialize temper. Kind of an unrolled for loop.  This is actually at
	// Timestep 0.

    if (full)
    {
        temper[idx.x] = right[gid];
        temper[idx.y] = left[gid];
    }
    else
    {
        gid += blockDim.x;
        temper[idx.x] = right[gid];
        temper[idx.y] = left[gid];
    }

    __syncthreads();

    if (full)
    {
        if (gid == 0)
        {
            temper[tididx] = dbd[0];
            temper[tididx+base] = dbd[0];
        }
        else if (gid == lastidx)
        {
            temper[tididx] = dbd[1];
            temper[tididx+base] = dbd[1];
        }
        else if (gid == lastidx-1)
        {
            tid_top[4] = tid_top[3];
        }
        else if (gid == 1)
        {
            tid_top[0] = tid_top[1];
        }
    }
    __syncthreads();

    if (tididx < (base-(height-2)) && tididx >= (height-2))
    {
        temper[tid_top[2]] = eulerStutterStep(temper, tid_bottom);
    }


    for (int k = (height-4); k>4; k-=4)
    {
        if (tididx < (base-k) && tididx >= k)
        {
            temper[tididx] += eulerFinalStep(temper, tid_top);

            temper[tididx].w = pressure(temper[tididx]);
        }

        step2 = k-2;
        __syncthreads();

        if (tididx < (base-step2) && tididx >= step2)
        {
            temper[tid_top[2]] = eulerStutterStep(temper, tid_bottom);
        }
        //Make sure the threads are synced
        __syncthreads();
    }

    // -------------------TOP PART------------------------------------------
    idx.x = ((tid/4 & 1) * base) + (tid/4)*2 + (tid & 3) + 2; //left get
    idx.y = (base-6) + ((tid/4 & 1) * base) + (tid & 3) - (tid/4)*2; //right get
    //	int leftidx = height + ((tid/4 & 1) * base) + (tid & 3) - (4 + (tid/4) * 2);
    //	int rightidx = height + ((tid/4 & 1) * base) + (tid/4)*2 + (tid & 3);

    if (full)
        if (gid > 0 &&  gid < lastidx)
        {
            temper[tididx] += eulerFinalStep(temper, tid_top);

            temper[tididx].w = pressure(temper[tididx]);
        }
    else
    {
        temper[tididx] += eulerFinalStep(temper, tid_top);

        temper[tididx].w = pressure(temper[tididx]);
    }

    __syncthreads();

    if (tid > 1 && tid <(blockDim.x-2))
	{
        temper[tid_top[2]] = eulerStutterStep(temper, tid_bottom);
	}
	//The initial conditions are timeslice 0 so start k at 1.

	__syncthreads();

    //The initial conditions are timslice 0 so start k at 1.
	for (int k = 4; k<(blockDim.x/2); k+=4)
	{
		if (tid < (blockDim.x-k) && tid >= k)
		{
            temper[tididx] += eulerFinalStep(temper, tid_top);

            temper[tididx].w = pressure(temper[tididx]);
		}
        step2 = k+2;
        __syncthreads();

        if (tid < (blockDim.x-step2) && tid >= step2)
        {
            temper[tid_top[2]] = eulerStutterStep(temper, tid_bottom);
		}

		//Make sure the threads are synced
		__syncthreads();

	}

    right[gid] = temper[idx.y];
	left[gid] = temper[idx.x];

}

//Split one is always first.
__global__
void
splitDiamond(REALfour *right, REALfour *left)
{
    extern __shared__ REALfour temper[];

    //Same as upTriangle
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
    int tididx = tid + 2;
	int base = blockDim.x + 4;
	int height = base/2;
    int height2 = height - 2;
    int step2;

    bool t_ht = (gid == height);
    bool t_htm = (gid == (height-1));

    int tid_top[5], tid_bottom[5];
    #pragma unroll
    for (int k = -2; k<3; k++)
    {
        tid_bottom[k+2] = tididx + k;
        tid_top[k+2] = tididx + k + base;
    }

    int2 idx;
	idx.x = height + ((tid/4 & 1) * base) + (tid & 3) - (4 + (tid/4) * 2); //left set
	idx.y = height + ((tid/4 & 1) * base) + (tid/4)*2 + (tid & 3); //right set
	// Initialize temper. Kind of an unrolled for loop.  This is actually at
	// Timestep 0.

    temper[idx.x] = right[gid];
	temper[idx.y] = left[gid];

    __syncthreads();

    if (t_ht)
    {
        temper[tididx] = dbd[0];
        temper[tididx+base] = dbd[0];
    }
    else if (t_htm)
    {
        temper[tididx] = dbd[1];
        temper[tididx+base] = dbd[1];
    }
    else if (gid == height2)
    {
        tid_top[4] = tid_top[3];
    }
    else if (gid == (height+1))
    {
        tid_top[0] = tid_top[1];
    }

    for (int k = height2; k>0; k-=4)
    {

        if (!t_ht && !t_htm && tididx < (base-k) && tididx >= k)
        {
            temper[tid_top[2]] = eulerStutterStep(temper, tid_bottom);
        }

        step2 = k-2;
        __syncthreads();

        if (!t_ht && !t_htm && tididx < (base-step2) && tididx >= step2)
        {
            temper[tididx] += eulerFinalStep(temper, tid_top);

            temper[tididx].w = pressure(temper[tididx]);
        }

        __syncthreads();
    }

    idx.x = ((tid/4 & 1) * base) + (tid/4)*2 + (tid & 3) + 2; //left get
    idx.y = (base-6) + ((tid/4 & 1) * base) + (tid & 3) - (tid/4)*2; //right get

    __syncthreads();

    if (!t_ht && !t_htm && tid > 1 && tid <(blockDim.x-2))
	{
        temper[tid_top[2]] = eulerStutterStep(temper, tid_bottom);
	}

	__syncthreads();


    //The initial conditions are timslice 0 so start k at 1.
    for (int k = 4; k<(blockDim.x/2); k+=4)
    {
        if (!t_ht && !t_htm && tid < (blockDim.x-k) && tid >= k)
        {
            temper[tididx] += eulerFinalStep(temper, tid_top);

            temper[tididx].w = pressure(temper[tididx]);
        }
        if (tid == 0) temper[tid].x = k+2;
        __syncthreads();

        if (!t_ht && !t_htm && tid < (blockDim.x-(int)temper[0].x) && tid >= (int)temper[tid].x)
        {
            temper[tid_top[2]] = eulerStutterStep(temper, tid_bottom);
        }

        //Make sure the threads are synced
        __syncthreads();

    }

    right[gid] = temper[idx.y];
    left[gid] = temper[idx.x];
}

using namespace std;

__host__
void
CPU_diamond(REALfour *temper, int tpb)
{
    int step2;
    int base = tpb + 4;
    int height = base/2;
    int height2 = height-2;

    //Splitting it is the whole point!
    for (int k = height2; k>0; k-=4)
    {
        for(int n = k; n<(base-k); n++)
        {
            if (n == (height-1)) //case 1
            {
                temper[n+base] = bd[1];
            }
            else if (n == height)  //case 2
            {
                temper[n+base] = bd[0];
            }
            else if (n == height2) //case 0
            {
                temper[n+base] = eulerStutterClassicStep(temper[n-2].w, temper[n-1], temper[n],
                    bd[1], bd[1].w);
            }
            else if (n == (height+1)) //case 3
            {
                temper[n+base] = eulerStutterClassicStep(bd[0].w, bd[0], temper[n],
                    temper[n+1], temper[n+2].w);
            }
            else
            {
                temper[n+base] = eulerStutterClassicStep(temper[n-2].w, temper[n-1], temper[n],
                    temper[n+1], temper[n+2].w);
            }
        }

        step2 = k-2;

        for(int n = step2; n<(base-step2); n++)
        {
            if (n == (height-1)) //case 1
            {
                temper[n] = bd[1];
            }
            else if (n == height)  //case 2
            {
                temper[n] = bd[0];
            }
            else if (n == height2) //case 0
            {
                temper[n] += eulerFinalClassicStep(temper[base+n-2].w, temper[base+n-1], temper[base+n],
                    bd[1], bd[1].w);
                temper[n].w = pressure(temper[n]);
            }
            else if (n == (height+1)) //case 3
            {
                temper[n] += eulerFinalClassicStep(bd[0].w, bd[0], temper[base+n],
                    temper[base+n+1], temper[base+n+2].w);
                temper[n].w = pressure(temper[n]);
            }
            else
            {
                temper[n] += eulerFinalClassicStep(temper[base+n-2].w, temper[base+n-1], temper[base+n],
                    temper[base+n+1], temper[base+n+2].w);
                temper[n].w = pressure(temper[n]);
            }
        }
    }

    for (int k = 0; k<tpb; k++) temper[k] = temper[k+2];

    height -= 2;
    height2 -= 2;

    for(int n = 2; n<(tpb-2); n++)
    {
        if (n == (height-1)) //case 1
        {
            temper[n+tpb] = bd[1];
        }
        else if (n == height)  //case 2
        {
            temper[n+tpb] = bd[0];
        }
        else if (n == height2) //case 0
        {
            temper[n+tpb] = eulerStutterClassicStep(temper[n-2].w, temper[n-1], temper[n],
                bd[1], bd[1].w);
        }
        else if (n == (height+1)) //case 3
        {
            temper[n+tpb] = eulerStutterClassicStep(bd[0].w, bd[0], temper[n],
                temper[n+1], temper[n+2].w);
        }
        else
        {
            temper[n+tpb] = eulerStutterClassicStep(temper[n-2].w, temper[n-1], temper[n],
                temper[n+1], temper[n+2].w);
        }
    }

    //Top part.
    for (int k = 4; k<height; k+=4)
    {
        for(int n = k; n<(tpb-k); n++)
        {
            if (n == (height-1)) //case 1
            {
                temper[n] = bd[1];
            }
            else if (n == height)  //case 2
            {
                temper[n] = bd[0];
            }
            else if (n == height2) //case 0
            {
                temper[n] += eulerFinalClassicStep(temper[base+n-2].w, temper[base+n-1], temper[base+n],
                    bd[1], bd[1].w);
                temper[n].w = pressure(temper[n]);
            }
            else if (n == (height+1)) //case 3
            {
                temper[n] += eulerFinalClassicStep(bd[0].w, bd[0], temper[base+n],
                    temper[base+n+1], temper[base+n+2].w);
                temper[n].w = pressure(temper[n]);
            }
            else
            {
                temper[n] += eulerFinalClassicStep(temper[base+n-2].w, temper[base+n-1], temper[base+n],
                    temper[base+n+1], temper[base+n+2].w);
                temper[n].w = pressure(temper[n]);
            }
        }

        step2 = k+2;

        for(int n = step2; n<(tpb-step2); n++)
        {
            if (n == (height-1)) //case 1
            {
                temper[n+tpb] = bd[1];
            }
            else if (n == height)  //case 2
            {
                temper[n+tpb] = bd[0];
            }
            else if (n == height2) //case 0
            {
                temper[n+tpb] = eulerStutterClassicStep(temper[n-2].w, temper[n-1], temper[n],
                    bd[1], bd[1].w);
            }
            else if (n == (height+1)) //case 3
            {
                temper[n+tpb] = eulerStutterClassicStep(bd[0].w, bd[0], temper[n],
                    temper[n+1], temper[n+2].w);
            }
            else
            {
                temper[n+tpb] = eulerStutterClassicStep(temper[n-2].w, temper[n-1], temper[n],
                    temper[n+1], temper[n+2].w);
            }
        }
    }
}

REAL
__host__ __inline__
energy(REAL p, REAL rho, REAL u)
{
    return (p/(m_gamma*rho) + 0.5*rho*u*u);
}

//Classic Discretization wrapper.
double
classicWrapper(const int bks, int tpb, const int dv, const REAL dt, const REAL t_end,
    REALfour *IC, REALfour *T_f, const float freq, ofstream &fwr)
{
    REALfour *dEuler_in, *dEuler_out;

    cudaMalloc((void **)&dEuler_in, sizeof(REALfour)*dv);
    cudaMalloc((void **)&dEuler_out, sizeof(REALfour)*dv);

    // Copy the initial conditions to the device array.
    cudaMemcpy(dEuler_in,IC,sizeof(REALfour)*dv,cudaMemcpyHostToDevice);

    double t_eq = 0.0;
    double twrite = freq;

    while (t_eq < t_end)
    {
        classicEuler <<< bks,tpb >>> (dEuler_in, dEuler_out, false);
        classicEuler <<< bks,tpb >>> (dEuler_out, dEuler_in, true);
        t_eq += dt;

        if (t_eq > twrite)
        {
            cudaMemcpy(T_f, dEuler_in, sizeof(REALfour)*dv, cudaMemcpyDeviceToHost);

            fwr << " Density " << t_eq << " ";
            for (int k = 0; k<dv; k++) fwr << T_f[k].x << " ";
            fwr << endl;

            fwr << " Velocity " << t_eq << " ";
            for (int k = 0; k<dv; k++) fwr << T_f[k].y/T_f[k].x << " ";
            fwr << endl;

            fwr << " Energy " << t_eq << " ";
            for (int k = 0; k<dv; k++) fwr << (T_f[k].z/T_f[k].x) << " ";
            fwr << endl;

            fwr << " Pressure " << t_eq << " ";
            for (int k = 0; k<dv; k++) fwr << T_f[k].w << " ";
            fwr << endl;

            twrite += freq;
        }
    }

    cudaMemcpy(T_f, dEuler_in, sizeof(REALfour)*dv, cudaMemcpyDeviceToHost);

    cudaFree(dEuler_in);
    cudaFree(dEuler_out);

    return t_eq;

}

//The wrapper that calls the routine functions.
double
sweptWrapper(const int bks, int tpb, const int dv, REAL dt, const REAL t_end, const int cpu,
    REALfour *IC, REALfour *T_f, const float freq, ofstream &fwr)
{
    const int base = tpb + 4;
    const int height = base/2;
    const size_t smem1 = 2*tpb*sizeof(REALfour);
    const size_t smem2 = (2*base)*sizeof(REALfour);

    int indices[4][tpb];

    for (int k = 0; k<tpb; k++)
    {
        //Set indices
        indices[0][k] = height + ((k/4 & 1) * base) + (k & 3) - (4 + (k/4) *2); //left
        indices[1][k] = height + ((k/4 & 1) * base) + (k & 3) +  (k/4)*2; // right
        //Get indices
        indices[2][k] = (k/4)*2 + ((k/4 & 1) * tpb) + (k & 3); //left
        indices[3][k] = (tpb - 4) + ((k/4 & 1) * tpb) + (k & 3) -  (k/4)*2; //right
    }

	REALfour *d_IC, *d_right, *d_left, *d_bin;

	cudaMalloc((void **)&d_IC, sizeof(REALfour)*dv);
	cudaMalloc((void **)&d_right, sizeof(REALfour)*dv);
	cudaMalloc((void **)&d_left, sizeof(REALfour)*dv);
    cudaMalloc((void **)&d_bin, sizeof(REALfour)*dv);

	// Copy the initial conditions to the device array.
	cudaMemcpy(d_IC,IC,sizeof(REALfour)*dv,cudaMemcpyHostToDevice);
	// Start the counter and start the clock.
	const double t_fullstep = 0.25*dt*(double)tpb;

	upTriangle <<< bks,tpb,smem1 >>>(d_IC,d_right,d_left);

    swapKernel <<< bks,tpb >>> (d_right, d_bin, 1);
    swapKernel <<< bks,tpb >>> (d_bin, d_right, 0);

    double t_eq;
    double twrite = freq;

	// Call the kernels until you reach the iteration limit.

    if (cpu)
    {
        REALfour *tmpr = (REALfour*)malloc(smem2);
        REALfour *h_right, *h_left;
        cudaHostAlloc((void **) &h_right, tpb*sizeof(REALfour), cudaHostAllocDefault);
        cudaHostAlloc((void **) &h_left, tpb*sizeof(REALfour), cudaHostAllocDefault);

        // h_right = (REALfour *) malloc(tpb*sizeof(REALfour));
        // h_left = (REALfour *) malloc(tpb*sizeof(REALfour));

        t_eq = t_fullstep;
        omp_set_num_threads( 2 );

        //Split Diamond Begin------



        #pragma omp parallel sections
        {
        #pragma omp section
        {
            cudaMemcpy(h_right, d_left, tpb*sizeof(REALfour), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_left, d_right , tpb*sizeof(REALfour), cudaMemcpyDeviceToHost);
            double time0 = omp_get_wtime( );
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
            double time1 = omp_get_wtime( );
            double tf = (time1-time0)*1.0e6; //In us

            cout << "CPU time: " << tf << endl;
            cudaMemcpy(d_right, h_right, tpb*sizeof(REALfour), cudaMemcpyHostToDevice);
            cudaMemcpy(d_left, h_left, tpb*sizeof(REALfour), cudaMemcpyHostToDevice);
        }
        #pragma omp section
        {
            wholeDiamond <<< bks-1,tpb,smem2 >>>(d_right,d_left,false);
        }
        }

        swapKernel <<< bks,tpb >>> (d_left, d_bin, -1);
        swapKernel <<< bks,tpb >>> (d_bin, d_left, 0);

        while(t_eq < t_end)
        {

            wholeDiamond <<< bks,tpb,smem2 >>>(d_right,d_left,true);

            swapKernel <<< bks,tpb >>> (d_right, d_bin, 1);
            swapKernel <<< bks,tpb >>> (d_bin, d_right, 0);

            //Split Diamond Begin------

            cudaMemcpy(h_right,d_left, tpb*sizeof(REALfour), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_left, d_right, tpb*sizeof(REALfour), cudaMemcpyDeviceToHost);

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

            cudaMemcpy(d_right, h_right, tpb*sizeof(REALfour), cudaMemcpyHostToDevice);
            cudaMemcpy(d_left, h_left, tpb*sizeof(REALfour), cudaMemcpyHostToDevice);

            swapKernel <<< bks,tpb >>> (d_left, d_bin, -1);
            swapKernel <<< bks,tpb >>> (d_bin, d_left, 0);

            //Split Diamond End------

            t_eq += t_fullstep;

    	    if (t_eq > twrite)
    		{
    			downTriangle <<< bks,tpb,smem2 >>>(d_IC,d_right,d_left);

    			cudaMemcpy(T_f, d_IC, sizeof(REALfour)*dv, cudaMemcpyDeviceToHost);

                fwr << " Density " << t_eq << " ";
                for (int k = 0; k<dv; k++) fwr << T_f[k].x << " ";
                fwr << endl;

                fwr << " Velocity " << t_eq << " ";
                for (int k = 0; k<dv; k++) fwr << (T_f[k].y/T_f[k].x) << " ";
                fwr << endl;

                fwr << " Energy " << t_eq << " ";
                for (int k = 0; k<dv; k++) fwr << (T_f[k].z/T_f[k].x) << " ";
                fwr << endl;

                fwr << " Pressure " << t_eq << " ";
                for (int k = 0; k<dv; k++) fwr << T_f[k].w << " ";
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

            t_eq += t_fullstep;

            if (t_eq > twrite)
            {
                downTriangle <<< bks,tpb,smem2 >>>(d_IC,d_right,d_left);

                cudaMemcpy(T_f, d_IC, sizeof(REALfour)*dv, cudaMemcpyDeviceToHost);

                fwr << " Density " << t_eq << " ";
            	for (int k = 0; k<dv; k++) fwr << T_f[k].x << " ";
                fwr << endl;

                fwr << " Velocity " << t_eq << " ";
            	for (int k = 0; k<dv; k++) fwr << (T_f[k].y/T_f[k].x) << " ";
                fwr << endl;

                fwr << " Energy " << t_eq << " ";
                for (int k = 0; k<dv; k++) fwr << (T_f[k].z/T_f[k].x) << " ";
                fwr << endl;

                fwr << " Pressure " << t_eq << " ";
                for (int k = 0; k<dv; k++) fwr << T_f[k].w << " ";
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

	cudaMemcpy(T_f, d_IC, sizeof(REALfour)*dv, cudaMemcpyDeviceToHost);

	cudaFree(d_IC);
	cudaFree(d_right);
	cudaFree(d_left);
    cudaFree(d_bin);

    return t_eq;
}

int main( int argc, char *argv[] )
{
    //That is, there are less than 8 arguments.
    if (argc < 9)
	{
		cout << "The Program takes 9 inputs, #Divisions, #Threads/block, deltat, finish time, output frequency..." << endl;
        cout << "Classic/Swept, CPU sharing Y/N, Variable Output File, Timing Output File (optional)" << endl;
		exit(-1);
	}

	// Choose the GPGPU.  This is device 0 in my machine which has 2 devices.
	cudaSetDevice(0);
    if (sizeof(REAL)>6) cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    bd[0].x = 1.0; //Density
    bd[1].x = 0.125;
    bd[0].y = 0.0; //Velocity
    bd[1].y = 0.0;
    bd[0].w = 1.0; //Pressure
    bd[1].w = 0.1;
    bd[0].z = bd[0].w/m_gamma; //Energy
    bd[1].z = bd[1].w/m_gamma;


    const int dv = atoi(argv[1]); //Number of spatial points
	const int tpb = atoi(argv[2]); //Threads per Block
    const REAL dt = atof(argv[3]);
	const float tf = atof(argv[4]); //Finish time
    const float freq = atof(argv[5]);
    const int scheme = atoi(argv[6]); //1 for Swept 0 for classic
    const int share = atoi(argv[7]);
    const int bks = dv/tpb; //The number of blocks
    const REAL dx = lx/((REAL)dv-1.f);
    char const *prec;
    prec = (sizeof(REAL)<6) ? "Single": "Double";

    //Declare the dimensions in constant memory.
    dimz.x = dt/dx; // dt/dx
    dimz.y = gam;
    dimz.z = m_gamma;

    cout << "Euler --- #Blocks: " << bks << " | Length: " << lx << " | Precision: " << prec << " | dt/dx: " << dimz.x << endl;

	//Conditions for main input.  Unit testing kinda.
	//dv and tpb must be powers of two.  dv must be larger than tpb and divisible by
	//tpb.

	if ((dv & (tpb-1) !=0) || (tpb&31) != 0)
    {
        cout << "INVALID NUMERIC INPUT!! "<< endl;
        cout << "2nd ARGUMENT MUST BE A POWER OF TWO >= 32 AND FIRST ARGUMENT MUST BE DIVISIBLE BY SECOND" << endl;
        exit(-1);
    }

    if (dimz.x > .1)
    {
        cout << "The value of dt/dx (" << dimz.x << ") is too high.  In general it must be <=.1 for stability." << endl;
        exit(-1);
    }

	// Initialize arrays.
    REALfour *IC, *T_final;
	cudaHostAlloc((void **) &IC, dv*sizeof(REALfour), cudaHostAllocDefault);
	cudaHostAlloc((void **) &T_final, dv*sizeof(REALfour), cudaHostAllocDefault);
    // IC = (REALfour *) malloc(dv*sizeof(REALfour));
    // T_final = (REALfour *) malloc(dv*sizeof(REALfour));


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
	ofstream fwr;
	fwr.open(argv[8],ios::trunc);
	// Write out x length and then delta x and then delta t.
	// First item of each line is variable second is timestamp.
	// energy(IC[k].w, IC[k].x, IC[k].y/IC[k].x)
	fwr << lx << " " << dv << " " << dx << " " << endl;

    fwr << " Density " << 0 << " ";
    for (int k = 0; k<dv; k++) fwr << IC[k].x << " ";
    fwr << endl;

    fwr << " Velocity " << 0 << " ";
    for (int k = 0; k<dv; k++) fwr << IC[k].y << " ";
    fwr << endl;

    fwr << " Energy " << 0 << " ";
    for (int k = 0; k<dv; k++) fwr << IC[k].z/IC[k].x << " ";
    fwr << endl;

    fwr << " Pressure " << 0 << " ";
    for (int k = 0; k<dv; k++) fwr << IC[k].w << " ";
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

    cout << scheme << " ";
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

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

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

    //energy(T_final[k].w, T_final[k].x, T_final[k].y/T_final[k].x)

	fwr << " Density " << tfm << " ";
	for (int k = 0; k<dv; k++) fwr << T_final[k].x << " ";
    fwr << endl;

    fwr << " Velocity " << tfm << " ";
	for (int k = 0; k<dv; k++) fwr << T_final[k].y/T_final[k].x << " ";
    fwr << endl;

    fwr << " Energy " << tfm << " ";
    for (int k = 0; k<dv; k++) fwr << T_final[k].z/T_final[k].x << " ";
    fwr << endl;

    fwr << " Pressure " << tfm << " ";
    for (int k = 0; k<dv; k++) fwr << T_final[k].w << " ";
    fwr << endl;

	fwr.close();

	// Free the memory and reset the device.

    cudaDeviceSynchronize();

	cudaEventDestroy( start );
	cudaEventDestroy( stop );
    cudaDeviceReset();

    cudaFreeHost(IC);
    cudaFreeHost(T_final);
    // free(IC);
    // free(T_final);

	return 0;

}
