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
int4 h_geo;
//dbd is the boundary condition
__constant__ REALfour dbd[2]; //0 is left 1 is right.
//dimens has three fields x is dt/dx, y is gamma, z is gamma-1
__constant__ REALthree dimens;
//geo.x is number of divisions. y is base, z is last idx or div-1
__constant__ int4 geo;

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
eulerStutterStep(REAL pfarLeft, REALfour stateLeft, REALfour stateCenter, REALfour stateRight, REAL pfarRight)
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

//Same thing as the predictor step, but this final step adds the result to the original state variables to advance to the next timestep.
//But the predictor variables to find the fluxes.
__device__ __host__
REALfour
eulerFinalStep(REAL pfarLeft, REALfour stateLeft, REALfour stateCenter, REALfour stateRight, REAL pfarRight)
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

__global__
void
swapKernel(const REALfour *passing_side, REALfour *bin, int direction)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
    int gidout = (gid + direction*blockDim.x) & geo.z;

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
            euler_out[gid] += eulerFinalStep(dbd[0].w,dbd[0],euler_in[gid],
                euler_in[(gid+1)],euler_in[(gid+2)].w);

            euler_out[gid].w = pressure(euler_out[gid]);
        }
        else if (gid == (lastidx-1))
        {
            euler_out[gid] += eulerFinalStep(euler_in[(gid-2)].w,euler_in[(gid-1)],euler_in[gid],
                dbd[1],dbd[1].w);
            euler_out[gid].w = pressure(euler_out[gid]);
        }
        else
        {
            euler_out[gid] += eulerFinalStep(euler_in[(gid-2)].w,euler_in[(gid-1)],euler_in[gid],
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
            euler_out[gid] = eulerStutterStep(dbd[0].w,dbd[0],euler_in[gid],euler_in[(gid+1)],euler_in[(gid+2)].w);
        }
        else if (gid == (lastidx-1))
        {
            euler_out[gid] = eulerStutterStep(euler_in[(gid-2)].w,euler_in[(gid-1)],euler_in[gid],dbd[1],dbd[1].w);
        }
        else
        {
            euler_out[gid] = eulerStutterStep(euler_in[(gid-2)].w,euler_in[(gid-1)],euler_in[gid],euler_in[(gid+1)],euler_in[(gid+2)].w);
        }
    }
}

__global__
void
upTriangle(const REALfour *IC, REALfour *right, REALfour *left)
{

	extern __shared__ REALfour temper[];

	int tid = threadIdx.x; //Block Thread ID

    int tid_call[5];
	#pragma unroll
	for (int k = -2; k<3; k++)	tid_call[k+2] = tid + k;

	int leftidx = ((tid/4 & 1) * blockDim.x) + (tid/4)*2 + (tid & 3);
	int rightidx = (blockDim.x - 4) + ((tid/4 & 1) * blockDim.x) + (tid & 3) - (tid/4)*2;

    int step2;

    //Assign the initial values to the first row in temper, each block
    //has it's own version of temper shared among its threads.

    for (int gid = blockDim.x * blockIdx.x + threadIdx.x;
        gid < geo.x;
        gid += blockDim.x*gridDim.x)
    {
    	temper[tid] = IC[gid];

        __syncthreads();

    	if (tid > 1 && tid <(blockDim.x-2))
    	{
    		temper[tid+blockDim.x] = eulerStutterStep(temper[tid_call[0]].w, temper[tid_call[1]], temper[tid_call[2]],
    			temper[tid_call[3]], temper[tid_call[4]].w);
    	}

    	__syncthreads();

    	//The initial conditions are timslice 0 so start k at 1.
    	for (int k = 4; k<(blockDim.x/2); k+=4)
    	{
    		if (tid < (blockDim.x-k) && tid >= k)
    		{
    			temper[tid] += eulerFinalStep(temper[tid_call[0]+blockDim.x].w, temper[tid_call[1]+blockDim.x], temper[tid_call[2]+blockDim.x],
        			temper[tid_call[3]+blockDim.x], temper[tid_call[4]+blockDim.x].w);

                temper[tid].w = pressure(temper[tid]);
    		}

            step2 = k+2;
    		__syncthreads();

    		if (tid < (blockDim.x-step2) && tid >= step2)
    		{
                temper[tid+blockDim.x] = eulerStutterStep(temper[tid_call[0]].w, temper[tid_call[1]], temper[tid_call[2]],
        			temper[tid_call[3]], temper[tid_call[4]].w);
    		}

    		//Make sure the threads are synced
    		__syncthreads();

    	}

    	//After the triangle has been computed, the right and left shared arrays are
    	//stored in global memory by the global thread ID since (conveniently),
    	//they're the same size as a warp!
    	right[gid] = temper[rightidx];
    	left[gid] = temper[leftidx];

        __syncthreads();

    }
}

// Down triangle is only called at the end when data is passed left.  It's never split.
// It returns IC which is a full 1D result at a certain time.
__global__
void
downTriangle(REALfour *IC, const REALfour *right, const REALfour *left)
{
	extern __shared__ REALfour temper[];

	int tid = threadIdx.x;
	int tididx = tid + 2;
    int step2;

	int height = geo.y/2;
    int height2 = height-2;

	int leftidx = height + ((tid/4 & 1) * geo.y) + (tid & 3) - (4 + (tid/4) * 2);
	int rightidx = height + ((tid/4 & 1) * geo.y) + (tid/4)*2 + (tid & 3);

    //Ignore first and last block of global memory.  We're going to make a new kernel for them.
    for (int gid = blockDim.x * blockIdx.x + threadIdx.x;
        gid < geo.x;
        gid += blockDim.x*gridDim.x)
    {
    	temper[leftidx] = right[gid];
    	temper[rightidx] = left[gid];

        int tid_call[5];
        #pragma unroll
        for (int k = -2; k<3; k++)
        {
            tid_call[k+2] = tididx + k;
        }

        bool t_ht = (gid == 0);
        bool t_htm = (gid == geo.z);

        __syncthreads();

        if (t_ht)
        {
            temper[tididx] = dbd[0];
            temper[tididx+geo.y] = dbd[0];
        }
        else if (t_htm)
        {
            temper[tididx] = dbd[1];
            temper[tididx+geo.y] = dbd[1];
        }
        else if (gid == geo.z-1)
        {
            tid_call[4] = tid_call[3];
        }
        else if (gid == 1)
        {
            tid_call[0] = tid_call[1];
        }

    	for (int k = height2; k>1; k-=4)
    	{
    		if (tididx < (geo.y-k) && tididx >= k)
    		{
                temper[tididx+geo.y] = eulerStutterStep(temper[tid_call[0]].w, temper[tid_call[1]], temper[tid_call[2]],
                    temper[tid_call[3]], temper[tid_call[4]].w);
    		}

            step2 = k-2;
            __syncthreads();

            if (t_ht && t_htm && tididx < (geo.y-step2) && tididx >= step2)
            {
                temper[tididx] += eulerFinalStep(temper[tid_call[0]+geo.y].w, temper[tid_call[1]+geo.y], temper[tid_call[2]+geo.y],
    				temper[tid_call[3]+geo.y], temper[tid_call[4]+geo.y].w);

                temper[tididx].w = pressure(temper[tididx]);
            }
    		//Make sure the threads are synced
    		__syncthreads();
    	}

        //__syncthreads();
        IC[gid] = temper[tididx];

        __syncthreads();
    }
}

//Full refers to whether or not there is a node run on the CPU.
__global__
void
wholeDiamond(REALfour *right, REALfour *left, bool split)
{

    extern __shared__ REALfour temper[];

	int tid = threadIdx.x;
	int tididx = tid + 2;
    int step2;

    const int height = geo.y/2;
    const int height2 = height-2;

    int tid_call[5];
	#pragma unroll
	for (int k = -2; k<3; k++)
    {
        tid_call[k+2] = tididx + k;
    }

    int4 idx;
	idx.x = height + ((tid/4 & 1) * geo.y) + (tid & 3) - (4 + (tid/4) * 2); //left set
	idx.y = height + ((tid/4 & 1) * geo.y) + (tid/4)*2 + (tid & 3); //right set
    idx.z = ((tid/4 & 1) * geo.y) + (tid/4)*2 + (tid & 3) + 2; //left get
    idx.w = (geo.y-6) + ((tid/4 & 1) * geo.y) + (tid & 3) - (tid/4)*2; //right get

    for (int gid = blockDim.x * blockIdx.x + threadIdx.x + blockDim.x;
        gid < geo.x-int(!split)*blockDim.x;
        gid += blockDim.x*gridDim.x)
    {

        temper[idx.x] = right[gid];
        temper[idx.y] = left[gid];

        __syncthreads();


        for (int k = height2; k>1; k-=4)
        {
            if (tididx < (geo.y-k) && tididx >= k)
            {
                temper[tididx+geo.y] = eulerStutterStep(temper[tid_call[0]].w, temper[tid_call[1]], temper[tid_call[2]],
                    temper[tid_call[3]], temper[tid_call[4]].w);
            }

            step2 = k+2;
            __syncthreads();

            if (tididx < (geo.y-step2) && tididx >= step2)
            {
                temper[tididx] += eulerFinalStep(temper[tid_call[0]+geo.y].w, temper[tid_call[1]+geo.y], temper[tid_call[2]+geo.y],
    				temper[tid_call[3]+geo.y], temper[tid_call[4]+geo.y].w);

                temper[tididx].w = pressure(temper[tididx]);
            }
            //Make sure the threads are synced
            __syncthreads();
        }

    // -------------------TOP PART------------------------------------------
        if (tid >1  && tid >= (blockDim.x-2))
		{
            temper[tididx+geo.y] = eulerStutterStep(temper[tid_call[0]].w, temper[tid_call[1]], temper[tid_call[2]],
                temper[tid_call[3]], temper[tid_call[4]].w);
		}

    	__syncthreads();

    	for (int k = 4; k<(height2-2); k+=4)
    	{
    		if (tid < (blockDim.x-k) && tid >= k)
    		{
                temper[tididx] += eulerFinalStep(temper[tid_call[0]+geo.y].w, temper[tid_call[1]+geo.y], temper[tid_call[2]+geo.y],
    				temper[tid_call[3]+geo.y], temper[tid_call[4]+geo.y].w);

                temper[tididx].w = pressure(temper[tididx]);
    		}

            step2 = k+2;
    		__syncthreads();

    		if (tid < (blockDim.x-step2) && tid >= step2)
    		{
                temper[tididx+geo.y] = eulerStutterStep(temper[tid_call[0]].w, temper[tid_call[1]], temper[tid_call[2]],
                    temper[tid_call[3]], temper[tid_call[4]].w);
    		}

    		//Make sure the threads are synced
    		__syncthreads();

    	}

        left[gid] = temper[idx.z];
        right[gid] = temper[idx.w];
    }
}

//Split one is always first.
__global__
void
edgeDiamond(REALfour *right, REALfour *left, int split, int edge)
{
    extern __shared__ REALfour temper[];

    //Same as upTriangle
	int tid = threadIdx.x;
    int gid = (!edge || split) ? (tid) : (geo.x-blockDim.x) + tid;
    int tididx = tid + 2;
    int step2;
    bool t_ht;

	const int height = geo.y/2;
    const int height2 = height-2;

    int tid_call[5];
    #pragma unroll
    for (int k = -2; k<3; k++)
    {
        tid_call[k+2] = tididx + k;
    }

    int4 idx;
	idx.x = height + ((tid/4 & 1) * geo.y) + (tid & 3) - (4 + (tid/4) * 2); //left set
	idx.y = height + ((tid/4 & 1) * geo.y) + (tid/4)*2 + (tid & 3); //right set
    idx.z = ((tid/4 & 1) * geo.y) + (tid/4)*2 + (tid & 3) + 2; //left get
    idx.w = (geo.y-6) + ((tid/4 & 1) * geo.y) + (tid & 3) - (tid/4)*2; //right get

    temper[idx.x] = right[gid];
    temper[idx.y] = left[gid];

    if (split)
    {
        t_ht = (gid == height);
        bool t_htm = (gid == (height-1));

        __syncthreads();

        if (t_ht)
        {
            temper[tididx] = dbd[0];
            temper[tididx+geo.y] = dbd[0];
        }
        else if (t_htm)
        {
            temper[tididx] = dbd[1];
            temper[tididx+geo.y] = dbd[1];
        }
        else if (gid == height2)
        {
            tid_call[4] = tid_call[3];
        }
        else if (gid == (height+1))
        {
            tid_call[0] = tid_call[1];
        }

        for (int k = height2; k>1; k-=4)
        {
            if (!t_ht && !t_htm && tididx < (geo.y-k) && tididx >= k)
            {
                temper[tididx+geo.y] = eulerStutterStep(temper[tid_call[0]].w, temper[tid_call[1]], temper[tid_call[2]],
                    temper[tid_call[3]], temper[tid_call[4]].w);
            }

            step2 = k-2;
            __syncthreads();

            if (!t_ht && !t_htm && tididx < (geo.y-step2) && tididx >= step2)
            {
                temper[tididx] += eulerFinalStep(temper[tid_call[0]+geo.y].w, temper[tid_call[1]+geo.y], temper[tid_call[2]+geo.y],
    				temper[tid_call[3]+geo.y], temper[tid_call[4]+geo.y].w);

                temper[tididx].w = pressure(temper[tididx]);
            }

            __syncthreads();
        }


        if (!t_ht && !t_htm && tid > 1 && tid <(blockDim.x-2))
    	{
            temper[tididx+geo.y] = eulerStutterStep(temper[tid_call[0]].w, temper[tid_call[1]], temper[tid_call[2]],
                temper[tid_call[3]], temper[tid_call[4]].w);
    	}

        __syncthreads();

        //The initial conditions are timslice 0 so start k at 1.
        for (int k = 4; k<(height2-2); k+=4)
        {
            if (!t_ht && !t_htm && tid < (blockDim.x-k) && tid >= k)
            {
                temper[tididx] += eulerFinalStep(temper[tid_call[0]+geo.y].w, temper[tid_call[1]+geo.y], temper[tid_call[2]+geo.y],
                    temper[tid_call[3]+geo.y], temper[tid_call[4]+geo.y].w);

                temper[tididx].w = pressure(temper[tididx]);
            }
            step2 = k+2;
    		__syncthreads();

            if (!t_ht && !t_htm && tid < (blockDim.x-step2) && tid >= step2)
            {
                temper[tididx+geo.y] = eulerStutterStep(temper[tid_call[0]].w, temper[tid_call[1]], temper[tid_call[2]],
                    temper[tid_call[3]], temper[tid_call[4]].w);
            }

            __syncthreads();

        }

    }
    else
    {
        if (edge) //Far edge (1)
        {
            t_ht = (gid == geo.z);
            if (t_ht)
            {
                temper[tididx] = dbd[1];
                temper[tididx+geo.y] = dbd[1];
            }
            else if (gid == (geo.z-1))
            {
                tid_call[4] = tid_call[3];
            }
        }
        else //Near edge (0)
        {
            t_ht = (gid == 0);
            if (t_ht)
            {
                temper[tididx] = dbd[1];
                temper[tididx+geo.y] = dbd[1];
            }
            else if (gid == 1)
            {
                tid_call[0] = tid_call[1];
            }

        }

        for (int k = height2; k>1; k-=4)
        {
            if (tididx < (geo.y-k) && tididx >= k)
            {
                temper[tididx+geo.y] = eulerStutterStep(temper[tid_call[0]].w, temper[tid_call[1]], temper[tid_call[2]],
                    temper[tid_call[3]], temper[tid_call[4]].w);
            }

            step2 = k+2;
            __syncthreads();

            if (!t_ht && tididx < (geo.y-step2) && tididx >= step2)
            {
                temper[tididx] += eulerFinalStep(temper[tid_call[0]+geo.y].w, temper[tid_call[1]+geo.y], temper[tid_call[2]+geo.y],
                    temper[tid_call[3]+geo.y], temper[tid_call[4]+geo.y].w);

                temper[tididx].w = pressure(temper[tididx]);
            }
            //Make sure the threads are synced
            __syncthreads();
        }


        if (tid >1  && tid >= (blockDim.x-2))
        {
            temper[tididx+geo.y] = eulerStutterStep(temper[tid_call[0]].w, temper[tid_call[1]], temper[tid_call[2]],
                temper[tid_call[3]], temper[tid_call[4]].w);
        }

        __syncthreads();

        for (int k = 4; k<(height2-2); k+=4)
        {
            if (tid < (blockDim.x-k) && tid >= k)
            {
                temper[tididx] += eulerFinalStep(temper[tid_call[0]+geo.y].w, temper[tid_call[1]+geo.y], temper[tid_call[2]+geo.y],
                    temper[tid_call[3]+geo.y], temper[tid_call[4]+geo.y].w);

                temper[tididx].w = pressure(temper[tididx]);
            }

            step2 = k+2;
            __syncthreads();

            if (tid < (blockDim.x-step2) && tid >= step2)
            {
                temper[tididx+geo.y] = eulerStutterStep(temper[tid_call[0]].w, temper[tid_call[1]], temper[tid_call[2]],
                    temper[tid_call[3]], temper[tid_call[4]].w);
            }

            //Make sure the threads are synced
            __syncthreads();

        }

    }

    right[gid] = temper[idx.w];
    left[gid] = temper[idx.z];
}

using namespace std;

// __host__
// void
// CPU_diamond(REALfour *temper, int tpb)
// {
//     int step2;
//     int base = tpb + 4;
//     int height = base/2;
//     int height2 = height-2;
//     omp_set_num_threads(8)
//
//     //Splitting it is the whole point!
//     for (int k = height2; k>0; k-=4)
//     {
//         #pragma omp parallel for
//         for(int n = k; n<(base-k); n++)
//         {
//             if (n == (height-1)) //case 1
//             {
//                 temper[n+base] = bd[1];
//             }
//             else if (n == height)  //case 2
//             {
//                 temper[n+base] = bd[0];
//             }
//             else if (n == height2) //case 0
//             {
//                 temper[n+base] = eulerStutterStep(temper[n-2].w, temper[n-1], temper[n],
//                     bd[1], bd[1].w);
//             }
//             else if (n == (height+1)) //case 3
//             {
//                 temper[n+base] = eulerStutterStep(bd[0].w, bd[0], temper[n],
//                     temper[n+1], temper[n+2].w);
//             }
//             else
//             {
//                 temper[n+base] = eulerStutterStep(temper[n-2].w, temper[n-1], temper[n],
//                     temper[n+1], temper[n+2].w);
//             }
//         }
//
//         step2 = k-2;
//
//         #pragma omp parallel for
//         for(int n = step2; n<(base-step2); n++)
//         {
//             if (n == (height-1)) //case 1
//             {
//                 temper[n] = bd[1];
//             }
//             else if (n == height)  //case 2
//             {
//                 temper[n] = bd[0];
//             }
//             else if (n == height2) //case 0
//             {
//                 temper[n] += eulerFinalStep(temper[base+n-2].w, temper[base+n-1], temper[base+n],
//                     bd[1], bd[1].w);
//                 temper[n].w = pressure(temper[n]);
//             }
//             else if (n == (height+1)) //case 3
//             {
//                 temper[n] += eulerFinalStep(bd[0].w, bd[0], temper[base+n],
//                     temper[base+n+1], temper[base+n+2].w);
//                 temper[n].w = pressure(temper[n]);
//             }
//             else
//             {
//                 temper[n] += eulerFinalStep(temper[base+n-2].w, temper[base+n-1], temper[base+n],
//                     temper[base+n+1], temper[base+n+2].w);
//                 temper[n].w = pressure(temper[n]);
//             }
//         }
//     }
//
//     #pragma omp parallel for
//     for (int k = 0; k<tpb; k++) temper[k] = temper[k+2];
//
//     height -= 2;
//     height2 -= 2;
//     #pragma omp parallel for
//     for(int n = 2; n<(tpb-2); n++)
//     {
//         if (n == (height-1)) //case 1
//         {
//             temper[n+tpb] = bd[1];
//         }
//         else if (n == height)  //case 2
//         {
//             temper[n+tpb] = bd[0];
//         }
//         else if (n == height2) //case 0
//         {
//             temper[n+tpb] = eulerStutterStep(temper[n-2].w, temper[n-1], temper[n],
//                 bd[1], bd[1].w);
//         }
//         else if (n == (height+1)) //case 3
//         {
//             temper[n+tpb] = eulerStutterStep(bd[0].w, bd[0], temper[n],
//                 temper[n+1], temper[n+2].w);
//         }
//         else
//         {
//             temper[n+tpb] = eulerStutterStep(temper[n-2].w, temper[n-1], temper[n],
//                 temper[n+1], temper[n+2].w);
//         }
//     }
//
//     //Top part.
//     for (int k = 4; k<height; k+=4)
//     {
//         #pragma omp parallel for
//         for(int n = k; n<(tpb-k); n++)
//         {
//             if (n == (height-1)) //case 1
//             {
//                 temper[n] = bd[1];
//             }
//             else if (n == height)  //case 2
//             {
//                 temper[n] = bd[0];
//             }
//             else if (n == height2) //case 0
//             {
//                 temper[n] += eulerFinalStep(temper[base+n-2].w, temper[base+n-1], temper[base+n],
//                     bd[1], bd[1].w);
//                 temper[n].w = pressure(temper[n]);
//             }
//             else if (n == (height+1)) //case 3
//             {
//                 temper[n] += eulerFinalStep(bd[0].w, bd[0], temper[base+n],
//                     temper[base+n+1], temper[base+n+2].w);
//                 temper[n].w = pressure(temper[n]);
//             }
//             else
//             {
//                 temper[n] += eulerFinalStep(temper[base+n-2].w, temper[base+n-1], temper[base+n],
//                     temper[base+n+1], temper[base+n+2].w);
//                 temper[n].w = pressure(temper[n]);
//             }
//         }
//
//         step2 = k+2;
//         #pragma omp parallel for
//         for(int n = step2; n<(tpb-step2); n++)
//         {
//             if (n == (height-1)) //case 1
//             {
//                 temper[n+tpb] = bd[1];
//             }
//             else if (n == height)  //case 2
//             {
//                 temper[n+tpb] = bd[0];
//             }
//             else if (n == height2) //case 0
//             {
//                 temper[n+tpb] = eulerStutterStep(temper[n-2].w, temper[n-1], temper[n],
//                     bd[1], bd[1].w);
//             }
//             else if (n == (height+1)) //case 3
//             {
//                 temper[n+tpb] = eulerStutterStep(bd[0].w, bd[0], temper[n],
//                     temper[n+1], temper[n+2].w);
//             }
//             else
//             {
//                 temper[n+tpb] = eulerStutterStep(temper[n-2].w, temper[n-1], temper[n],
//                     temper[n+1], temper[n+2].w);
//             }
//         }
//     }
// }

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
	REALfour *d_IC, *d_right, *d_left, *d_bin;
    const size_t smem = (2*h_geo.y)*sizeof(REALfour);
    int bks_kern = 60;

	cudaMalloc((void **)&d_IC, sizeof(REALfour)*dv);
	cudaMalloc((void **)&d_right, sizeof(REALfour)*dv);
	cudaMalloc((void **)&d_left, sizeof(REALfour)*dv);
    cudaMalloc((void **)&d_bin, sizeof(REALfour)*dv);

    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

	// Copy the initial conditions to the device array.
	cudaMemcpy(d_IC,IC,sizeof(REALfour)*dv,cudaMemcpyHostToDevice);
	// Start the counter and start the clock.
	const double t_fullstep = 0.25*dt*(double)tpb;

	upTriangle <<< bks_kern,tpb,smem >>>(d_IC,d_right,d_left);

    swapKernel <<< bks,tpb >>> (d_right, d_bin, 1);
    swapKernel <<< bks,tpb >>> (d_bin, d_right, 0);

    double t_eq;
    double twrite = freq;

	// Call the kernels until you reach the iteration limit.
    edgeDiamond <<< 1,tpb,smem,stream1 >>>(d_right,d_left, 1, 0);
    wholeDiamond <<< bks_kern,tpb,smem,stream2 >>>(d_right,d_left, true);

    swapKernel <<< bks,tpb >>> (d_left, d_bin, -1);
    swapKernel <<< bks,tpb >>> (d_bin, d_left, 0);

    t_eq = t_fullstep; //Up and split diamond was a full timestep.

    while(t_eq < t_end)
    {
        // Call the kernels until you reach the iteration limit.
        edgeDiamond <<< 1,tpb,smem,stream1 >>>(d_right,d_left, 0, 0);
        edgeDiamond <<< 1,tpb,smem,stream2 >>>(d_right,d_left, 0, 1);
        wholeDiamond <<< bks_kern,tpb,smem,stream3 >>>(d_right,d_left, false);

        swapKernel <<< bks,tpb >>> (d_right, d_bin, 1);
        swapKernel <<< bks,tpb >>> (d_bin, d_right, 0);

        edgeDiamond <<< 1,tpb,smem,stream1 >>>(d_right,d_left, 1, 0);
        wholeDiamond <<< bks_kern,tpb,smem,stream2 >>>(d_right,d_left, true);

        swapKernel <<< bks,tpb >>> (d_left, d_bin, -1);
        swapKernel <<< bks,tpb >>> (d_bin, d_left, 0);

        t_eq += t_fullstep;

        if (t_eq > twrite)
        {
            downTriangle <<< bks_kern,tpb,smem >>>(d_IC,d_right,d_left);

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

            upTriangle <<< bks_kern,tpb,smem >>>(d_IC,d_right,d_left);

            swapKernel <<< bks,tpb >>> (d_right, d_bin, 1);
            swapKernel <<< bks,tpb >>> (d_bin, d_right, 0);

            edgeDiamond <<< 1,tpb,smem,stream1 >>>(d_right,d_left, 1, 0);
            wholeDiamond <<< bks_kern,tpb,smem,stream2 >>>(d_right,d_left, true);

            swapKernel <<< bks,tpb >>> (d_left, d_bin, -1);
            swapKernel <<< bks,tpb >>> (d_bin, d_left, 0);

            t_eq += t_fullstep;

            twrite += freq;
        }
    }


	downTriangle <<< bks_kern,tpb,smem >>>(d_IC,d_right,d_left);

	cudaMemcpy(T_f, d_IC, sizeof(REALfour)*dv, cudaMemcpyDeviceToHost);

	cudaFree(d_IC);
	cudaFree(d_right);
	cudaFree(d_left);
    cudaFree(d_bin);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);

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
    // If the precision is double, config the shared memory for doubles.
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

    h_geo.x = dv;
    h_geo.y = tpb+4;
    h_geo.z = dv-1;

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
    cudaMemcpyToSymbol(geo,&h_geo,sizeof(int4));

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
