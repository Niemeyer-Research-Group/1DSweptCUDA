// Templating to reduce divergence.


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

//First idea implemented.  Change all constantly recalculated constant values to constant mem.
//How does this affect registers.
//High hopes for CUDA 8 anyway.
//Other ideas: More referencing in device functions/ don't carry pressure.

#include <ostream>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <omp.h>

#ifndef REAL
    #define REAL        float
    #define REALtwo     float2
    #define REALthree   float3
    #define REALfour    float4

    #define TWOVEC( ... ) make_float2(__VA_ARGS__)
    #define THREEVEC( ... ) make_float3(__VA_ARGS__)
    #define FOURVEC( ... )  make_float4(__VA_ARGS__)
    #define ZERO        0.0f
    #define QUARTER     0.25f
    #define HALF        0.5f
    #define ONE         1.f
    #define TWO         2.f
#else
    #define TWOVEC( ... ) make_double2(__VA_ARGS__)
    #define THREEVEC( ... ) make_double3(__VA_ARGS__)
    #define FOURVEC( ... )  make_double4(__VA_ARGS__)
    #define ZERO        0.0
    #define QUARTER     0.25
    #define HALF        0.5
    #define ONE         1.0
    #define TWO         2.0

#endif

const REAL lx = 1.0;
REALfour bd[2];

struct dimensions
{
    REAL gam = 1.4;
    REAL mgam = 0.4;
    REAL dt_dx;
    unsigned int base;
    unsigned int idxend;
    unsigned int idxend_1;
    unsigned int hts[5];
};

//dbd is the boundary condition in device constant memory.
__constant__ REALfour dbd[2]; //0 is left 1 is right.
//dimens dimension struct in global memory.
__constant__ dimensions dimens;

__host__ __device__
__forceinline__
void
readIn(REALthree *temp, const REALthree *rights, const REALthree *lefts, int td, int gd)
{

	int leftidx = dimens.hts[4] + (((td>>2) & 1) * dimens.base) + (td & 3) - (4 + ((td>>2)<<1));
	int rightidx = dimens.hts[4] + (((td>>2) & 1) * dimens.base) + ((td>>2)<<1) + (td & 3);

	temp[leftidx] = rights[gd];
	temp[rightidx] = lefts[gd];

}

__host__ __device__
__forceinline__
void
writeOut(REALthree *temp, REALthree *rights, REALthree *lefts, int td, int gd)
{

    int leftidx = (((td>>2) & 1)  * dimens.base) + ((td>>2)<<1) + (td & 3) + 2; //left get
    int rightidx = (dimens.base-6) + (((td>>2) & 1)  * dimens.base) + (td & 3) - ((td>>2)<<1); //right get

	rights[gd] = temp[rightidx];
	lefts[gd] = temp[leftidx];

}

//Calculates the pressure at the current node with the rho, u, e state variables.
__device__ __host__
__forceinline__
REAL
pressure(REALfour current)
{
    #ifdef __CUDA_ARCH__
    return dimens.mgam * (current.z - (0.5 * current.y * current.y/current.x));
    #else
    return dimz.mgam * (current.z - (0.5 * current.y * current.y/current.x));
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
    spectreRadius = sqrt(dimens.gam * halfState.w/halfState.x) + fabs(halfState.y);
    #else
    spectreRadius = sqrt(dimz.gam * halfState.w/halfState.x) + fabs(halfState.y);
    #endif

    flux += 0.5 * spectreRadius * (THREEVEC(cvLeft) - THREEVEC(cvRight));

    return flux;
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
    tempStateRight = state[tr[2]] + FOURVEC(0.5 * dimens.dt_dx * (fluxL-fluxR));
    #else
    tempStateRight = state[tr[2]] + FOURVEC(0.5 * dimz.dt_dx * (fluxL-fluxR));
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
    return FOURVEC(dimens.dt_dx * (fluxL-fluxR));
    #else
    return FOURVEC(dimz.dt_dx * (fluxL-fluxR));
    #endif

}

__global__
void
swapKernel(const REALfour *passing_side, REALfour *bin, int direction)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
    int gidout = (gid + direction*blockDim.x) & dimens.idxend;

    bin[gidout] = passing_side[gid];

}

//Simple scheme with dirchlet boundary condition.
__global__
void
classicEuler(REALfour *euler_in, REALfour *euler_out, bool final)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID

    const char4 truth = {gid == 0, gid == 1, gid == dimens.idxend_1, gid == dimens.idxend};
    int gids[5];

    #pragma unroll
    for (k = -2; k<3; k++) gids[k+2] = k+gid;

    if (truth.x)
    {
        euler_out[gid] = dbd[0];
        return;
    }
    else if (truth.w)
    {
        euler_out[gid] = dbd[1];
        return;
    }

    if (truth.y) gids[0] = gids[1] ;
    if (truth.z) gids[4] = gids[3];

    if (final)
    {
        euler_out[gid] += eulerFinalStep(euler_in, gids;
    }
    else
    {
        euler_out[gid] = eulerStutterStep(euler_in, gids);
    }

}

__global__
void
upTriangle(const REALfour *IC, REALfour *right, REALfour *left)
{

	extern __shared__ REALfour temper[];

	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	int tid = threadIdx.x; //Block Thread ID
    int tididx = tid + 2;
    int step2;

    int tid_top[5], tid_bottom[5];
	#pragma unroll
	for (int k = -2; k<3; k++)
    {
        tid_bottom[k+2] = tididx + k;
        tid_top[k+2] = tididx + k + blockDim.x;
    }

    //Assign the initial values to the first row in temper, each block
    //has it's own version of temper shared among its threads.
	temper[tididx] = IC[gid];

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
    writeOut(temper, right, left, tid, gid);

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
    int step2;

    int tid_top[5], tid_bottom[5];
    #pragma unroll
    for (int k = -2; k<3; k++)
    {
        tid_bottom[k+2] = tididx + k;
        tid_top[k+2] = tididx + k + dimens.base;
    }

    readIn(temper, right, left, tid, gid);

    __syncthreads();

    if (gid == 0)
    {
        temper[tididx] = dbd[0];
        temper[tididx+dimens.base] = dbd[0];
        IC[gid] = temper[tididx];
        return;
    }
    else if (gid == dimens.idxend)
    {
        temper[tididx] = dbd[1];
        temper[tididx+dimens.base] = dbd[1];
        IC[gid] = temper[tididx];
        return;
    }
    else if (gid == dimens.idxend_1)
    {
        tid_top[4] = tid_top[3];
    }
    else if (gid == 1)
    {
        tid_top[0] = tid_top[1];
    }

    __syncthreads();

	for (int k = dimens.hts[2]; k>1; k-=4)
	{
		if (tididx < (dimens.base-k) && tididx >= k)
		{
            temper[tid_top[2]] = eulerStutterStep(temper, tid_bottom);
		}

        step2 = k-2;
        __syncthreads();

        if (tididx < (dimens.base-step2) && tididx >= step2)
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
wholeDiamond(REALfour *right, REALfour *left, const bool full)
{

    extern __shared__ REALfour temper[];

	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	int tididx = tid + 2;
    int step2;

    int tid_top[5], tid_bottom[5];
    #pragma unroll
    for (int k = -2; k<3; k++)
    {
        tid_bottom[k+2] = tididx + k;
        tid_top[k+2] = tididx + k + dimens.base;
    }

    if (!full)
    {
        gid += blockDim.x;
    }

    readIn(temper, right, left, tid, gid);

    __syncthreads();

    if (full)
    {
        if (gid == 0)
        {
            temper[tididx] = dbd[0];
            temper[tididx+dimens.base] = dbd[0];
        }
        else if (gid == dimens.idxend)
        {
            temper[tididx] = dbd[1];
            temper[tididx+dimens.base] = dbd[1];
        }
        else if (gid == dimens.idxend_1)
        {
            tid_top[4] = tid_top[3];
        }
        else if (gid == 1)
        {
            tid_top[0] = tid_top[1];
        }
    }
    __syncthreads();

    if (tididx < (dimens.base-dimens.hts[2]) && tididx >= dimens.hts[2])
    {
        temper[tid_top[2]] = eulerStutterStep(temper, tid_bottom);
    }

    __syncthreads();

    for (int k = dimens.hts[0]; k>4; k-=4)
    {
        if (tididx < (dimens.base-k) && tididx >= k)
        {
            temper[tididx] += eulerFinalStep(temper, tid_top);

            temper[tididx].w = pressure(temper[tididx]);
        }

        step2 = k-2;
        __syncthreads();

        if (tididx < (dimens.base-step2) && tididx >= step2)
        {
            temper[tid_top[2]] = eulerStutterStep(temper, tid_bottom);
        }
        //Make sure the threads are synced
        __syncthreads();
    }

    // -------------------TOP PART------------------------------------------

    if (full)
        if (gid > 0 &&  gid < dimens.idxend)
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

    if (tididx > 3 && tididx <(dimens.base-4))
	{
        temper[tid_top[2]] = eulerStutterStep(temper, tid_bottom);
	}
	//The initial conditions are timeslice 0 so start k at 1.

	__syncthreads();

    //The initial conditions are timslice 0 so start k at 1.
	for (int k = 6; k<dimens.hts[4]; k+=4)
	{
		if (tididx< (dimens.base-k) && tididx >= k)
		{
            temper[tididx] += eulerFinalStep(temper, tid_top);

            temper[tididx].w = pressure(temper[tididx]);
		}
        step2 = k+2;
        __syncthreads();

        if (tididx < (dimens.base-step2) && tididx >= step2)
        {
            temper[tid_top[2]] = eulerStutterStep(temper, tid_bottom);
		}

		//Make sure the threads are synced
		__syncthreads();

	}

    writeOut(temper, right, left, tid, gid);

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
    int step2;

    int tid_top[5], tid_bottom[5];
    #pragma unroll
    for (int k = -2; k<3; k++)
    {
        tid_bottom[k+2] = tididx + k;
        tid_top[k+2] = tididx + k + dimens.base;
    }

    bool t_ht = (gid == dimens.hts[2]);
    bool t_htm = (gid == dimens.hts[1]);

    readIn(temper, right, left, tid, gid);

    __syncthreads();

    if (t_ht)
    {
        temper[tididx] = dbd[0];
        temper[tididx+dimens.base] = dbd[0];
    }
    else if (t_htm)
    {
        temper[tididx] = dbd[1];
        temper[tididx+dimens.base] = dbd[1];
    }
    else if (gid == dimens.hts[0])
    {
        tid_top[4] = tid_top[3];
        tid_bottom[4] = tid_bottom[3];
    }
    else if (gid == dimens.hts[3])
    {
        tid_top[0] = tid_top[1];
        tid_bottom[0] = tid_bottom[1];
    }

    __syncthreads();

    for (int k = dimens.hts[2]; k>0; k-=4)
    {

        if (!t_ht && !t_htm && tididx < (dimens.base-k) && tididx >= k)
        {
            temper[tid_top[2]] = eulerStutterStep(temper, tid_bottom);
        }

        step2 = k-2;
        __syncthreads();

        if (!t_ht && !t_htm && tididx < (dimens.base-step2) && tididx >= step2)
        {
            temper[tididx] += eulerFinalStep(temper, tid_top);

            temper[tididx].w = pressure(temper[tididx]);
        }

        __syncthreads();
    }

    if (!t_ht && !t_htm && tid > 1 && tid <(blockDim.x-2))
	{
        temper[tid_top[2]] = eulerStutterStep(temper, tid_bottom);
	}

	__syncthreads();


    //The initial conditions are timslice 0 so start k at 1.
    for (int k = 4; k<dimens.hts[2]; k+=4)
    {
        if (!t_ht && !t_htm && tid < (blockDim.x-k) && tid >= k)
        {
            temper[tididx] += eulerFinalStep(temper, tid_top);

            temper[tididx].w = pressure(temper[tididx]);
        }

        step2 = k+2;
        __syncthreads();

        if (!t_ht && !t_htm && tid < (blockDim.x-step2) && tid >= step2)
        {
            temper[tid_top[2]] = eulerStutterStep(temper, tid_bottom);
        }

        //Make sure the threads are synced
        __syncthreads();

    }

    writeOut(temper, right, left, tid, gid);
}

using namespace std;

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
            for (int k = 1; k<(dv-1); k++) fwr << T_f[k].x << " ";
            fwr << endl;

            fwr << " Velocity " << t_eq << " ";
            for (int k = 1; k<(dv-1); k++) fwr << T_f[k].y/T_f[k].x << " ";
            fwr << endl;

            fwr << " Energy " << t_eq << " ";
            for (int k = 1; k<(dv-1); k++) fwr << (T_f[k].z/T_f[k].x) << " ";
            fwr << endl;

            fwr << " Pressure " << t_eq << " ";
            for (int k = 1; k<(dv-1); k++) fwr << T_f[k].w << " ";
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

    const size_t smem = (2*dimz.base)*sizeof(REALfour);

	REALfour *d_IC, *d_right, *d_left;

	cudaMalloc((void **)&d_IC, sizeof(REALfour)*dv);
	cudaMalloc((void **)&d_right, sizeof(REALfour)*dv);
	cudaMalloc((void **)&d_left, sizeof(REALfour)*dv);

	// Copy the initial conditions to the device array.
	cudaMemcpy(d_IC,IC,sizeof(REALfour)*dv,cudaMemcpyHostToDevice);
	// Start the counter and start the clock.
    const double t_fullstep = 0.25*dt*(double)tpb;

    upTriangle <<< bks,tpb,smem >>>(d_IC,d_right,d_left);

    swapKernel <<< bks,tpb >>> (d_right, d_IC, 1);
    swapKernel <<< bks,tpb >>> (d_IC, d_right, 0);

    double t_eq;
    double twrite = freq;

    // Call the kernels until you reach the iteration limit.

    splitDiamond <<< bks,tpb,smem >>>(d_right,d_left);
    t_eq = t_fullstep;
    swapKernel <<< bks,tpb >>> (d_left, d_IC, -1);
    swapKernel <<< bks,tpb >>> (d_IC, d_left, 0);

    while(t_eq < t_end)
    {
        wholeDiamond <<< bks,tpb,smem >>>(d_right, d_left, true);

        swapKernel <<< bks,tpb >>> (d_right, d_IC, 1);
        swapKernel <<< bks,tpb >>> (d_IC, d_right, 0);

        splitDiamond <<< bks,tpb,smem >>>(d_right,d_left);

        swapKernel <<< bks,tpb >>> (d_left, d_IC, -1);
        swapKernel <<< bks,tpb >>> (d_IC, d_left, 0);

        t_eq += t_fullstep;

        if (t_eq > twrite)
        {
            downTriangle <<< bks,tpb,smem >>>(d_IC,d_right,d_left);

            cudaMemcpy(T_f, d_IC, sizeof(REALthree)*dv, cudaMemcpyDeviceToHost);

            fwr << " Density " << t_eq << " ";
            for (int k = 1; k<(dv-1); k++) fwr << T_f[k].x << " ";
            fwr << endl;

            fwr << " Velocity " << t_eq << " ";
            for (int k = 1; k<(dv-1); k++) fwr << (T_f[k].y/T_f[k].x) << " ";
            fwr << endl;

            fwr << " Energy " << t_eq << " ";
            for (int k = 1; k<(dv-1); k++) fwr << (T_f[k].z/T_f[k].x) << " ";
            fwr << endl;

            fwr << " Pressure " << t_eq << " ";
            for (int k = 1; k<(dv-1); k++) fwr << pressure(T_f[k]) << " ";
            fwr << endl;

            upTriangle <<< bks,tpb,smem >>>(d_IC,d_right,d_left);

            swapKernel <<< bks,tpb >>> (d_right, d_IC, 1);
            swapKernel <<< bks,tpb >>> (d_IC, d_right, 0);

            splitDiamond <<< bks,tpb,smem >>>(d_right,d_left);

            swapKernel <<< bks,tpb >>> (d_left, d_IC, -1);
            swapKernel <<< bks,tpb >>> (d_IC, d_left, 0);

            t_eq += t_fullstep;

            twrite += freq;
        }
    }


    downTriangle <<< bks,tpb,smem >>>(d_IC,d_right,d_left);

    cudaMemcpy(T_f, d_IC, sizeof(REALthree)*dv, cudaMemcpyDeviceToHost);

    cudaFree(d_IC);
    cudaFree(d_right);
    cudaFree(d_left);

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
    const REAL dx = lx/((REAL)dv-2.0);
    char const *prec;
    prec = (sizeof(REAL)<6) ? "Single": "Double";

    //Declare the dimensions in constant memory.
    dimensions dimz;
    dimz.dt_dx = dt/dx; // dt/dx
    dimz.base = tpb+4;
    dimz.idxend = dv-1;
    dimz.idxend_1 = dv-2;

    for (int k=-2; k<2; k++) dimz.hts[k] = (tpb/2) + k;

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
	fwr << lx << " " << (dv-2) << " " << dx << " " << endl;

    fwr << " Density " << 0 << " ";
    for (int k = 1; k<(dv-1); k++) fwr << IC[k].x << " ";
    fwr << endl;

    fwr << " Velocity " << 0 << " ";
    for (int k = 1; k<(dv-1); k++) fwr << IC[k].y << " ";
    fwr << endl;

    fwr << " Energy " << 0 << " ";
    for (int k = 1; k<(dv-1); k++) fwr << IC[k].z/IC[k].x << " ";
    fwr << endl;

    fwr << " Pressure " << 0 << " ";
    for (int k = 1; k<(dv-1); k++) fwr << IC[k].w << " ";
    fwr << endl;

    //Transfer data to GPU.
	// This puts the Fourier number in constant memory.
	cudaMemcpyToSymbol(dimens,&dimz,sizeof(dimensions));
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
	for (int k = 1; k<(dv-1); k++) fwr << T_final[k].x << " ";
    fwr << endl;

    fwr << " Velocity " << tfm << " ";
	for (int k = 1; k<(dv-1); k++) fwr << T_final[k].y/T_final[k].x << " ";
    fwr << endl;

    fwr << " Energy " << tfm << " ";
    for (int k = 1; k<(dv-1); k++) fwr << T_final[k].z/T_final[k].x << " ";
    fwr << endl;

    fwr << " Pressure " << tfm << " ";
    for (int k = 1; k<(dv-1); k++) fwr << T_final[k].w << " ";
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
