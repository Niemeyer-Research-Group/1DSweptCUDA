//


//COMPILE LINE:
// nvcc -o ./bin/EulerOut Euler1D_SweptShared.cu -gencode arch=compute_35,code=sm_35 -lm -w -std=c++11 -Xcompiler -fopenmp

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <helper_math.h>

#include <ostream>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <omp.h>

#define REAL        float
#define REALfour    float4
#define REALthree   float3

const REAL gam = 1.4;
const REAL m_gamma = 0.4;
const REAL dx = 0.5;

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
REALfour
limitor(REALthree cvCurrent, REALthree cvOther, REAL pRatio)
{
    #ifdef __CUDA_ARCH__
    if (isfinite(pRatio) && pRatio > 0) //If it's finite and positive
    {
        REAL fact = (pRatio < 1) ? pRatio : 1.f;
        return make_float4(cvCurrent + 0.5* fact * (cvOther - cvCurrent));

    }
    #else
    if (std::isfinite(pRatio) && pRatio > 0) //If it's finite and positive
    {
        REAL fact = (pRatio < 1) ? pRatio : 1.f;
        return make_float4(cvCurrent + 0.5* fact * (cvOther - cvCurrent));

    }
    #endif

    return make_float4(cvCurrent);

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
    REAL rhoLeftsqrt = sqrtf(cvLeft.x); REAL rhoRightsqrt = sqrtf(cvRight.x);
    halfState.x = rhoLeftsqrt * rhoRightsqrt;
    halfState.y = (rhoLeftsqrt*uLeft + rhoRightsqrt*uRight)/(rhoLeftsqrt+rhoRightsqrt);
    halfState.z = (rhoLeftsqrt*eLeft + rhoRightsqrt*eRight)/(rhoLeftsqrt+rhoRightsqrt); //Seems to be unnecessary.
    halfState.w = pressure(halfState);

    #ifdef __CUDA_ARCH__
    spectreRadius = sqrtf(dimens.y * halfState.w/halfState.x) + fabs(halfState.y);
    #else
    spectreRadius = sqrtf(dimz.y * halfState.w/halfState.x) + fabs(halfState.y);
    #endif

    flux += 0.5 * spectreRadius * (make_float3(cvLeft) - make_float3(cvRight));

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
    pR = make_float3(pressureRatio(pfarLeft,stateLeft.w,stateCenter.w),
        pressureRatio(stateLeft.w,stateCenter.w,stateRight.w),
        pressureRatio(stateCenter.w,stateRight.w,pfarRight));

    //This is the temporary state bounded by the limitor function.
    tempStateLeft = limitor(make_float3(stateLeft), make_float3(stateCenter), pR.x);
    tempStateRight = limitor(make_float3(stateCenter), make_float3(stateLeft), 1.0/pR.y);

    //Pressure needs to be recalculated for the new limited state variables.
    tempStateLeft.w = pressure(tempStateLeft);
    tempStateRight.w = pressure(tempStateRight);
    fluxL = eulerFlux(tempStateLeft,tempStateRight);

    //Do the same thing with the right side.
    tempStateLeft = limitor(make_float3(stateCenter), make_float3(stateRight), pR.y);
    tempStateRight = limitor(make_float3(stateRight), make_float3(stateCenter), 1.0/pR.z);
    tempStateLeft.w = pressure(tempStateLeft);
    tempStateRight.w = pressure(tempStateRight);
    fluxR = eulerFlux(tempStateLeft,tempStateRight);

    //Add the change back to the node in question.
    #ifdef __CUDA_ARCH__
    stateCenter += make_float4(0.5 * dimens.x * (fluxL-fluxR));
    #else
    stateCenter += make_float4(0.5 * dimz.x * (fluxL-fluxR));
    #endif
    stateCenter.w = pressure(stateCenter);

    return stateCenter;
}

//Same thing as the predictor step, but this final step adds the result to the original state variables to advance to the next timestep.
//But the predictor variables to find the fluxes.
__device__ __host__
REALfour
eulerFinalStep(REAL pfarLeft, REALfour stateLeft, REALfour stateCenter, REALfour stateCenter_orig, REALfour stateRight, REAL pfarRight)
{
    REALthree fluxL, fluxR, pR;
    REALfour tempStateLeft, tempStateRight;

    pR = make_float3(pressureRatio(pfarLeft,stateLeft.w,stateCenter.w),
        pressureRatio(stateLeft.w,stateCenter.w,stateRight.w),
        pressureRatio(stateCenter.w,stateRight.w,pfarRight));

    tempStateLeft = limitor(make_float3(stateLeft), make_float3(stateCenter), pR.x);
    tempStateRight = limitor(make_float3(stateCenter), make_float3(stateLeft), 1.0/pR.y);
    tempStateLeft.w = pressure(tempStateLeft);
    tempStateRight.w = pressure(tempStateRight);
    fluxL = eulerFlux(tempStateLeft,tempStateRight);

    tempStateLeft = limitor(make_float3(stateCenter), make_float3(stateRight), pR.y);
    tempStateRight = limitor(make_float3(stateRight), make_float3(stateCenter), 1.0/pR.z);
    tempStateLeft.w = pressure(tempStateLeft);
    tempStateRight.w = pressure(tempStateRight);
    fluxR = eulerFlux(tempStateLeft,tempStateRight);

    #ifdef __CUDA_ARCH__
    stateCenter_orig += make_float4(0.5 * dimens.x * (fluxL-fluxR));
    #else
    stateCenter_orig += make_float4(0.5 * dimz.x * (fluxL-fluxR));
    #endif
    stateCenter_orig.w = pressure(stateCenter_orig);

    return stateCenter_orig;
}

//Simple scheme with dirchlet boundary condition.
__global__
void
classicEuler(REALfour *euler_in, REALfour *euler_out)
{

    int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
    int lastidx = ((blockDim.x*gridDim.x)-1);

    REALfour temp[5];
    REALfour persist = euler_in[gid];

    #pragma unroll
	for (int k = -2; k<3; k++)
	{
		temp[k+2] = euler_in[gid+k];
	}

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
        euler_out[gid] = eulerStutterStep(dbd[0].w,dbd[0],temp[2],temp[3],temp[4].w);
    }
    else if (gid == (lastidx-1))
    {
        euler_out[gid] = eulerStutterStep(temp[0].w,temp[1],temp[2],dbd[1],dbd[1].w);
    }
    else
    {
        euler_out[gid] = eulerStutterStep(temp[0].w,temp[1],temp[2],temp[3],temp[4].w);
    }

    __syncthreads();

    #pragma unroll
    for (int k = -2; k<3; k++)
    {
        temp[k+2] = euler_out[gid+k];
    }

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
        euler_out[gid] = eulerFinalStep(dbd[0].w,dbd[0],temp[2],persist,temp[3],temp[4].w);
    }
    else if (gid == (lastidx-1))
    {
        euler_out[gid] = eulerFinalStep(temp[0].w,temp[1],temp[2],persist,dbd[1],dbd[1].w);
    }
    else
    {
        euler_out[gid] = eulerFinalStep(temp[0].w,temp[1],temp[2],persist,temp[3],temp[4].w);
    }

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
    int height2 = height-2;
    int lastidx = ((blockDim.x*gridDim.x)-1);
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
	int gidin = (gid + blockDim.x) & lastidx;

	temper[leftidx] = right[gid];
	temper[rightidx] = left[gidin];

    if (gid == 0)
    {
        temper[base] = dbd[0];
        temper[base+1] = dbd[0];
    }
    if (gid == lastidx)
    {
        temper[2*base-1] = dbd[1];
        temper[2*base-2] = dbd[1];
    }


    if (tididx < (base-height2) && tididx >= height2)
    {
        temper[tid_top[2]] = eulerStutterStep(temper[tid_bottom[0]].w, temper[tid_bottom[1]], temper[tid_bottom[2]],
            temper[tid_bottom[3]], temper[tid_bottom[4]].w);
    }

    __syncthreads();

	for (int k = (height-4); k>4; k-=4)
	{

		if (tididx < (base-k) && tididx >= k)
		{
			temper[tididx] = eulerFinalStep(temper[tid_top[0]].w, temper[tid_top[1]], temper[tid_top[2]],
				temper[tididx], temper[tid_top[3]], temper[tid_top[4]].w);
		}

        step2 = k-2;

        if (tididx < (base-step2) && tididx >= step2)
        {
            temper[tid_top[2]] = eulerStutterStep(temper[tid_bottom[0]].w, temper[tid_bottom[1]], temper[tid_bottom[2]],
                temper[tid_bottom[3]], temper[tid_bottom[4]].w);

        }
		//Make sure the threads are synced
		__syncthreads();
	}

    if (gid == 0)
    {
        temper[tididx] = dbd[0];
    }
    else if (gid == lastidx)
    {
        temper[tididx] = dbd[1];
    }
    else
    {
        temper[tididx] = eulerFinalStep(temper[tid_top[0]].w, temper[tid_top[1]], temper[tid_top[2]],
            temper[tididx], temper[tid_top[3]], temper[tid_top[4]].w);
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
    int height2 = height-2;
	int step2;
    int lastidx = ((blockDim.x*gridDim.x)-1);

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
        temper[leftidx] = right[gid];
        gid += blockDim.x;
        temper[rightidx] = left[gid];
    }

    if (gid < 2)
    {
        temper[base+gid] = dbd[0];
    }
    if (gid == lastidx)
    {
        temper[2*base-1] = dbd[1];
        temper[2*base-2] = dbd[1];
    }


    if (tididx < (base-height2) && tididx >= height2)
    {
        temper[tid_top[2]] = eulerStutterStep(temper[tid_bottom[0]].w, temper[tid_bottom[1]], temper[tid_bottom[2]],
            temper[tid_bottom[3]], temper[tid_bottom[4]].w);
    }

    __syncthreads();

    for (int k = (height-4); k>4; k-=4)
    {
        if (tididx < (base-k) && tididx >= k)
        {
            temper[tididx] = eulerFinalStep(temper[tid_top[0]].w, temper[tid_top[1]], temper[tid_top[2]],
                temper[tididx], temper[tid_top[3]], temper[tid_top[4]].w);
        }

        step2 = k-2;

        if (tididx < (base-step2) && tididx >= step2)
        {
            temper[tid_top[2]] = eulerStutterStep(temper[tid_bottom[0]].w, temper[tid_bottom[1]], temper[tid_bottom[2]],
                temper[tid_bottom[3]], temper[tid_bottom[4]].w);

        }
        //Make sure the threads are synced
        __syncthreads();
    }

    if (gid == 0)
    {
        temper[tididx] = dbd[0];
    }
    else if (gid == lastidx)
    {
        temper[tididx] = dbd[1];
    }
    else
    {
        temper[tididx] = eulerFinalStep(temper[tid_top[0]].w, temper[tid_top[1]], temper[tid_top[2]],
            temper[tididx], temper[tid_top[3]], temper[tid_top[4]].w);
    }

    temper[tid] = temper[tididx];

    // -------------------TOP PART------------------------------------------

    leftidx = ((tid/4 & 1) * blockDim.x) + (tid/4)*2 + (tid & 3);
	rightidx = (blockDim.x - 4) + ((tid/4 & 1) * blockDim.x) + (tid & 3) - (tid/4)*2;

    #pragma unroll
    for (int k = -2; k<3; k++)
    {
        tid_top[k+2] = tid + k + blockDim.x;
        tid_bottom[k+2] = tid + k;
    }

    if (tid > 1 && tid <(blockDim.x-2))
	{
		temper[tid_top[2]] = eulerStutterStep(temper[tid_bottom[0]].w, temper[tid_bottom[1]], temper[tid_bottom[2]],
			temper[tid_bottom[3]], temper[tid_bottom[4]].w);
	}
	//The initial conditions are timeslice 0 so start k at 1.

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

    right[gid] = temper[rightidx];
	left[gid] = temper[leftidx];

}

//Split one is always first.  Passing left like the downTriangle.  downTriangle
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
    int height2 = height-2;
	int step2;
    int lastidx = ((blockDim.x*gridDim.x)-1);

	int tid_top[5], tid_bottom[5];
	#pragma unroll
	for (int k = -2; k<3; k++)
	{
		tid_top[k+2] = tididx + k + base;
		tid_bottom[k+2] = tididx + k;
	}

	int leftidx = height + ((tid/4 & 1) * base) + (tid & 3) - (4 + (tid/4) * 2);
	int rightidx = height + ((tid/4 & 1) * base) + (tid/4)*2 + (tid & 3);
    int gidin = (gid - blockDim.x) & lastidx;
	// Initialize temper. Kind of an unrolled for loop.  This is actually at
	// Timestep 0.

    temper[leftidx] = right[gidin];
	temper[rightidx] = left[gid];

    for (int k = height2; k>0; k-=4)
    {
        if (blockIdx.x > 0)
        {
            if (tididx < (base-k) && tididx >= k)
            {
                temper[tid_top[2]] = eulerStutterStep(temper[tid_bottom[0]].w, temper[tid_bottom[1]], temper[tid_bottom[2]],
                    temper[tid_bottom[3]], temper[tid_bottom[4]].w);
            }
        }
        else
        {
            if (tididx < (base-k) && tididx >= k)
            {
                if (tididx == (height-1)) //case 1
                {
                    temper[tididx+base] = dbd[1];
                }
                else if (tididx == height)  //case 2
                {
                    temper[tididx+base] = dbd[0];
                }
                else if (tididx == height2) //case 0
                {
                    temper[tididx+base] = eulerStutterStep(temper[tid_bottom[0]].w, temper[tid_bottom[1]], temper[tid_bottom[2]],
                        dbd[1], dbd[1].w);
                }
                else if (tididx == (height+1)) //case 3
                {
                    temper[tididx+base] = eulerStutterStep(dbd[0].w, dbd[0], temper[tid_bottom[2]],
                        temper[tid_bottom[3]], temper[tid_bottom[4]].w);
                }
                else
                {
                    temper[tid_top[2]] = eulerStutterStep(temper[tid_bottom[0]].w, temper[tid_bottom[1]], temper[tid_bottom[2]],
                        temper[tid_bottom[3]], temper[tid_bottom[4]].w);
                }
            }

        }

        step2 = k-2;

        if (blockIdx.x > 0)
        {
            if (tididx < (base-step2) && tididx >= step2)
            {
                temper[tididx] = eulerFinalStep(temper[tid_top[0]].w, temper[tid_top[1]], temper[tid_top[2]],
                    temper[tididx], temper[tid_top[3]], temper[tid_top[4]].w);
            }
        }
        else
        {
            if (tididx < (base-step2) && tididx >= step2)
            {
                if (tididx == (height-1)) //case 1
                {
                    temper[tididx] = dbd[1];
                }
                else if (tididx == height)  //case 2
                {
                    temper[tididx] = dbd[0];
                }
                else if (tididx == height2) //case 0
                {
                    temper[tididx] = eulerFinalStep(temper[tid_top[0]].w, temper[tid_top[1]], temper[tid_top[2]], temper[tididx],
                        dbd[1], dbd[1].w);
                }
                else if (tididx == (height+1)) //case 3
                {
                    temper[tididx] = eulerFinalStep(dbd[0].w, dbd[0], temper[tid_top[2]],temper[tididx],
                        temper[tid_top[3]], temper[tid_top[4]].w);
                }
                else
                {
                    temper[tididx] = eulerFinalStep(temper[tid_top[0]].w, temper[tid_top[1]], temper[tid_top[2]],
                        temper[tididx], temper[tid_top[3]], temper[tid_top[4]].w);
                }
            }

        }

        __syncthreads();
    }

    //Justify the result at 0 index.
    temper[tid] = temper[tididx];

    leftidx = ((tid/4 & 1) * blockDim.x) + (tid/4)*2 + (tid & 3);
	rightidx = (blockDim.x - 4) + ((tid/4 & 1) * blockDim.x) + (tid & 3) - (tid/4)*2;

    #pragma unroll
    for (int k = -2; k<3; k++)
    {
        tid_top[k+2] = tid + k + blockDim.x;
        tid_bottom[k+2] = tid + k;
    }

    //The initial conditions are timslice 0 so start k at 1.
    height -= 2;
    height2 -= 2;

    //Single step start to top part.
    if (blockIdx.x > 0)
    {
        if (tid > 1 && tid <(blockDim.x-2))
        {
            temper[tid_top[2]] = eulerStutterStep(temper[tid_bottom[0]].w, temper[tid_bottom[1]], temper[tid_bottom[2]],
                temper[tid_bottom[3]], temper[tid_bottom[4]].w);
        }
    }
    else
    {
        if (tid > 1 && tid <(blockDim.x-2))
        {
            if (tid == (height-1)) //case 1
            {
                temper[tid_top[2]] = dbd[1];
            }
            else if (tid == height)  //case 2
            {
                temper[tid_top[2]] = dbd[0];
            }
            else if (tid == height2) //case 0
            {
                temper[tid_top[2]] = eulerStutterStep(temper[tid_bottom[0]].w, temper[tid_bottom[1]], temper[tid_bottom[2]],
                    dbd[1], dbd[1].w);
            }
            else if (tid == (height+1)) //case 3
            {
                temper[tid_top[2]] = eulerStutterStep(dbd[0].w, dbd[0], temper[tid_bottom[2]],
                    temper[tid_bottom[3]], temper[tid_bottom[4]].w);
            }
            else
            {
                temper[tid_top[2]] = eulerStutterStep(temper[tid_bottom[0]].w, temper[tid_bottom[1]], temper[tid_bottom[2]],
                    temper[tid_bottom[3]], temper[tid_bottom[4]].w);
            }
        }

    }

	__syncthreads();

    //The first new row is complete 2->blk-2 next is 4->blk-4
	for (int k = 4; k<height; k+=4)
	{
        if (blockIdx.x > 0)
        {
            if (tid < (blockDim.x-k) && tid >= k)
            {
                temper[tid] = eulerFinalStep(temper[tid_top[0]].w, temper[tid_top[1]], temper[tid_top[2]],
                    temper[tid], temper[tid_top[3]], temper[tid_top[4]].w);
            }
        }
        else
        {
            if (tid < (blockDim.x-k) && tid >= k)
            {
                if (tid == (height-1)) //case 1
                {
                    temper[tid] = dbd[1];
                }
                else if (tid == height)  //case 2
                {
                    temper[tid] = dbd[0];
                }
                else if (tid == height2) //case 0
                {
                    temper[tid] = eulerFinalStep(temper[tid_top[0]].w, temper[tid_top[1]], temper[tid_top[2]],temper[tid],
                        dbd[1], dbd[1].w);
                }
                else if (tid == (height+1)) //case 3
                {
                    temper[tid] = eulerFinalStep(dbd[0].w, dbd[0], temper[tid_top[2]],temper[tid],
                        temper[tid_top[3]], temper[tid_top[4]].w);
                }
                else
                {
                    temper[tid] = eulerFinalStep(temper[tid_top[0]].w, temper[tid_top[1]], temper[tid_top[2]],
                        temper[tid], temper[tid_top[3]], temper[tid_top[4]].w);
                }
            }

        }
		step2 = k + 2;
		__syncthreads();

        if (blockIdx.x > 0)
        {
            if (tid < (blockDim.x-step2) && tid >= step2)
            {
                temper[tid_top[2]] = eulerStutterStep(temper[tid_bottom[0]].w, temper[tid_bottom[1]], temper[tid_bottom[2]],
                    temper[tid_bottom[3]], temper[tid_bottom[4]].w);
            }
        }
        else
        {
            if (tid < (blockDim.x-step2) && tid >= step2)
            {
                if (tid == (height-1)) //case 1
                {
                    temper[tid_top[2]] = dbd[1];
                }
                else if (tid == height)  //case 2
                {
                    temper[tid_top[2]] = dbd[0];
                }
                else if (tid == height2) //case 0
                {
                    temper[tid_top[2]] = eulerStutterStep(temper[tid_bottom[0]].w, temper[tid_bottom[1]], temper[tid_bottom[2]],
                        dbd[1], dbd[1].w);
                }
                else if (tid == (height+1)) //case 3
                {
                    temper[tid_top[2]] = eulerStutterStep(dbd[0].w, dbd[0], temper[tid_bottom[2]],
                        temper[tid_bottom[3]], temper[tid_bottom[4]].w);
                }
                else
                {
                    temper[tid_top[2]] = eulerStutterStep(temper[tid_bottom[0]].w, temper[tid_bottom[1]], temper[tid_bottom[2]],
                        temper[tid_bottom[3]], temper[tid_bottom[4]].w);
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

using namespace std;
// Do the split diamond on the CPU?
// What's the idea?  Say malloc the pointers and pass values in the wrapper.

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
                temper[n+base] = eulerStutterStep(temper[n-2].w, temper[n-1], temper[n],
                    bd[1], bd[1].w);
            }
            else if (n == (height+1)) //case 3
            {
                temper[n+base] = eulerStutterStep(bd[0].w, bd[0], temper[n],
                    temper[n+1], temper[n+2].w);
            }
            else
            {
                temper[n+base] = eulerStutterStep(temper[n-2].w, temper[n-1], temper[n],
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
                temper[n] = eulerFinalStep(temper[base+n-2].w, temper[base+n-1], temper[base+n], temper[n],
                    bd[1], bd[1].w);
            }
            else if (n == (height+1)) //case 3
            {
                temper[n] = eulerFinalStep(bd[0].w, bd[0], temper[base+n], temper[n],
                    temper[base+n+1], temper[base+n+2].w);
            }
            else
            {
                temper[n] = eulerFinalStep(temper[base+n-2].w, temper[base+n-1], temper[base+n], temper[n],
                    temper[base+n+1], temper[base+n+2].w);
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
            temper[n+tpb] = eulerStutterStep(temper[n-2].w, temper[n-1], temper[n],
                bd[1], bd[1].w);
        }
        else if (n == (height+1)) //case 3
        {
            temper[n+tpb] = eulerStutterStep(bd[0].w, bd[0], temper[n],
                temper[n+1], temper[n+2].w);
        }
        else
        {
            temper[n+tpb] = eulerStutterStep(temper[n-2].w, temper[n-1], temper[n],
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
                temper[n] = eulerFinalStep(temper[tpb+n-2].w, temper[tpb+n-1], temper[tpb+n], temper[n],
                    bd[1], bd[1].w);
            }
            else if (n == (height+1)) //case 3
            {
                temper[n] = eulerFinalStep(bd[0].w, bd[0], temper[tpb+n], temper[n],
                    temper[tpb+n+1], temper[tpb+n+2].w);
            }
            else
            {
                temper[n] = eulerFinalStep(temper[tpb+n-2].w, temper[tpb+n-1], temper[tpb+n], temper[n],
                    temper[tpb+n+1], temper[tpb+n+2].w);
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
                temper[n+tpb] = eulerStutterStep(temper[n-2].w, temper[n-1], temper[n],
                    bd[1], bd[1].w);
            }
            else if (n == (height+1)) //case 3
            {
                temper[n+tpb] = eulerStutterStep(bd[0].w, bd[0], temper[n],
                    temper[n+1], temper[n+2].w);
            }
            else
            {
                temper[n+tpb] = eulerStutterStep(temper[n-2].w, temper[n-1], temper[n],
                    temper[n+1], temper[n+2].w);
            }
        }
    }
}

//Classic Discretization wrapper.
double
classicWrapper(const int bks, int tpb, const int dv, const REAL dt, const int t_end,
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
        classicEuler <<< bks,tpb >>> (dEuler_in, dEuler_out);
        classicEuler <<< bks,tpb >>> (dEuler_out, dEuler_in);
        t_eq += 2*dt;

        if (t_eq > twrite)
        {
            cudaMemcpy(T_f, dEuler_in, sizeof(REALfour)*dv, cudaMemcpyDeviceToHost);
            fwr << t_eq << " ";

            for (int k = 0; k<dv; k++)
            {
                fwr << T_f[k].x << " ";
            }
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
sweptWrapper(const int bks, int tpb, const int dv, REAL dt, const int t_end, const int cpu,
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

    REALfour *tmpr = (REALfour*)malloc(smem2);
	REALfour *d_IC, *d_right, *d_left;
    REALfour *right, *left;
    cudaHostAlloc((void **) &right, tpb*sizeof(REALfour), cudaHostAllocDefault);
    cudaHostAlloc((void **) &left, tpb*sizeof(REALfour), cudaHostAllocDefault);

	cudaMalloc((void **)&d_IC, sizeof(REALfour)*dv);
	cudaMalloc((void **)&d_right, sizeof(REALfour)*dv);
	cudaMalloc((void **)&d_left, sizeof(REALfour)*dv);

	// Copy the initial conditions to the device array.
	cudaMemcpy(d_IC,IC,sizeof(REALfour)*dv,cudaMemcpyHostToDevice);
	// Start the counter and start the clock.
	const double t_fullstep = 0.25*dt*(double)tpb;

	upTriangle <<< bks,tpb,smem1 >>>(d_IC,d_right,d_left);

    double t_eq;
    double twrite = freq;

	// Call the kernels until you reach the iteration limit.

    if (cpu)
    {
        t_eq = t_fullstep;
        omp_set_num_threads( 2 );

        cudaMemcpy(right,d_left, tpb*sizeof(REALfour), cudaMemcpyDeviceToHost);
        cudaMemcpy(left, d_right+(dv-tpb) , tpb*sizeof(REALfour), cudaMemcpyDeviceToHost);

        #pragma omp parallel sections
        {
        #pragma omp section
        {
            for (int k = 0; k<tpb; k++)
            {
                tmpr[indices[0][k]] = left[k];
                tmpr[indices[1][k]] = right[k];
            }

            CPU_diamond(tmpr, tpb);

            for (int k = 0; k<tpb; k++)
            {
                left[k] = tmpr[indices[2][k]];
                right[k] = tmpr[indices[3][k]];
            }
        }
        #pragma omp section
        {
            wholeDiamond <<< bks-1,tpb,smem2 >>>(d_right,d_left,false);
        }
        }

        cudaMemcpy(d_right, right, tpb*sizeof(REALfour), cudaMemcpyHostToDevice);
        cudaMemcpy(d_left, left, tpb*sizeof(REALfour), cudaMemcpyHostToDevice);

        while(t_eq < t_end)
        {

            wholeDiamond <<< bks,tpb,smem2 >>>(d_right,d_left,true);

            cudaMemcpy(right,d_left, tpb*sizeof(REALfour), cudaMemcpyDeviceToHost);
            cudaMemcpy(left, d_right+(dv-tpb) , tpb*sizeof(REALfour), cudaMemcpyDeviceToHost);
            #pragma omp parallel sections
            {
            #pragma omp section
            {
                for (int k = 0; k<tpb; k++)
                {
                    tmpr[indices[0][k]] = left[k];
                    tmpr[indices[1][k]] = right[k];
                }
                CPU_diamond(tmpr, tpb);

                for (int k = 0; k<tpb; k++)
                {
                    left[k] = tmpr[indices[2][k]];
                    right[k] = tmpr[indices[3][k]];
                }
            }
            #pragma omp section
            {
                wholeDiamond <<< bks-1,tpb,smem2 >>>(d_right,d_left,false);
            }
            }
            cudaMemcpy(d_right, right, tpb*sizeof(REALfour), cudaMemcpyHostToDevice);
            cudaMemcpy(d_left, left, tpb*sizeof(REALfour), cudaMemcpyHostToDevice);
            t_eq += t_fullstep;

    	    if (t_eq > twrite)
    		{
    			downTriangle <<< bks,tpb,smem2 >>>(d_IC,d_right,d_left);
    			cudaMemcpy(T_f, d_IC, sizeof(REALfour)*dv, cudaMemcpyDeviceToHost);
    			fwr << t_eq << " ";

    			for (int k = 0; k<dv; k++)
    			{
    					fwr << T_f[k].x << " ";
    			}

    			fwr << endl;

    			upTriangle <<< bks,tpb,smem1 >>>(d_IC,d_right,d_left);
    			splitDiamond <<< bks,tpb,smem2 >>>(d_right,d_left);
                twrite += freq;
    		}
        }
	}
    else
    {
        splitDiamond <<< bks,tpb,smem2 >>>(d_right,d_left);
        t_eq = t_fullstep;

        while(t_eq < t_end)
        {

            wholeDiamond <<< bks,tpb,smem2 >>>(d_right,d_left,true);

            splitDiamond <<< bks,tpb,smem2 >>>(d_right,d_left);
            //So it always ends on a left pass since the down triangle is a right pass.

            t_eq += t_fullstep;

            if (t_eq > twrite)
            {
                downTriangle <<< bks,tpb,smem2 >>>(d_IC,d_right,d_left);
                cudaMemcpy(T_f, d_IC, sizeof(REALfour)*dv, cudaMemcpyDeviceToHost);
                fwr << t_eq << " ";

                for (int k = 0; k<dv; k++)
                {
                        fwr << T_f[k].x << " ";
                }

                fwr << endl;

                upTriangle <<< bks,tpb,smem1 >>>(d_IC,d_right,d_left);
                splitDiamond <<< bks,tpb,smem2 >>>(d_right,d_left);
                twrite += freq;
            }


        }
    }

	downTriangle <<< bks,tpb,smem2 >>>(d_IC,d_right,d_left);

	cudaMemcpy(T_f, d_IC, sizeof(REALfour)*dv, cudaMemcpyDeviceToHost);

	cudaFree(d_IC);
	cudaFree(d_right);
	cudaFree(d_left);
    cudaFreeHost(right);
    cudaFreeHost(left);

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
    const int tst = atoi(argv[7]);
    const int bks = dv/tpb; //The number of blocks
    REAL lx = dx*((float)dv-1.f);

    //Declare the dimensions in constant memory.
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
    REALfour *IC, *T_final;
	cudaHostAlloc((void **) &IC, dv*sizeof(REALfour), cudaHostAllocDefault);
	cudaHostAlloc((void **) &T_final, dv*sizeof(REALfour), cudaHostAllocDefault);

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

    double tfm;
    if (scheme)
    {
        tfm = sweptWrapper(bks, tpb, dv, dt, tf, tst, IC, T_final, freq, fwr);
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

    if (argc>8)
    {
        ofstream ftime;
        ftime.open(argv[9],ios::app);
    	ftime << dv << "\t" << tpb << "\t" << per_ts << endl;
    	ftime.close();
    }

	fwr << tfm << " ";
	for (int k = 0; k<dv; k++)
	{
		fwr << T_final[k].x << " ";
	}

    fwr << endl;

	fwr.close();

	// Free the memory and reset the device.

	cudaEventDestroy( start );
	cudaEventDestroy( stop );
    cudaDeviceReset();
    free(IC);
    free(T_final);

	return 0;

}
