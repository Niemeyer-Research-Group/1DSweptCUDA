/* 
Third Euler version committed Oct 18
*/

//COMPILE LINE:
// nvcc -o ./bin/EulerOut Euler1D_SweptShared.cu -gencode arch=compute_35,code=sm_35 -lm -restrict -Xcompiler -fopenmp

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include "myVectorTypes.h"


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
REALthree bd[2];

struct dimensions {
    REAL gam;
    REAL mgam;
    REAL dt_dx;
    int base;
    int idxend;
    int idxend_1;
    int hts[5];
};

dimensions dimz;
//dbd is the boundary condition in device constant memory.
__constant__ REALthree dbd[2]; //0 is left 1 is right.
//dimens dimension struct in global memory.
__constant__ dimensions dimens;

__host__ __device__
__forceinline__
void
readIn(REALthree *temp, const REALthree *rights, const REALthree *lefts, int td, int gd)
{
    #ifdef __CUDA_ARCH__
	int leftidx = dimens.hts[4] + (((td>>2) & 1) * dimens.base) + (td & 3) - (4 + ((td>>2)<<1));
	int rightidx = dimens.hts[4] + (((td>>2) & 1) * dimens.base) + ((td>>2)<<1) + (td & 3);
    #else
    int leftidx = dimz.hts[4] + (((td>>2) & 1) * dimz.base) + (td & 3) - (4 + ((td>>2)<<1));
    int rightidx = dimz.hts[4] + (((td>>2) & 1) * dimz.base) + ((td>>2)<<1) + (td & 3);
    #endif

	temp[leftidx] = rights[gd];
	temp[rightidx] = lefts[gd];
}

__host__ __device__
__forceinline__
void
writeOut(REALthree *temp, REALthree *rights, REALthree *lefts, int td, int gd)
{
    #ifdef __CUDA_ARCH__
    int leftidx = (((td>>2) & 1)  * dimens.base) + ((td>>2)<<1) + (td & 3) + 2; //left get
    int rightidx = (dimens.base-6) + (((td>>2) & 1)  * dimens.base) + (td & 3) - ((td>>2)<<1); //right get
    #else
    int leftidx = (((td>>2) & 1)  * dimz.base) + ((td>>2)<<1) + (td & 3) + 2;
    int rightidx = (dimz.base-6) + (((td>>2) & 1)  * dimz.base) + (td & 3) - ((td>>2)<<1);
    #endif
	rights[gd] = temp[rightidx];
	lefts[gd] = temp[leftidx];

}


//Calculates the pressure at the current node with the rho, u, e state variables.
__device__ __host__
__forceinline__
REAL
pressure(REALthree current)
{
    #ifdef __CUDA_ARCH__
    return dimens.mgam * (current.z - (HALF * current.y * current.y/current.x));
    #else
    return dimz.mgam * (current.z - (HALF * current.y * current.y/current.x));
    #endif
}

//Really P/rho.
__device__ __host__
__forceinline__
REAL
pressureHalf(REALthree current)
{
    #ifdef __CUDA_ARCH__
    return dimens.mgam * (current.z - (HALF * current.y * current.y));
    #else
    return dimz.mgam * (current.z - (HALF * current.y * current.y));
    #endif
}


//Reconstructs the state variables if the pressure ratio is finite and positive.
//I think it's that internal boundary condition.
__device__ __host__
__forceinline__
REALthree
limitor(REALthree cvCurrent, REALthree cvOther, REAL pRatio)
{
    return (cvCurrent + HALF * min(pRatio,ONE) * (cvOther - cvCurrent));
}


//Left and Center then Left and right.
//This is the meat of the flux calculation.  Fields: x is rho, y is u, z is e, w is p.
__device__ __host__
__forceinline__
REALthree
eulerFlux(REALthree cvLeft, REALthree cvRight)
{
    #ifndef __CUDA_ARCH__
    using namespace std;
    #endif
    //For the first calculation rho and p remain the same.
    REAL uLeft = cvLeft.y/cvLeft.x;
    REAL uRight = cvRight.y/cvRight.x;
    REAL eLeft = cvLeft.z/cvLeft.x;
    REAL eRight = cvRight.z/cvRight.x;

    REALthree halfState;
    REAL rhoLeftsqrt = sqrt(cvLeft.x);
    REAL rhoRightsqrt = sqrt(cvRight.x);
    REAL pL = pressure(cvLeft);
    REAL pR = pressure(cvRight);

    halfState.x = rhoLeftsqrt * rhoRightsqrt;
    REAL halfDenom = (rhoLeftsqrt + rhoRightsqrt);

    REALthree flux;
    flux.x = (cvLeft.y + cvRight.y);

    halfState.y =  (rhoLeftsqrt*uLeft + rhoRightsqrt*uRight)/halfDenom;
    halfState.z =  (rhoLeftsqrt*eLeft + rhoRightsqrt*eRight)/halfDenom;

    flux.y = (cvLeft.y*uLeft + cvRight.y*uRight + pL + pR);
    flux.z = (cvLeft.y*eLeft + cvRight.y*eRight + uLeft*pL + uRight*pR);

    REAL pH = pressureHalf(halfState);

    #ifdef __CUDA_ARCH__
    return (flux + (pH*dimens.gam + fabs(halfState.y)) * (cvLeft - cvRight));
    #else
    return (flux + (pH*dimz.gam + fabs(halfState.y)) * (cvLeft - cvRight));
    #endif


}

//This is the predictor step of the finite volume scheme.
__device__ __host__
REALthree
eulerStutterStep(REALthree *state, int tr, char flagLeft, char flagRight)
{

    REAL pLL = (flagLeft) ? ZERO : (TWO * state[tr-1].x * state[tr-2].x * (state[tr-1].z - state[tr-2].z) +
        (state[tr-2].y * state[tr-2].y*  state[tr-1].x - state[tr-1].y * state[tr-1].y * state[tr-2].x)) ;

    REAL pL = (TWO * state[tr].x  *state[tr-1].x * (state[tr].z - state[tr-1].z) +
        (state[tr-1].y * state[tr-1].y * state[tr].x - state[tr].y * state[tr].y * state[tr-1].x));

    REAL pR = (TWO * state[tr].x * state[tr+1].x * (state[tr+1].z - state[tr].z) +
        (state[tr].y * state[tr].y * state[tr+1].x - state[tr+1].y * state[tr+1].y * state[tr].x));

    REAL pRR = (flagRight) ? ZERO : (TWO * state[tr+1].x * state[tr+2].x * (state[tr+2].z - state[tr+1].z) +
        (state[tr+1].y * state[tr+1].y * state[tr+2].x - state[tr+2].y * state[tr+2].y * state[tr+1].x));


    //This is the temporary state bounded by the limitor function.
    REALthree tempStateLeft = (!pLL || !pL || (pLL < 0 != pL <0)) ? state[tr-1] : limitor(state[tr-1], state[tr], (state[tr-2].x*pL/(state[tr].x*pLL)));
    REALthree tempStateRight = (!pL || !pR || (pL < 0 != pR <0)) ? state[tr] : limitor(state[tr], state[tr-1], (state[tr+1].x*pL/(state[tr-1].x*pR)));

    //Pressure needs to be recalculated for the new limited state variables.
    REALthree flux = eulerFlux(tempStateLeft,tempStateRight);

    //Do the same thing with the right side.
    tempStateLeft = (!pL || !pR || (pL < 0 != pR <0)) ? state[tr] : limitor(state[tr], state[tr+1], (state[tr-1].x*pR/(state[tr+1].x*pL)));
    tempStateRight = (!pRR || !pR || (pRR < 0 != pR <0)) ? state[tr+1] : limitor(state[tr+1], state[tr], (state[tr+2].x*pR/(state[tr].x*pRR)));

    flux -= eulerFlux(tempStateLeft,tempStateRight);

    //Add the change back to the node in question.
    #ifdef __CUDA_ARCH__
    return state[tr] + (QUARTER * dimens.dt_dx * flux);
    #else
    return state[tr] + (QUARTER * dimz.dt_dx * flux);
    #endif


}

//Same thing as the predictor step, but this final step adds the result to the original state variables to advance to the next timestep.
//But the predictor variables to find the fluxes.
__device__ __host__
REALthree
eulerFinalStep(REALthree *state, int tr, char flagLeft, char flagRight)
{

    REAL pLL = (flagLeft) ? ZERO : (TWO * state[tr-1].x * state[tr-2].x * (state[tr-1].z - state[tr-2].z) +
        (state[tr-2].y * state[tr-2].y*  state[tr-1].x - state[tr-1].y * state[tr-1].y * state[tr-2].x)) ;

    REAL pL = (TWO * state[tr].x  *state[tr-1].x * (state[tr].z - state[tr-1].z) +
        (state[tr-1].y * state[tr-1].y * state[tr].x - state[tr].y * state[tr].y * state[tr-1].x));

    REAL pR = (TWO * state[tr].x * state[tr+1].x * (state[tr+1].z - state[tr].z) +
        (state[tr].y * state[tr].y * state[tr+1].x - state[tr+1].y * state[tr+1].y * state[tr].x));

    REAL pRR = (flagRight) ?  ZERO : (TWO * state[tr+1].x * state[tr+2].x * (state[tr+2].z - state[tr+1].z) +
        (state[tr+1].y * state[tr+1].y * state[tr+2].x - state[tr+2].y * state[tr+2].y * state[tr+1].x));


    //This is the temporary state bounded by the limitor function.
    REALthree tempStateLeft = (!pLL || !pL || (pLL < 0 != pL <0)) ? state[tr-1] : limitor(state[tr-1], state[tr], (state[tr-2].x*pL/(state[tr].x*pLL)));
    REALthree tempStateRight = (!pL || !pR || (pL < 0 != pR <0)) ? state[tr] : limitor(state[tr], state[tr-1], (state[tr+1].x*pL/(state[tr-1].x*pR)));

    //Pressure needs to be recalculated for the new limited state variables.
    REALthree flux = eulerFlux(tempStateLeft,tempStateRight);

    //Do the same thing with the right side.
    tempStateLeft = (!pL || !pR || (pL < 0 != pR <0)) ? state[tr] : limitor(state[tr], state[tr+1], (state[tr-1].x*pR/(state[tr+1].x*pL)));
    tempStateRight = (!pRR || !pR || (pRR < 0 != pR <0))  ? state[tr+1] : limitor(state[tr+1], state[tr], (state[tr+2].x*pR/(state[tr].x*pRR)));

    flux -= eulerFlux(tempStateLeft,tempStateRight);

    #ifdef __CUDA_ARCH__
    return (HALF * dimens.dt_dx * flux);
    #else
    return (HALF * dimz.dt_dx * flux);
    #endif

}

__global__
void
swapKernel(const REALthree *passing_side, REALthree *bin, int direction)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
    int gidout = (gid + direction*blockDim.x) & dimens.idxend;

    bin[gidout] = passing_side[gid];

}

//Simple scheme with dirchlet boundary condition.
__global__
void
classicEuler(REALthree *euler_in, REALthree *euler_out, const bool final)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID

    const char4 truth = {gid == 0, gid == 1, gid == dimens.idxend_1, gid == dimens.idxend};

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

    if (final)
    {
        euler_out[gid] += eulerFinalStep(euler_in, gid, truth.y, truth.z);
    }
    else
    {
        euler_out[gid] = eulerStutterStep(euler_in, gid, truth.y, truth.z);
    }
}

__global__
void
upTriangle(const REALthree *IC, REALthree *right, REALthree *left)
{

	extern __shared__ REALthree temper[];

	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	int tid = threadIdx.x; //Block Thread ID
    int tididx = tid + 2;
    int tidxTop = tididx + dimens.base;
    int k=4;

    //Assign the initial values to the first row in temper, each block
    //has it's own version of temper shared among its threads.
	temper[tididx] = IC[gid];

    __syncthreads();

	if (tid > 1 && tid <(blockDim.x-2))
	{
		temper[tidxTop] = eulerStutterStep(temper, tididx, false, false);
	}

	__syncthreads();

	//The initial conditions are timslice 0 so start k at 1.
	while (k<(blockDim.x>>1))
	{
		if (tid < (blockDim.x-k) && tid >= k)
		{
            temper[tididx] += eulerFinalStep(temper, tidxTop, false, false);

		}

        k+=2;
		__syncthreads();

		if (tid < (blockDim.x-k) && tid >= k)
		{
            temper[tidxTop] = eulerStutterStep(temper, tididx, false, false);
		}

		k+=2;
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
downTriangle(REALthree *IC, const REALthree *right, const REALthree *left)
{
	extern __shared__ REALthree temper[];
    //REALthree *temper_top = (REALthree*)&temper[dimens.base];

	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
    int tididx = tid + 2;
    int k = dimens.hts[2];
    int tidxTop = tididx + dimens.base;

	readIn(temper, right, left, tid, gid);

    const char4 truth = {gid == 0, gid == 1, gid == dimens.idxend_1, gid == dimens.idxend};

    __syncthreads();

	while(k>1)
	{
		if (tididx < (dimens.base-k) && tididx >= k)
		{
            temper[tidxTop] = eulerStutterStep(temper, tididx, truth.y, truth.z);
		}

        k-=2;
        __syncthreads();

        if (!truth.x && !truth.w && tididx < (dimens.base-k) && tididx >= k)
        {
            temper[tididx] += eulerFinalStep(temper, tidxTop, truth.y, truth.z);

        }

		k-=2;
		__syncthreads();
	}


    IC[gid] = temper[tididx];
}

//Full refers to whether or not there is a node run on the CPU.
__global__
void
wholeDiamond(REALthree *right, REALthree *left, bool full)
{

    extern __shared__ REALthree temper[];

	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	int tididx = tid + 2;
    int tidxTop = tididx + dimens.base;

    char4 truth = {gid == 0, gid == 1, gid == dimens.idxend_1, gid == dimens.idxend};

    if (!full)
    {
        gid += blockDim.x;
        truth.x = false, truth.y = false, truth.z = false, truth.w = false;
    }

    readIn(temper, right, left, tid, gid);

    __syncthreads();

    int k = dimens.hts[0];

    if (tididx < (dimens.base-dimens.hts[2]) && tididx >= dimens.hts[2])
    {
        temper[tidxTop] = eulerStutterStep(temper, tididx, truth.y, truth.z);
    }

    __syncthreads();

    while(k>4)
    {
        if (tididx < (dimens.base-k) && tididx >= k)
        {
            temper[tididx] += eulerFinalStep(temper, tidxTop, truth.y, truth.z);
        }

        k -= 2;
        __syncthreads();

        if (tididx < (dimens.base-k) && tididx >= k)
        {
            temper[tidxTop] = eulerStutterStep(temper, tididx, truth.y, truth.z);
        }

        k -= 2;
        __syncthreads();
    }

    // -------------------TOP PART------------------------------------------

    if (!truth.w  &&  !truth.x)
    {
        temper[tididx] += eulerFinalStep(temper, tidxTop, truth.y, truth.z);
    }

    __syncthreads();

    if (tididx > 3 && tididx <(dimens.base-4))
	{
        temper[tidxTop] = eulerStutterStep(temper, tididx, truth.y, truth.z);
	}


    k=6;
	__syncthreads();

	while(k<dimens.hts[4])
	{
		if (tididx < (dimens.base-k) && tididx >= k)
		{
            temper[tididx] += eulerFinalStep(temper, tidxTop, truth.y, truth.z);
        }

        k+=2;
        __syncthreads();

        if (tididx < (dimens.base-k) && tididx >= k)
        {
            temper[tidxTop] = eulerStutterStep(temper, tididx, truth.y, truth.z);
		}


		k+=2;
		__syncthreads();

	}

    writeOut(temper, right, left, tid, gid);

}


//Split one is always first.
__global__
void
splitDiamond(REALthree *right, REALthree *left)
{
    extern __shared__ REALthree temper[];

    //Same as upTriangle
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
    int tididx = tid + 2;
    int tidxTop = tididx + dimens.base;

	readIn(temper, right, left, tid, gid);

    const char4 truth = {gid == dimens.hts[0], gid == dimens.hts[1], gid == dimens.hts[2], gid == dimens.hts[3]};

    int k = dimens.hts[2];
    __syncthreads();

    if (truth.z)
    {
        temper[tididx] = dbd[0];
        temper[tidxTop] = dbd[0];
    }
    if (truth.y)
    {
        temper[tididx] = dbd[1];
        temper[tidxTop] = dbd[1];
    }

    __syncthreads();

    while(k>0)
    {

        if (!truth.y && !truth.z && tididx < (dimens.base-k) && tididx >= k)
        {
            temper[tidxTop] = eulerStutterStep(temper, tididx, truth.w, truth.x);
        }

        k -= 2;
        __syncthreads();

        if (!truth.y && !truth.z && tididx < (dimens.base-k) && tididx >= k)
        {
            temper[tididx] += eulerFinalStep(temper, tidxTop, truth.w, truth.x);
        }

        k -= 2;
        __syncthreads();
    }

    if (!truth.y && !truth.z && tid > 1 && tid <(blockDim.x-2))
	{
        temper[tidxTop] = eulerStutterStep(temper, tididx, truth.w, truth.x);
	}

	__syncthreads();
    k=4;

    //The initial conditions are timslice 0 so start k at 1.
    while(k<dimens.hts[2])
    {
        if (!truth.y && !truth.z && tid < (blockDim.x-k) && tid >= k)
        {
            temper[tididx] += eulerFinalStep(temper, tidxTop, truth.w, truth.x);

        }

        k+=2;
        __syncthreads();

        if (!truth.y && !truth.z && tid < (blockDim.x-k) && tid >= k)
        {
            temper[tidxTop] = eulerStutterStep(temper, tididx, truth.w, truth.x);
        }


        k+=2;
        __syncthreads();

    }

	writeOut(temper, right, left, tid, gid);
}


using namespace std;

__host__
void
CPU_diamond(REALthree *temper, int htcpu[5])
{

    omp_set_num_threads(8);

    temper[htcpu[2]] = bd[0];
    temper[htcpu[2]+dimz.base] = bd[0];

    temper[htcpu[1]] = bd[1];
    temper[htcpu[1]+dimz.base] = bd[1];

    //Splitting it is the whole point!
    for (int k = htcpu[0]; k>0; k-=4)
    {
        #pragma omp parallel for
        for(int n = k; n<(dimz.base-k); n++)
        {
            if (n!=htcpu[1] && n!=htcpu[2])
            {
                temper[n+dimz.base] = eulerStutterStep(temper, n, (n==htcpu[3]),(n==htcpu[0]));
            }
        }

        #pragma omp parallel for
        for(int n = k-2; n<(dimz.base-(k-2)); n++)
        {
            if (n!=htcpu[1] && n!=htcpu[2])
            {
                temper[n] += eulerFinalStep(temper, n+dimz.base, n==htcpu[3],(n==htcpu[0]));
            }
        }
    }

    #pragma omp parallel for
    for(int n = 4; n < (dimz.base-4); n++)
    {
        if (n!=htcpu[1] && n!=htcpu[2])
        {
            temper[n+dimz.base] = eulerStutterStep(temper, n, (n==htcpu[3]),(n==htcpu[0]));
        }
    }

    //Top part.
    for (int k = 6; k<htcpu[2]; k+=4)
    {
        #pragma omp parallel
        for(int n = k; n<(dimz.base-k); n++)
        {
            if (n!=htcpu[1] && n!=htcpu[2])
            {
                temper[n] += eulerFinalStep(temper, n + dimz.base, (n==htcpu[3]), (n==htcpu[0]));
            }
        }

        #pragma omp parallel for
        for(int n = (k+2); n<(dimz.base-(k+2)); n++)
        {
            if (n!=htcpu[1] && n!=htcpu[2])
            {
                temper[n+dimz.base] = eulerStutterStep(temper, n, (n==htcpu[3]),(n==htcpu[0]));
            }
        }
    }

    // temper[htcpu[1]] = bd[1]
    // temper[htcpu[1]] = bd[1]

}

REAL
__host__ __inline__
energy(REAL p, REAL rho, REAL u)
{
    return (p/(dimz.mgam*rho) + HALF*rho*u*u);
}

//Classic Discretization wrapper.
double
classicWrapper(const int bks, int tpb, const int dv, const REAL dt, const REAL t_end,
    REALthree *IC, REALthree *T_f, const float freq, ofstream &fwr)
{
    REALthree *dEuler_in, *dEuler_out;

    cudaMalloc((void **)&dEuler_in, sizeof(REALthree)*dv);
    cudaMalloc((void **)&dEuler_out, sizeof(REALthree)*dv);

    // Copy the initial conditions to the device array.
    cudaMemcpy(dEuler_in,IC,sizeof(REALthree)*dv,cudaMemcpyHostToDevice);

    double t_eq = 0.0;
    double twrite = freq;

    while (t_eq < t_end)
    {
        classicEuler <<< bks,tpb >>> (dEuler_in, dEuler_out, false);
        classicEuler <<< bks,tpb >>> (dEuler_out, dEuler_in, true);
        t_eq += dt;

        if (t_eq > twrite)
        {
            cudaMemcpy(T_f, dEuler_in, sizeof(REALthree)*dv, cudaMemcpyDeviceToHost);

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
            for (int k = 1; k<(dv-1); k++) fwr << pressure(T_f[k]) << " ";
            fwr << endl;

            twrite += freq;
        }
    }

    cudaMemcpy(T_f, dEuler_in, sizeof(REALthree)*dv, cudaMemcpyDeviceToHost);

    cudaFree(dEuler_in);
    cudaFree(dEuler_out);

    return t_eq;

}

//The wrapper that calls the routine functions.
double
sweptWrapper(const int bks, int tpb, const int dv, REAL dt, const REAL t_end, const int cpu,
    REALthree *IC, REALthree *T_f, const float freq, ofstream &fwr)
{

    const size_t smem = (2*dimz.base)*sizeof(REALthree);

    int htcpu[5];
    for (int k=0; k<5; k++) htcpu[k] = dimz.hts[k]+2;

	REALthree *d_IC, *d_right, *d_left;

	cudaMalloc((void **)&d_IC, sizeof(REALthree)*dv);
	cudaMalloc((void **)&d_right, sizeof(REALthree)*dv);
	cudaMalloc((void **)&d_left, sizeof(REALthree)*dv);

	// Copy the initial conditions to the device array.
	cudaMemcpy(d_IC,IC,sizeof(REALthree)*dv,cudaMemcpyHostToDevice);
	// Start the counter and start the clock.
	const double t_fullstep = 0.25*dt*(double)tpb;

	upTriangle <<< bks,tpb,smem >>>(d_IC,d_right,d_left);

    swapKernel <<< bks,tpb >>> (d_right, d_IC, 1);
    swapKernel <<< bks,tpb >>> (d_IC, d_right, 0);

    double t_eq;
    double twrite = freq;

	// Call the kernels until you reach the iteration limit.

    if (cpu)
    {
        REALthree *h_right, *h_left;
        REALthree *tmpr = (REALthree *) malloc(smem);
        cudaHostAlloc((void **) &h_right, tpb*sizeof(REALthree), cudaHostAllocDefault);
        cudaHostAlloc((void **) &h_left, tpb*sizeof(REALthree), cudaHostAllocDefault);
        double time0, time1;
        double tf=0.0;
        int cnt=0;

        // h_right = (REALthree *) malloc(tpb*sizeof(REALthree));
        // h_left = (REALthree *) malloc(tpb*sizeof(REALthree));

        t_eq = t_fullstep;

        cudaStream_t st1, st2, st3;
        cudaStreamCreate(&st1);
        cudaStreamCreate(&st2);
        cudaStreamCreate(&st3);

        //Split Diamond Begin------

        wholeDiamond <<< bks-1,tpb,smem >>>(d_right, d_left, false);

        cudaMemcpyAsync(h_left, d_left, tpb*sizeof(REALthree), cudaMemcpyDeviceToHost, st2);
        cudaMemcpyAsync(h_right, d_right , tpb*sizeof(REALthree), cudaMemcpyDeviceToHost, st3);

        cudaStreamSynchronize(st2);
        cudaStreamSynchronize(st3);

        // CPU Part Start -----

        time0 = omp_get_wtime( );

        #pragma omp parallel for num_threads(8)
        for (int k = 0; k<tpb; k++)  readIn(tmpr, h_right, h_left, k, k);

        CPU_diamond(tmpr, htcpu);

        #pragma omp parallel for num_threads(8)
        for (int k = 0; k<tpb; k++)  writeOut(tmpr, h_right, h_left, k, k);


        time1 = omp_get_wtime( );
        tf += (time1-time0)*1.0e6; //In us
        cnt++;

        cout << "CPU time 1: " << tf << " (us)" << endl;

        cudaMemcpyAsync(d_right, h_right, tpb*sizeof(REALthree), cudaMemcpyHostToDevice,st2);
        cudaMemcpyAsync(d_left, h_left, tpb*sizeof(REALthree), cudaMemcpyHostToDevice,st3);

        // CPU Part End -----

        // Automatic synchronization with memcpy in default stream
        swapKernel <<< bks,tpb >>> (d_left, d_IC, -1);
        swapKernel <<< bks,tpb >>> (d_IC, d_left, 0);

        while(t_eq < t_end)
        {

            wholeDiamond <<< bks,tpb,smem >>>(d_right, d_left, true);

            swapKernel <<< bks,tpb >>> (d_right, d_IC, 1);
            swapKernel <<< bks,tpb >>> (d_IC, d_right, 0);

            //Split Diamond Begin------

            wholeDiamond <<< bks-1,tpb,smem >>>(d_right, d_left, false);

            cudaMemcpyAsync(h_right, d_right, tpb*sizeof(REALthree), cudaMemcpyDeviceToHost, st2);
            cudaMemcpyAsync(h_left, d_left , tpb*sizeof(REALthree), cudaMemcpyDeviceToHost, st3);

            cudaStreamSynchronize(st2);
            cudaStreamSynchronize(st3);

            // CPU Part End -----

            time0 = omp_get_wtime( );

            #pragma omp parallel for num_threads(8)
            for (int k = 0; k<tpb; k++)
            {
                readIn(tmpr, h_right, h_left, k, k);
            }

            CPU_diamond(tmpr, htcpu);

            #pragma omp parallel for num_threads(8)
            for (int k = 0; k<tpb; k++)
            {
                writeOut(tmpr, h_right, h_left, k, k);
            }

            time1 = omp_get_wtime( );
            tf += (time1-time0)*1.0e6; //In us
            cnt++;

            cudaMemcpyAsync(d_right, h_right, tpb*sizeof(REALthree), cudaMemcpyHostToDevice,st2);
            cudaMemcpyAsync(d_left, h_left, tpb*sizeof(REALthree), cudaMemcpyHostToDevice,st3);


            // CPU Part End -----

            // Automatic synchronization with memcpy in default stream
            swapKernel <<< bks,tpb >>> (d_left, d_IC, -1);
            swapKernel <<< bks,tpb >>> (d_IC, d_left, 0);

            //Split Diamond End------

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

        cudaFreeHost(h_right);
        cudaFreeHost(h_left);
        cudaStreamDestroy(st1);
        cudaStreamDestroy(st2);
        cudaStreamDestroy(st3);
        // free(h_right);
        // free(h_left);
        free(tmpr);

        cout << "Average CPU time: " << tf/(double)cnt << " (us)" << endl;
	}
    else
    {
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

    dimz.gam = 1.4;
    dimz.mgam = 0.4;

    bd[0].x = ONE; //Density
    bd[1].x = 0.125;
    bd[0].y = ZERO; //Velocity
    bd[1].y = ZERO;
    //bd[0].w = ONE; //Pressure
    //bd[1].w = 0.1;
    bd[0].z = ONE/dimz.mgam; //Energy
    bd[1].z = 0.1/dimz.mgam;


    const int dv = atoi(argv[1]); //Number of spatial points
	const int tpb = atoi(argv[2]); //Threads per Block
    const REAL dt = atof(argv[3]);
	const float tf = atof(argv[4]); //Finish time
    const float freq = atof(argv[5]);
    const int scheme = atoi(argv[6]); //1 for Swept 0 for classic
    const int share = atoi(argv[7]);
    const int bks = dv/tpb; //The number of blocks
    const REAL dx = lx/((REAL)dv-TWO);
    char const *prec;
    prec = (sizeof(REAL)<6) ? "Single": "Double";

    //Declare the dimensions in constant memory.
    dimz.dt_dx = dt/dx; // dt/dx
    dimz.base = tpb+4;
    dimz.idxend = dv-1;
    dimz.idxend_1 = dv-2;

    for (int k=-2; k<3; k++) dimz.hts[k+2] = (tpb/2) + k;

    cout << "Euler --- #Blocks: " << bks << " | Length: " << lx << " | Precision: " << prec << " | dt/dx: " << dimz.dt_dx << endl;

	//Conditions for main input.  Unit testing kinda.
	//dv and tpb must be powers of two.  dv must be larger than tpb and divisible by
	//tpb.

	if ((dv & (tpb-1) !=0) || (tpb&31) != 0)
    {
        cout << "INVALID NUMERIC INPUT!! "<< endl;
        cout << "2nd ARGUMENT MUST BE A POWER OF TWO >= 32 AND FIRST ARGUMENT MUST BE DIVISIBLE BY SECOND" << endl;
        exit(-1);
    }

    if (dimz.dt_dx > .15)
    {
        cout << "The value of dt/dx (" << dimz.dt_dx << ") is too high.  In general it must be <=.15 for stability." << endl;
        exit(-1);
    }

	// Initialize arrays.
    REALthree *IC, *T_final, *d_temp;
    cudaMalloc((void **)&d_temp, sizeof(REALthree)*tpb);
	cudaHostAlloc((void **) &IC, dv*sizeof(REALthree), cudaHostAllocDefault);
	cudaHostAlloc((void **) &T_final, dv*sizeof(REALthree), cudaHostAllocDefault);

    cudaFree(d_temp);

    // IC = (REALthree *) malloc(dv*sizeof(REALthree));
    // T_final = (REALthree *) malloc(dv*sizeof(REALthree));

    #pragma omp parallel for num_threads(8)
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
    for (int k = 1; k<(dv-1); k++) fwr << pressure(IC[k]) << " ";
    fwr << endl;

    //Transfer data to GPU.
	// This puts the Fourier number in constant memory.
	cudaMemcpyToSymbol(dimens,&dimz,sizeof(dimensions));
    cudaMemcpyToSymbol(dbd,&bd,2*sizeof(REALthree));

    cudaError_t error1 = cudaGetLastError();
    if(error1 != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error ahead: %s\n", cudaGetErrorString(error1));
        exit(-1);
    }

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
    for (int k = 1; k<(dv-1); k++) fwr << pressure(T_final[k]) << " ";
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