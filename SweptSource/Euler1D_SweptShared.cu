/**
    NOTE: This file is where the explanatory comments for this package appear. The other source files only have superficial comments.

    This file evaluates the Euler equations applied to the 1D Sod Shock Tube problem.  It demonstrates the numerical solution to this problem in parallel using the GPU. The solution procedure uses a second order finite volume scheme with a minmod limiter parameterized by the Pressure ratio at cells on a three point stencil.  The solution also uses a second-order in time (RK2 or midpoint) scheme.

    The boundary conditions are:
    Q(t=0,x) = QL if x<L/2 else QR
    Q(t,x=0,dx) = QL
    Q(t,x=L,L-dx) = QR
    Where Q is the vector of dependent variables.
    
    The problem may be evaluated in three ways: Classic, SharedGPU, and Hybrid.  Classic simply steps forward in time and calls the kernel once every timestep (predictor step or full step).  SharedGPU uses the GPU for all computation and applies the swept rule.  Hybrid applies the swept rule but computes the node on the boundary with the CPU.  
*/
/* 
    Copyright (C) 2017 Kyle Niemeyer, niemeyek@oregonstate.edu AND
    Daniel Magee, mageed@oregonstate.edu
*/
/*
    This file is distribued under the MIT License.  See LICENSE at top level of directory or: <https://opensource.org/licenses/MIT>.
*/

// To compile this program alone from the command line:
// nvcc -o ./bin/EulerOut Euler1D_SweptShared.cu -gencode arch=compute_35,code=sm_35 -lm -restrict -Xcompiler -fopenmp
// Use whatever compute_xx, sm_xx applies to the GPU you are using.
// Add -Xptxas=-v to the end of the compile line to inspect register usage.

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



// This file uses vector types to hold the dependent variables so fundamental operations on those types are defined as macros to accommodate different data types.  Also, keeping types consistent for common constants (0, 1, 2, etc) used in computation has an appreciable positive effect on performance.
#ifndef REAL
    #define REAL            float
    #define REALtwo         float2
    #define REALthree       float3
    #define SQUAREROOT(x)   sqrtf(x)

    #define ZERO            0.0f
    #define QUARTER         0.25f
    #define HALF            0.5f
    #define ONE             1.f
    #define TWO             2.f
#else

    #define ZERO            0.0
    #define QUARTER         0.25
    #define HALF            0.5
    #define ONE             1.0
    #define TWO             2.0
    #define SQUAREROOT(x)   sqrt(x)
#endif

// Hardwire in the length of the 
const REAL lx = 1.0;

// The structure to carry the initial and boundary conditions.
// 0 is left 1 is right.
REALthree bd[2];

//dbd is the boundary condition in device constant memory.
__constant__ REALthree dbd[2]; 

//Protoype for useful information struct.
struct dimensions {
    REAL gam; // Heat capacity ratio
    REAL mgam; // 1- Heat capacity ratio
    REAL dt_dx; // deltat/deltax
    int base; // Length of node + stencils at end (4)
    int idxend; // Last index (number of spatial points - 1)
    int idxend_1; // Num spatial points - 2
    int hts[5]; // The five point stencil around base/2
};

// structure of dimensions in cpu memory
dimensions dimz;

// Useful and efficient to keep the important constant information in GPU constant memory.
__constant__ dimensions dimens;

/**
    Takes the passed the right and left arrays from previous cycle and inserts them into new SHARED memory working array.  
    
    Called at start of kernel/ function.  The passed arrays have already been passed readIn only finds the correct index and inserts the values and flips the indices to seed the working array for the cycle.
    @param rights  The array for the right side of the triangle.
    @param lefts  The array for the left side.
    @param td  The thread block array id.
    @param gd  The thread global array id.
    @param temp  The working array in shared memory
*/
__host__ __device__
__forceinline__
void
readIn(REALthree *temp, const REALthree *rights, const REALthree *lefts, int td, int gd)
{
    // The index in the SHARED memory working array to place the corresponding member of right or left.
    #ifdef __CUDA_ARCH__  // Accesses the correct structure in constant memory.
	int leftidx = dimens.hts[4] + (((td>>2) & 1) * dimens.base) + (td & 3) - (4 + ((td>>2)<<1));
	int rightidx = dimens.hts[4] + (((td>>2) & 1) * dimens.base) + ((td>>2)<<1) + (td & 3);
    #else
    int leftidx = dimz.hts[4] + (((td>>2) & 1) * dimz.base) + (td & 3) - (4 + ((td>>2)<<1));
    int rightidx = dimz.hts[4] + (((td>>2) & 1) * dimz.base) + ((td>>2)<<1) + (td & 3);
    #endif

	temp[leftidx] = rights[gd];
	temp[rightidx] = lefts[gd];
}

/**
    Write out the right and left arrays at the end of a kernel when passing right.  
    
    Called at end of the kernel/ function.  The values of the working array are collected in the right and left arrays.  As they are collected, the passed edge (right) is inserted at an offset. This function is never called from the host so it doesn't need the preprocessor CUDA flags.'
    @param temp  The working array in shared memory
    @param rights  The array for the right side of the triangle.
    @param lefts  The array for the left side.
    @param td  The thread block array id.
    @param gd  The thread global array id.
    @param bd  The number of threads in a block (spatial points in a node).
*/
__device__
__forceinline__
void
writeOutRight(REALthree *temp, REALthree *rights, REALthree *lefts, int td, int gd, int bd)
{
    int gdskew = (gd + bd) & dimens.idxend; //The offset for the right array.
    int leftidx = (((td>>2) & 1)  * dimens.base) + ((td>>2)<<1) + (td & 3) + 2; 
    int rightidx = (dimens.base-6) + (((td>>2) & 1)  * dimens.base) + (td & 3) - ((td>>2)<<1); 
	rights[gdskew] = temp[rightidx];
	lefts[gd] = temp[leftidx];
}

/**
    Write out the right and left arrays at the end of a kernel when passing left.  
    
    Called at end of the kernel/ function.  The values of the working array are collected in the right and left arrays.  As they are collected, the passed edge (left) is inserted at an offset. 
    @param temp  The working array in shared memory
    @param rights  The array for the right side of the triangle.
    @param lefts  The array for the left side.
    @param td  The thread block array id.
    @param gd  The thread global array id.
    @param bd  The number of threads in a block (spatial points in a node).
*/
__host__ __device__
__forceinline__
void
writeOutLeft(REALthree *temp, REALthree *rights, REALthree *lefts, int td, int gd, int bd)
{
    #ifdef __CUDA_ARCH__
    int gdskew = (gd - bd) & dimens.idxend; //The offset for the right array.
    int leftidx = (((td>>2) & 1)  * dimens.base) + ((td>>2)<<1) + (td & 3) + 2;
    int rightidx = (dimens.base-6) + (((td>>2) & 1)  * dimens.base) + (td & 3) - ((td>>2)<<1); 
    #else
    int gdskew = gd;
    int leftidx = (((td>>2) & 1)  * dimz.base) + ((td>>2)<<1) + (td & 3) + 2;
    int rightidx = (dimz.base-6) + (((td>>2) & 1)  * dimz.base) + (td & 3) - ((td>>2)<<1);
    #endif

    rights[gd] = temp[rightidx];
    lefts[gdskew] = temp[leftidx];
}

/**
    Calculates the pressure at the current spatial point with the (x,y,z) rho, u * rho, e *rho state variables.
    
    Calculates pressure from working array variables.  Pressure is not stored outside procedure to save memory.
    @param current  The state variables at current node
    @return Pressure at subject node
*/
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

/**
    Calculates the parameter for the first term in the spectral radius formula (P/rho).
    
    Since this formula is essentially a lambda for a single calculation, input vector y and z are u_sp and e_sp respectively without multipliation by rho and it returns the pressure over rho to skip the next step.
    @param current  The Roe averaged state variables at the interface.  Where y and z are u_sp and e_sp respectively without multipliation by rho.
    @return Roe averaged pressure over rho at the interface 
*/
__device__ __host__
__forceinline__
REAL
pressureHalf(REALthree current)
{
    #ifdef __CUDA_ARCH__
    return dimens.mgam * (current.z - HALF * current.y * current.y);
    #else
    return dimz.mgam * (current.z - HALF * current.y * current.y);
    #endif
}

/**
    Reconstructs the state variables if the pressure ratio is finite and positive.

    @param cvCurrent  The state variables at the point in question.
    @param cvOther  The neighboring spatial point state variables.
    @param pRatio  The pressure ratio Pr-Pc/(Pc-Pl).
    @return The reconstructed value at the current side of the interface.
*/
__device__ __host__
__forceinline__
REALthree
limitor(REALthree cvCurrent, REALthree cvOther, REAL pRatio)
{
    return (cvCurrent + HALF * min(pRatio,ONE) * (cvOther - cvCurrent));
}

/**
    Uses the reconstructed interface values as inputs to flux function F(Q)

    @param cvLeft Reconstructed value at the left side of the interface.
    @param cvRight  Reconstructed value at the left side of the interface.
    @return  The combined flux from the function.
*/
__device__ __host__
__forceinline__
REALthree
eulerFlux(REALthree cvLeft, REALthree cvRight)
{
    #ifndef __CUDA_ARCH__
    using namespace std;
    #endif
    REAL uLeft = cvLeft.y/cvLeft.x;
    REAL uRight = cvRight.y/cvRight.x;

    REAL pL = pressure(cvLeft);
    REAL pR = pressure(cvRight);

    REALthree flux;
    flux.x = (cvLeft.y + cvRight.y);
    flux.y = (cvLeft.y*uLeft + cvRight.y*uRight + pL + pR);
    flux.z = (cvLeft.z*uLeft + cvRight.z*uRight + uLeft*pL + uRight*pR);

    return flux;
}

/**
    Finds the spectral radius and applies it to the interface.

    @param cvLeft Reconstructed value at the left side of the interface.
    @param cvRight  Reconstructed value at the left side of the interface.
    @return  The spectral radius multiplied by the difference of the reconstructed values
*/
__device__ __host__
__forceinline__
REALthree
eulerSpectral(REALthree cvLeft, REALthree cvRight)
{
    #ifndef __CUDA_ARCH__
    using namespace std;
    #endif

    REALthree halfState;
    REAL rhoLeftsqrt = SQUAREROOT(cvLeft.x);
    REAL rhoRightsqrt = SQUAREROOT(cvRight.x);

    halfState.x = rhoLeftsqrt * rhoRightsqrt;
    REAL halfDenom = ONE/(halfState.x*(rhoLeftsqrt + rhoRightsqrt));

    halfState.y = (rhoLeftsqrt*cvRight.y + rhoRightsqrt*cvLeft.y)*halfDenom;
    halfState.z = (rhoLeftsqrt*cvRight.z + rhoRightsqrt*cvLeft.z)*halfDenom;

    REAL pH = pressureHalf(halfState);

    #ifdef __CUDA_ARCH__
    return (SQUAREROOT(pH*dimens.gam) + fabs(halfState.y)) * (cvLeft - cvRight);
    #else
    return (SQUAREROOT(pH*dimz.gam) + fabs(halfState.y)) * (cvLeft - cvRight);
    #endif
}

/**
    The predictor step of the finite volume scheme.

    First: The pressure ratio calculation is decomposed to avoid division and calling the limitor unnecessarily.  Although 3 pressure ratios would be required, we can see that there are only 4 unique numerators and denominators in that calculation which can be calculated without using division or calling pressure (which uses division).  The edge values aren't guaranteed to have the correct conditions so the flags set the appropriate pressure values to 0 (Pressures are equal) at the edges.
    Second:  The numerator and denominator are tested to see if the pressure ratio will be Nan or <=0. If they are, the limitor doesn't need to be called.  If they are not, call the limitor and calculate the pressure ratio.
    Third:  Use the reconstructed values at the interfaces to get the flux at the interfaces using the spectral radius and flux functions and combine the results with the flux variable.
    Fourth: Repeat for second interface and update current volume. 
    @param state  Reference to the working array in SHARED memory holding the dependent variables.
    @param tr  The indices of the stencil points.
    @param flagLeft  True if the point is the first finite volume in the tube.
    @param flagRight  True if the point is the last finite volume in the tube.
    @return  The updated value at the current spatial point.
*/
__device__ __host__
REALthree
eulerStutterStep(REALthree *state, int tr, char flagLeft, char flagRight)
{
    //P1-P0
    REAL pLL = (flagLeft) ? ZERO : (TWO * state[tr-1].x * state[tr-2].x * (state[tr-1].z - state[tr-2].z) +
        (state[tr-2].y * state[tr-2].y*  state[tr-1].x - state[tr-1].y * state[tr-1].y * state[tr-2].x)) ;
    //P2-P1
    REAL pL = (TWO * state[tr].x  *state[tr-1].x * (state[tr].z - state[tr-1].z) +
        (state[tr-1].y * state[tr-1].y * state[tr].x - state[tr].y * state[tr].y * state[tr-1].x));
    //P3-P2
    REAL pR = (TWO * state[tr].x * state[tr+1].x * (state[tr+1].z - state[tr].z) +
        (state[tr].y * state[tr].y * state[tr+1].x - state[tr+1].y * state[tr+1].y * state[tr].x));
    //P4-P3
    REAL pRR = (flagRight) ? ZERO : (TWO * state[tr+1].x * state[tr+2].x * (state[tr+2].z - state[tr+1].z) +
        (state[tr+1].y * state[tr+1].y * state[tr+2].x - state[tr+2].y * state[tr+2].y * state[tr+1].x));

    //This is the temporary state bounded by the limitor function.
    //Pr0 = PL/PLL*rho0/rho2  Pr0 is not -, 0, or nan.
    REALthree tempStateLeft = (!pLL || !pL || (pLL < 0 != pL <0)) ? state[tr-1] : limitor(state[tr-1], state[tr], (state[tr-2].x*pL/(state[tr].x*pLL)));
    //Pr1 = PR/PL*rho1/rho3  Pr1 is not - or nan, pass Pr1^-1.
    REALthree tempStateRight = (!pL || !pR || (pL < 0 != pR <0)) ? state[tr] : limitor(state[tr], state[tr-1], (state[tr+1].x*pL/(state[tr-1].x*pR)));

    //Pressure needs to be recalculated for the new limited state variables.
    REALthree flux = eulerFlux(tempStateLeft,tempStateRight);
    flux += eulerSpectral(tempStateLeft,tempStateRight);

    //Do the same thing with the right side.
    //Pr1 = PR/PL*rho1/rho3  Pr1 is not - or nan.
    tempStateLeft = (!pL || !pR || (pL < 0 != pR <0)) ? state[tr] : limitor(state[tr], state[tr+1], (state[tr-1].x*pR/(state[tr+1].x*pL)));
    //Pr2 = PRR/PR*rho2/rho4  Pr2 is not - or nan, pass Pr2^-1.
    tempStateRight = (!pRR || !pR || (pRR < 0 != pR <0)) ? state[tr+1] : limitor(state[tr+1], state[tr], (state[tr+2].x*pR/(state[tr].x*pRR)));

    flux -= eulerFlux(tempStateLeft,tempStateRight);
    flux -= eulerSpectral(tempStateLeft,tempStateRight);

    //Add the change back to the node in question.
    #ifdef __CUDA_ARCH__
    return state[tr] + (QUARTER * dimens.dt_dx * flux);
    #else
    return state[tr] + (QUARTER * dimz.dt_dx * flux);
    #endif

}

//Same thing as the predictor step, but this final step adds the result to the original state variables to advance to the next timestep while using the predictor variables to find the flux.
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

    REALthree tempStateLeft = (!pLL || !pL || (pLL < 0 != pL <0)) ? state[tr-1] : limitor(state[tr-1], state[tr], (state[tr-2].x*pL/(state[tr].x*pLL)));
    REALthree tempStateRight = (!pL || !pR || (pL < 0 != pR <0)) ? state[tr] : limitor(state[tr], state[tr-1], (state[tr+1].x*pL/(state[tr-1].x*pR)));

    REALthree flux = eulerFlux(tempStateLeft,tempStateRight);
    flux += eulerSpectral(tempStateLeft,tempStateRight);

    tempStateLeft = (!pL || !pR || (pL < 0 != pR <0)) ? state[tr] : limitor(state[tr], state[tr+1], (state[tr-1].x*pR/(state[tr+1].x*pL)));
    tempStateRight = (!pRR || !pR || (pRR < 0 != pR <0))  ? state[tr+1] : limitor(state[tr+1], state[tr], (state[tr+2].x*pR/(state[tr].x*pRR)));

    flux -= eulerFlux(tempStateLeft,tempStateRight);
    flux -= eulerSpectral(tempStateLeft,tempStateRight);

    // Return only the RHS of the discretization.
    #ifdef __CUDA_ARCH__
    return (HALF * dimens.dt_dx * flux);
    #else
    return (HALF * dimz.dt_dx * flux);
    #endif

}

/**
    Classic kernel for simple decomposition of spatial domain.

    Uses dependent variable values in euler_in to calculate euler out.  If it's the predictor step, finalstep is false.  If it is the final step the result is added to the previous euler_out value because this is RK2.

    @param euler_in The working array result of the kernel call before last (or initial condition) used to calculate the RHS of the discretization.
    @param euler_out The working array from the kernel call before last which either stores the predictor values or the full step values after the RHS is added into the solution.
    @param finalstep Flag for whether this is the final (True) or predictor (False) step
*/
__global__
void
classicEuler(REALthree *euler_in, REALthree *euler_out, const bool finalstep)
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

    if (finalstep)
    {
        euler_out[gid] += eulerFinalStep(euler_in, gid, truth.y, truth.z);
    }
    else
    {
        euler_out[gid] = eulerStutterStep(euler_in, gid, truth.y, truth.z);
    }
}

/**
    Builds an upright triangle using the swept rule.

    Upright triangle using the swept rule.  This function is called first using the initial conditions or after results are read out using downTriange.  In the latter case, it takes the result of down triangle as IC.

    @param IC Array of initial condition values in order of spatial point.
    @param outRight Array to store the right sides of the triangles to be passed.
    @param outLeft Array to store the left sides of the triangles to be passed.
*/
__global__
void
upTriangle(const REALthree *IC, REALthree *outRight, REALthree *outLeft)
{
	extern __shared__ REALthree temper[];

	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	int tididx = threadIdx.x + 2; //Block Thread ID
    int tidxTop = tididx + dimens.base; //
    int k=4;

    //Assign the initial values to the first row in temper, each block
    //has it's own version of temper shared among its threads.
	temper[tididx] = IC[gid];

    __syncthreads();

	if (threadIdx.x > 1 && threadIdx.x <(blockDim.x-2))
	{
		temper[tidxTop] = eulerStutterStep(temper, tididx, false, false);
	}

	__syncthreads();

	//The initial conditions are timslice 0 so start k at 1.
	while (k < (blockDim.x>>1))
	{
		if (threadIdx.x < (blockDim.x-k) && threadIdx.x >= k)
		{
            temper[tididx] += eulerFinalStep(temper, tidxTop, false, false);
		}

        k+=2;
		__syncthreads();

		if (threadIdx.x < (blockDim.x-k) && threadIdx.x >= k)
		{
            temper[tidxTop] = eulerStutterStep(temper, tididx, false, false);
		}

		k+=2;
		__syncthreads();

	}
    // Passes right and keeps left
    writeOutRight(temper, outRight, outLeft, threadIdx.x, gid, blockDim.x);
}

/**
    Builds an inverted triangle using the swept rule.

    Inverted triangle using the swept rule.  downTriangle is only called at the end when data is passed left.  It's never split.  Sides have already been passed between nodes, but will be swapped and parsed by readIn function.

    @param IC Full solution at some timestep.
    @param inRight Array of right edges seeding solution vector.
    @param inLeft Array of left edges seeding solution vector.
*/
__global__
void
downTriangle(REALthree *IC, const REALthree *inRight, const REALthree *inLeft)
{
	extern __shared__ REALthree temper[];

	int gid = blockDim.x * blockIdx.x + threadIdx.x;
    int tididx = threadIdx.x + 2;
    int tidxTop = tididx + dimens.base;
    int k = dimens.hts[2];

	readIn(temper, inRight, inLeft, threadIdx.x, gid);

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


/**
    Builds an diamond using the swept rule after a left pass.

    Unsplit diamond using the swept rule.  wholeDiamond must apply boundary conditions only at it's center.

    @param inRight Array of right edges seeding solution vector.
    @param inLeft Array of left edges seeding solution vector.
    @param outRight Array to store the right sides of the triangles to be passed.
    @param outLeft Array to store the left sides of the triangles to be passed.
    @param Full True if there is not a node run on the CPU, false otherwise.
*/
__global__
void
wholeDiamond(const REALthree *inRight, const REALthree *inLeft, REALthree *outRight, REALthree *outLeft, const bool split)
{

    extern __shared__ REALthree temper[];

    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    int tididx = threadIdx.x + 2;
    int tidxTop = tididx + dimens.base;

    char4 truth = {gid == 0, gid == 1, gid == dimens.idxend_1, gid == dimens.idxend};

    if (split)
    {
        gid += blockDim.x;
        truth.x = false, truth.y = false, truth.z = false, truth.w = false;
    }

    readIn(temper, inRight, inLeft, threadIdx.x, gid);

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

    if (split)
    {
        writeOutLeft(temper, outRight, outLeft, threadIdx.x, gid, blockDim.x);
    }
    else
    {
        writeOutRight(temper, outRight, outLeft, threadIdx.x, gid, blockDim.x);
    }
}


//Split one is always first.
__global__
void
splitDiamond(REALthree *inRight, REALthree *inLeft, REALthree *outRight, REALthree *outLeft)
{
    extern __shared__ REALthree temper[];

    //Same as upTriangle
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    int tididx = threadIdx.x + 2;
    int tidxTop = tididx + dimens.base;
    int k = dimens.hts[2];

	readIn(temper, inRight, inLeft, threadIdx.x, gid);

    const char4 truth = {gid == dimens.hts[0], gid == dimens.hts[1], gid == dimens.hts[2], gid == dimens.hts[3]};

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

    if (!truth.y && !truth.z && threadIdx.x > 1 && threadIdx.x <(blockDim.x-2))
	{
        temper[tidxTop] = eulerStutterStep(temper, tididx, truth.w, truth.x);
	}

	__syncthreads();
    k=4;

    //The initial conditions are timslice 0 so start k at 1.
    while(k<dimens.hts[2])
    {
        if (!truth.y && !truth.z && threadIdx.x < (blockDim.x-k) && threadIdx.x >= k)
        {
            temper[tididx] += eulerFinalStep(temper, tidxTop, truth.w, truth.x);

        }

        k+=2;
        __syncthreads();

        if (!truth.y && !truth.z && threadIdx.x < (blockDim.x-k) && threadIdx.x >= k)
        {
            temper[tidxTop] = eulerStutterStep(temper, tididx, truth.w, truth.x);
        }
        k+=2;
        __syncthreads();

    }

	writeOutLeft(temper, outRight, outLeft, threadIdx.x, gid, blockDim.x);
}


using namespace std;

__host__
__forceinline__
REAL
energy(REALthree subj)
{
    REAL u = subj.y/subj.x;
    return subj.z/subj.x - HALF*u*u;
}

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
}

//Classic Discretization wrapper.
double
classicWrapper(const int bks, int tpb, const int dv, const double dt, const double t_end,
    REALthree *IC, REALthree *T_f, const double freq, ofstream &fwr)
{
    REALthree *dEuler_in, *dEuler_out;

    cudaMalloc((void **)&dEuler_in, sizeof(REALthree)*dv);
    cudaMalloc((void **)&dEuler_out, sizeof(REALthree)*dv);

    // Copy the initial conditions to the device array.
    cudaMemcpy(dEuler_in,IC,sizeof(REALthree)*dv,cudaMemcpyHostToDevice);

    cout << "Classic scheme" << endl;

    double t_eq = 0.0;
    double twrite = freq - QUARTER*dt;

    while (t_eq < t_end)
    {
        classicEuler <<< bks,tpb >>> (dEuler_in, dEuler_out, false);
        classicEuler <<< bks,tpb >>> (dEuler_out, dEuler_in, true);
        t_eq += dt;

        if (t_eq > twrite)
        {
            cudaMemcpy(T_f, dEuler_in, sizeof(REALthree)*dv, cudaMemcpyDeviceToHost);

            fwr << "Density " << t_eq << " ";
            for (int k = 1; k<(dv-1); k++) fwr << T_f[k].x << " ";
            fwr << endl;

            fwr << "Velocity " << t_eq << " ";
            for (int k = 1; k<(dv-1); k++) fwr << T_f[k].y/T_f[k].x << " ";
            fwr << endl;

            fwr << "Energy " << t_eq << " ";
            for (int k = 1; k<(dv-1); k++) fwr << energy(T_f[k]) << " ";
            fwr << endl;

            fwr << "Pressure " << t_eq << " ";
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
sweptWrapper(const int bks, int tpb, const int dv, const double dt, const double t_end, const int cpu,
    REALthree *IC, REALthree *T_f, const double freq, ofstream &fwr)
{
    const size_t smem = (2*dimz.base)*sizeof(REALthree);
    const int cpuLoc = dv-tpb;

    int htcpu[5];
    for (int k=0; k<5; k++) htcpu[k] = dimz.hts[k]+2;

	REALthree *d_IC, *d0_right, *d0_left, *d2_right, *d2_left;

	cudaMalloc((void **)&d_IC, sizeof(REALthree)*dv);
	cudaMalloc((void **)&d0_right, sizeof(REALthree)*dv);
	cudaMalloc((void **)&d0_left, sizeof(REALthree)*dv);
    cudaMalloc((void **)&d2_right, sizeof(REALthree)*dv);
	cudaMalloc((void **)&d2_left, sizeof(REALthree)*dv);

	cudaMemcpy(d_IC,IC,sizeof(REALthree)*dv,cudaMemcpyHostToDevice);

	// Start the counter and start the clock.
	const double t_fullstep = 0.25*dt*(double)tpb;

	upTriangle <<<bks, tpb, smem>>> (d_IC, d0_right, d0_left);

    double t_eq;
    double twrite = freq - QUARTER*dt;

	// Call the kernels until you reach the final time

    if (cpu)
    {
        cout << "Hybrid Swept scheme" << endl;

        REALthree *h_right, *h_left;
        REALthree *tmpr = (REALthree *) malloc(smem);
        cudaHostAlloc((void **) &h_right, tpb*sizeof(REALthree), cudaHostAllocDefault);
        cudaHostAlloc((void **) &h_left, tpb*sizeof(REALthree), cudaHostAllocDefault);

        t_eq = t_fullstep;

        cudaStream_t st1, st2, st3;
        cudaStreamCreate(&st1);
        cudaStreamCreate(&st2);
        cudaStreamCreate(&st3);

        //Split Diamond Begin------

        wholeDiamond <<<bks-1, tpb, smem, st1>>> (d0_right, d0_left, d2_right, d2_left, true);

        cudaMemcpyAsync(h_left, d0_left, tpb*sizeof(REALthree), cudaMemcpyDeviceToHost, st2);
        cudaMemcpyAsync(h_right, d0_right, tpb*sizeof(REALthree), cudaMemcpyDeviceToHost, st3);

        cudaStreamSynchronize(st2);
        cudaStreamSynchronize(st3);

        // CPU Part Start -----

        for (int k=0; k<tpb; k++)  readIn(tmpr, h_right, h_left, k, k);

        CPU_diamond(tmpr, htcpu);

        for (int k=0; k<tpb; k++)  writeOutLeft(tmpr, h_right, h_left, k, k, 0);

        cudaMemcpyAsync(d2_right, h_right, tpb*sizeof(REALthree), cudaMemcpyHostToDevice, st2);
        cudaMemcpyAsync(d2_left + cpuLoc, h_left, tpb*sizeof(REALthree), cudaMemcpyHostToDevice, st3);

        // CPU Part End -----

        while(t_eq < t_end)
        {
            wholeDiamond <<<bks, tpb, smem>>> (d2_right, d2_left, d0_right, d0_left, false);

            //Split Diamond Begin------

            wholeDiamond <<<bks-1, tpb, smem, st1>>> (d0_right, d0_left, d2_right, d2_left, true);

            cudaMemcpyAsync(h_left, d0_left, tpb*sizeof(REALthree), cudaMemcpyDeviceToHost, st2);
            cudaMemcpyAsync(h_right, d0_right, tpb*sizeof(REALthree), cudaMemcpyDeviceToHost, st3);

            cudaStreamSynchronize(st2);
            cudaStreamSynchronize(st3);

            // CPU Part Start -----

            for (int k=0; k<tpb; k++)  readIn(tmpr, h_right, h_left, k, k);

            CPU_diamond(tmpr, htcpu);

            for (int k=0; k<tpb; k++)  writeOutLeft(tmpr, h_right, h_left, k, k, 0);

            cudaMemcpyAsync(d2_right, h_right, tpb*sizeof(REALthree), cudaMemcpyHostToDevice, st2);
            cudaMemcpyAsync(d2_left + cpuLoc, h_left, tpb*sizeof(REALthree), cudaMemcpyHostToDevice, st3);

            // CPU Part End -----

            // Automatic synchronization with memcpy in default stream

            //Split Diamond End------

            t_eq += t_fullstep;

    	    if (t_eq > twrite)
    		{
    			downTriangle <<<bks, tpb, smem>>> (d_IC, d2_right, d2_left);

    			cudaMemcpy(T_f, d_IC, sizeof(REALthree)*dv, cudaMemcpyDeviceToHost);

                fwr << "Density " << t_eq << " ";
                for (int k = 1; k<(dv-1); k++) fwr << T_f[k].x << " ";
                fwr << endl;

                fwr << "Velocity " << t_eq << " ";
                for (int k = 1; k<(dv-1); k++) fwr << (T_f[k].y/T_f[k].x) << " ";
                fwr << endl;

                fwr << "Energy " << t_eq << " ";
                for (int k = 1; k<(dv-1); k++) fwr << energy(T_f[k]) << " ";
                fwr << endl;

                fwr << "Pressure " << t_eq << " ";
                for (int k = 1; k<(dv-1); k++) fwr << pressure(T_f[k]) << " ";
                fwr << endl;

                upTriangle <<<bks, tpb, smem>>> (d_IC, d0_right, d0_left);

    			splitDiamond <<<bks, tpb, smem>>> (d0_right, d0_left, d2_right, d2_left);

                t_eq += t_fullstep;

                twrite += freq;
    		}
        }

        cudaFreeHost(h_right);
        cudaFreeHost(h_left);
        cudaStreamDestroy(st1);
        cudaStreamDestroy(st2);
        cudaStreamDestroy(st3);
        free(tmpr);

	}
    else
    {
        cout << "GPU only Swept scheme" << endl;
        splitDiamond <<<bks, tpb, smem>>> (d0_right, d0_left, d2_right, d2_left);
        t_eq = t_fullstep;

        while(t_eq < t_end)
        {

            wholeDiamond <<<bks, tpb, smem>>> (d2_right, d2_left, d0_right, d0_left, false);

            splitDiamond <<<bks, tpb, smem>>> (d0_right, d0_left, d2_right, d2_left);
            //So it always ends on a left pass since the down triangle is a right pass.
            t_eq += t_fullstep;

            if (t_eq > twrite)
    		{
    			downTriangle <<<bks, tpb, smem>>> (d_IC, d2_right, d2_left);

    			cudaMemcpy(T_f, d_IC, sizeof(REALthree)*dv, cudaMemcpyDeviceToHost);

                fwr << "Density " << t_eq << " ";
                for (int k = 1; k<(dv-1); k++) fwr << T_f[k].x << " ";
                fwr << endl;

                fwr << "Velocity " << t_eq << " ";
                for (int k = 1; k<(dv-1); k++) fwr << (T_f[k].y/T_f[k].x) << " ";
                fwr << endl;

                fwr << "Energy " << t_eq << " ";
                for (int k = 1; k<(dv-1); k++) fwr << energy(T_f[k]) << " ";
                fwr << endl;

                fwr << "Pressure " << t_eq << " ";
                for (int k = 1; k<(dv-1); k++) fwr << pressure(T_f[k]) << " ";
                fwr << endl;

    			upTriangle <<<bks, tpb, smem>>> (d_IC, d0_right, d0_left);

    			splitDiamond <<<bks, tpb, smem>>> (d0_right, d0_left, d2_right, d2_left);

                t_eq += t_fullstep;

    			twrite += freq;
    		}
        }
    }

    downTriangle <<<bks, tpb, smem>>> (d_IC, d2_right, d2_left);

	cudaMemcpy(T_f, d_IC, sizeof(REALthree)*dv, cudaMemcpyDeviceToHost);

	cudaFree(d_IC);
	cudaFree(d0_right);
	cudaFree(d0_left);
    cudaFree(d2_right);
	cudaFree(d2_left);

    return t_eq;
}

int main( int argc, char *argv[] )
{
    //That is, there are less than 8 arguments.
    if (argc < 8)
	{
		cout << "The Program takes 8 inputs, #Divisions, #Threads/block, deltat, finish time, output frequency..." << endl;
        cout << "Algorithm type, Variable Output File, Timing Output File (optional)" << endl;
		exit(-1);
	}
    cout.precision(10);

	// Choose the GPGPU.  This is device 0 in my machine which has 2 devices.
	cudaSetDevice(0);
    if (sizeof(REAL)>6) cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    dimz.gam = 1.4;
    dimz.mgam = 0.4;

    bd[0].x = ONE; //Density
    bd[1].x = 0.125;
    bd[0].y = ZERO; //Velocity
    bd[1].y = ZERO;
    bd[0].z = ONE/dimz.mgam; //Energy
    bd[1].z = 0.1/dimz.mgam;


    const int dv = atoi(argv[1]); //Number of spatial points
	const int tpb = atoi(argv[2]); //Threads per Block
    const double dt = atof(argv[3]);
	const double tf = atof(argv[4]) - QUARTER*dt; //Finish time
    const double freq = atof(argv[5]);
    const int scheme = atoi(argv[6]); //2 for Alternate, 1 for GPUShared, 0 for Classic
    const int bks = dv/tpb; //The number of blocks
    const double dx = lx/((REAL)dv-TWO);
    char const *prec;
    prec = (sizeof(REAL)<6) ? "Single": "Double";

    //Declare the dimensions in constant memory.
    dimz.dt_dx = dt/dx; // dt/dx
    dimz.base = tpb+4;
    dimz.idxend = dv-1;
    dimz.idxend_1 = dv-2;

    for (int k=-2; k<3; k++) dimz.hts[k+2] = (tpb/2) + k;

    cout << "Euler --- #Blocks: " << bks << " | Length: " << lx << " | Precision: " << prec << " | dt/dx: " << dimz.dt_dx << endl;

	//Conditions for main input.
	//dv and tpb must be powers of two.  dv must be larger than tpb and divisible by tpb.

	if ((dv & (tpb-1) !=0) || (tpb&31) != 0)
    {
        cout << "INVALID NUMERIC INPUT!! "<< endl;
        cout << "2nd ARGUMENT MUST BE A POWER OF TWO >= 32 AND FIRST ARGUMENT MUST BE DIVISIBLE BY SECOND" << endl;
        exit(-1);
    }

    if (dimz.dt_dx > .21)
    {
        cout << "The value of dt/dx (" << dimz.dt_dx << ") is too high.  In general it must be <=.21 for stability." << endl;
        exit(-1);
    }

	// Initialize arrays.
    REALthree *IC, *T_final;
	cudaHostAlloc((void **) &IC, dv*sizeof(REALthree), cudaHostAllocDefault);
	cudaHostAlloc((void **) &T_final, dv*sizeof(REALthree), cudaHostAllocDefault);

	for (int k = 0; k<dv; k++) IC[k] = (k<dv/2) ? bd[0] : bd[1];

	// Call out the file before the loop and write out the initial condition.
	ofstream fwr;
	fwr.open(argv[7],ios::trunc);
    fwr.precision(10);

	// Write out x length and then delta x and then delta t.
	// First item of each line is variable second is timestamp.
	fwr << lx << " " << (dv-2) << " " << dx << " " << endl;

    fwr << "Density " << 0 << " ";
    for (int k = 1; k<(dv-1); k++) fwr << IC[k].x << " ";
    fwr << endl;

    fwr << "Velocity " << 0 << " ";
    for (int k = 1; k<(dv-1); k++) fwr << IC[k].y << " ";
    fwr << endl;

    fwr << "Energy " << 0 << " ";
    for (int k = 1; k<(dv-1); k++) fwr << IC[k].z/IC[k].x << " ";
    fwr << endl;

    fwr << "Pressure " << 0 << " ";
    for (int k = 1; k<(dv-1); k++) fwr << pressure(IC[k]) << " ";
    fwr << endl;

    //Transfer data to GPU.
	// This puts the Fourier number in constant memory.
	cudaMemcpyToSymbol(dimens,&dimz,sizeof(dimensions));
    cudaMemcpyToSymbol(dbd,&bd,2*sizeof(REALthree));

    // Start the counter and start the clock.
    cudaEvent_t start, stop;
	float timed;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
	cudaEventRecord( start, 0);

    // Call the correct function with the correct algorithm.
    cout << scheme << " " ;
    double tfm;
    if (scheme)
    {
        tfm = sweptWrapper(bks, tpb, dv, dt, tf, scheme-1, IC, T_final, freq, fwr);
    }
    else
    {
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

    if (argc>7)
    {
        ofstream ftime;
        ftime.open(argv[8],ios::app);
    	ftime << dv << "\t" << tpb << "\t" << per_ts << endl;
    	ftime.close();
    }

	fwr << "Density " << tfm << " ";
	for (int k = 1; k<(dv-1); k++) fwr << T_final[k].x << " ";
    fwr << endl;

    fwr << "Velocity " << tfm << " ";
	for (int k = 1; k<(dv-1); k++) fwr << T_final[k].y/T_final[k].x << " ";
    fwr << endl;

    fwr << "Energy " << tfm << " ";
    for (int k = 1; k<(dv-1); k++) fwr << energy(T_final[k]) << " ";
    fwr << endl;

    fwr << "Pressure " << tfm << " ";
    for (int k = 1; k<(dv-1); k++) fwr << pressure(T_final[k]) << " ";
    fwr << endl;

	fwr.close();

    cudaDeviceSynchronize();

	cudaEventDestroy( start );
	cudaEventDestroy( stop );
    cudaDeviceReset();
    cudaFreeHost(IC);
    cudaFreeHost(T_final);

	return 0;
}
