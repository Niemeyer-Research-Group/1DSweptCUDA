/**
	The equations specific global variables and function prototypes.
*/

#ifndef EULERCF_H
#define EULERCF_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include <mpi.h>

#include <string>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "myVectorTypes.h"
#include "json.hpp"

// We're just going to assume doubles
#define REAL            double
#define REALtwo         double2
#define REALthree       double3
#define MPI_R           MPI_DOUBLE    
#define ZERO            0.0
#define QUARTER         0.25
#define HALF            0.5
#define ONE             1.0
#define TWO             2.0
#define SQUAREROOT(x)   sqrt(x)

#define NSTEPS              4

// Since anyone would need to write a header and functions file, why not just hardwire this.  
// If the user's number of steps isn't a power of 2 use the other one.

#define MODULA(x)           x & (NSTEPS-1)  
// #define MODULA(x)           x % NSTEPS  

#define DIVMOD(x)           (MODULA(x)) >> 1   

/*
	============================================================
	DATA STRUCTURE PROTOTYPE
	============================================================
*/

/*
 We don't need the grid data anymore because the node shapes are much simpler.  
 And do not affect the discretization implementation, only the decomposition.  
 Well the structs aren't accessible like arrays so shit.
*/

using json = nlohmann::json;

//---------------// 
struct eqConsts {
    REAL gammma; // Heat capacity ratio
    REAL mgammma; // 1- Heat capacity ratio
    REAL dt_dx; // deltat/deltax
};

//---------------// 
struct states {
    REALthree Q[2]; // Full Step, Midpoint step state variables
    REAL Pr; // Pressure ratio
};

std::string outVars[4] = {"DENSITY", "VELOCITY", "ENERGY", "PRESSURE"}; //---------------// 

/*
	============================================================
	CUDA GLOBAL VARIABLES
	============================================================
*/
// The boundary points can't be on the device so there's no boundary device array.

__constant__ eqConsts deqConsts;  //---------------// 
eqConsts heqConsts; //---------------// 
REALthree hBound[2]; // Boundary Conditions

/*
	============================================================
	EQUATION SPECIFIC FUNCTIONS
	============================================================
*/

/*
//---------------// 
Means this functions is called from the primary program and therefore must be included BY NAME 
in any equation discretization handled by the software.
*/

const int bks, int tpb, const int dv, const double dt, const double t_end, const int cpu

__host__ REAL density(REALthree subj);

__host__ REAL velocity(REALthree subj);

__host__ REAL energy(REALthree subj);

__device__ __host__ 
__forceinline__ REAL pressure(REALthree qH);

__host__ void printout(const int i, REALthree subj); //---------------//

_host__ void equationSpecificArgs(json inJ) //---------------//

__host__ states initialState(json icMaker); //---------------//

__host__ void mpi_type(MPI_Datatype *dtype); //---------------//

__device__ __host__ 
__forceinline__ REAL pressureRoe(REALthree qH);

__device__ __host__ 
__forceinline__ void pressureRatio(states *state, int idx, int tstep);

__device__ __host__ 
__forceinline__ REALthree limitor(REALthree qH, REALthree qN, REAL pRatio);

__device__ __host__ 
__forceinline__ REALthree eulerFlux(REALthree qL, REALthree qR);

__device__ __host__ 
__forceinline__ REALthree eulerSpectral(REALthree qL, REALthree qR);

__device__ __host__ 
__forceinline__ void eulerStep(states *state, int idx, int tstep);

__device__ __host__ 
__forceinline__ void stepUpdate(states *state, int idx, int tstep); //---------------//

__global__ void classicStep(states *state, int tstep);

double classicWrapper(states *state, double *xpts, int *tstep);

__global__ void upTriangle(states *state, int tstep);

__global__ void downTriangle(states *state, int tstep);

__global__ void wholeDiamond(states *state, int tstep);

double sweptWrapper(states *state, double *xpts, int *tstep);

#endif
