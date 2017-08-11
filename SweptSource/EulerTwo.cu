/*
    The equation specific functions.
*/

#include "EulerCF.h"

/**
    Calculates the pressure at the current spatial point with the (x,y,z) rho, u * rho, e *rho state variables.
    
    Calculates pressure from working array variables.  Pressure is not stored outside procedure to save memory.
    @param current  The state variables at current node
    @return Pressure at subject node
*/
#ifdef __CUDA_ARCH__
    #define DIMS    deqConsts
#else
    #define DIMS    heqConsts
#endif

__host__ REAL density(REALthree subj)
{
    return subj.x;
}

__host__ REAL velocity(REALthree subj)
{
    return subj.y/subj.x;
}

__host__ REAL energy(REALthree subj)
{
    REAL u = subj.y/subj.x;
    return subj.z/subj.x - HALF*u*u;
}

__device__ __host__ 
__forceinline__
REAL pressure(REALthree qH)
{
    return DIMS.mgam * (qH.z - (HALF * qH.y * qH.y/qH.x));
}

__host__ REAL printout(const int i, REALthree subj)
{
    switch(i)
    {
        case 0: return density(subj);
        case 1: return velocity(subj):
        case 2: return energy(subj);
        case 3: return pressure(subj);
    } 
}

/*
dimensions heqConsts; //---------------// 
REALthree hBound[2]; // Boundary Conditions
double lx; // Length of domain.
*/

_host__ void equationSpecificArgs(json inJ)
{
    heqConsts.gammma = inJ["gamma"];
    heqConsts.mgammma = heqConsts.gammma - 1;
    REAL rhoL = inJ["rhoL"];
    REAL vL = inJ["vL"];
    REAL pL = inJ["pL"];
    REAL rhoR = inJ["rhoR"];
    REAL vR = inJ["vR"];
    REAL pR = inJ["pR"];
    hBounds[0] = {rhoL, vL*rhoL, pL/heqConsts.mgamma + HALF * rhoL * vL * vL};
    hBounds[1] = {rhoR, vR*rhoR, pR/heqConsts.mgamma + HALF * rhoR * vR * vR};
    REAL dtx = inJ["dt"];
    REAL dxx = inJ["dx"];
    heqConsts.dt_dx = dtx/dxx;
}

// One of the main uses of global variables is the fact that you don't need to pass
// anything so you don't need variable args.
// lxh is half the domain length assuming starting at 0.
__host__ void initialState(REALthree *intl, double xpt, double lx, char ic)
{
    if (ic == "PARTITION")
    {
        int side = (xpt < HALF*lx);
        intl = hBound[side];
    }
}

__host__ void mpi_type(MPI_Datatype *dtype)
{ 
    //double 3 type
    MPI_Datatype vtype;
    MPI_Datatype typs[3] = {MPI_R, MPI_R, MPI_R};
    int n[3] = {1};
    MPI_Aint disp[3] = {0, sizeof(REAL), 2*sizeof(REAL)};

    MPI_Type_struct(3, n, disp, typs, &vtype);
    MPI_Type_commit(&vtype);

    typs[0] = vtype;
    typs[2] = vtype;
    disp[1] = 3*sizeof(vtype);
    disp[2] = 4*sizeof(REAL);

    MPI_Type_struct(3, n, disp, typs, dtype);
    MPI_Type_commit(dtype);

    MPI_Type_free(&vtype);
}

__device__ __host__ 
__forceinline__
REAL pressureRoe(REALthree qH)
{
    return DIMS.mgam * (qH.z - HALF * qH.y * qH.y);
}

/**
    Ratio
*/
__device__ __host__ 
__forceinline__
void pressureRatio(states *state, int idx, int tstep)
{
    state[idx].Pr = (pressure(state[idx+1]->Q[tstep]) - pressure(state[idx]->Q[tstep]))/(pressure(state[idx]->Q[tstep]) - pressure(state[idx-1]->Q[tstep]));
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
REALthree limitor(REALthree qH, REALthree qN, REAL pRatio)
{   
    return (isnan(pRatio) || pRatio<0) ? qH :  (qH + HALF * min(pRatio, ONE) * (qN - qH));
}

/**
    Uses the reconstructed interface values as inputs to flux function F(Q)

    @param qL Reconstructed value at the left side of the interface.
    @param qR  Reconstructed value at the left side of the interface.
    @return  The combined flux from the function.
*/
__device__ __host__ 
__forceinline__ REALthree eulerFlux(REALthree qL, REALthree qR)
{
    REAL uLeft = qL.y/qL.x;
    REAL uRight = qR.y/qR.x;

    REAL pL = pressure(qL);
    REAL pR = pressure(qR);

    REALthree flux;
    flux.x = (qL.y + qR.y);
    flux.y = (qL.y*uLeft + qR.y*uRight + pL + pR);
    flux.z = (qL.z*uLeft + qR.z*uRight + uLeft*pL + uRight*pR);

    return flux;
}

/**
    Finds the spectral radius and applies it to the interface.

    @param qL Reconstructed value at the left side of the interface.
    @param qR  Reconstructed value at the left side of the interface.
    @return  The spectral radius multiplied by the difference of the reconstructed values
*/
__device__ __host__ 
__forceinline__ REALthree eulerSpectral(REALthree qL, REALthree qR)
{
    REALthree halfState;
    REAL rhoLeftsqrt = SQUAREROOT(qL.x);
    REAL rhoRightsqrt = SQUAREROOT(qR.x);

    halfState.x = rhoLeftsqrt * rhoRightsqrt;
    REAL halfDenom = ONE/(halfState.x*(rhoLeftsqrt + rhoRightsqrt));

    halfState.y = (rhoLeftsqrt*qR.y + rhoRightsqrt*qL.y)*halfDenom;
    halfState.z = (rhoLeftsqrt*qR.z + rhoRightsqrt*qL.z)*halfDenom;

    REAL pH = pressureRoe(halfState);

    return (SQUAREROOT(pH * DIMS.gam) + fabs(halfState.y)) * (qL - qR);
}

/**
    The Final step of the finite volume scheme.

    First: The pressure ratio calculation is decomposed to avoid division and calling the limitor unnecessarily.  Although 3 pressure ratios would be required, we can see that there are only 4 unique numerators and denominators in that calculation which can be calculated without using division or calling pressure (which uses division).  The edge values aren't guaranteed to have the correct conditions so the flags set the appropriate pressure values to 0 (Pressures are equal) at the edges.
    Second:  The numerator and denominator are tested to see if the pressure ratio will be Nan or <=0. If they are, the limitor doesn't need to be called.  If they are not, call the limitor and calculate the pressure ratio.
    Third:  Use the reconstructed values at the interfaces to get the flux at the interfaces using the spectral radius and flux functions and combine the results with the flux variable.
    Fourth: Repeat for second interface and update current volume. 

    @param state  Reference to the working array in SHARED memory holding the dependent variables.
    @param idx  The indices of the stencil points.
    @param flagLeft  True if the point is the first finite volume in the tube.
    @param flagRight  True if the point is the last finite volume in the tube.
    @return  The updated value at the current spatial point.
*/
__device__ __host__ void eulerStep(states *state, int idx, int tstep)
{
    REALthree tempStateLeft, tempStateRight;

    tempStateLeft = limitor(state[idx-1].Q[tstep], state[idx].Q[tstep], state[idx-1].Pr);
    tempStateRight = limitor(state[idx].Q[tstep], state[idx-1].Q[tstep], ONE/state[idx].Pr);
    REALthree flux = eulerFlux(tempStateLeft,tempStateRight);
    flux += eulerSpectral(tempStateLeft,tempStateRight);

    tempStateLeft = limitor(state[idx].Q[tstep], state[idx+1].Q[tstep], state[idx].Pr);
    tempStateRight = limitor(state[idx+1].Q[tstep], state[idx].Q[tstep], ONE/state[idx+1].Pr);
    flux -= eulerFlux(tempStateLeft,tempStateRight);
    flux -= eulerSpectral(tempStateLeft,tempStateRight);

    state[idx].Q[tstep] = state[idx].Q[0] + ((QUARTER * (tstep+1)) * DIMS.dt_dx * flux);
}

__device__ __host__ 
__forceinline__ void stepUpdate(states *state, int idx, int tstep)
{
    if (tstep & 1) //Odd 0 for even numbers
    {
        pressureRatio(state, idx, DIVMOD(tstep));
    }
    else
    {
        eulerStep(state, idx, DIVMOD(tstep));
    }
}


__global__ void classicStep(states *state, int ts)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x + 1; //Global Thread ID (one extra)

    stepUpdate(state, gid, ts)
}

__global__
void
upTriangle(states *state, int tstep)
{
	extern __shared__ states temper[];

	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	int tidx = threadIdx.x; //Block Thread ID
    int mid = blockDim.x >> 1;

    // Using tidx as tid is kind of confusing for reader but looks valid.

	temper[tidx] = state[gid + 1];

    __syncthreads();

    #pragma unroll
	for (int k=1; k<mid; k++)
	{
		if (tidx < (blockDim.x-k) && tidx >= k)
		{
            stepupdate(temper, tidx, tstep + k); 
		}
		__syncthreads();
	}
    state[gid + 1] = temper[tidx]
}

/**
    Builds an inverted triangle using the swept rule.

    Inverted triangle using the swept rule.  downTriangle is only called at the end when data is passed left.  It's never split.  Sides have already been passed between nodes, but will be swapped and parsed by readIn function.

    @param IC Full solution at some timestep.
    @param inRight Array of right edges seeding solution vector.
*/
__global__
void
downTriangle(states *state, int tstep)
{
	extern __shared__ states temper[];

    int tid = threadIdx.x; // Thread index
    int mid = blockDim.x >> 1; // Half of block size
    int base = blockDim.x + 2; 
	int gid = blockDim.x * blockIdx.x + tid; 
    int tidx = tid + 1;

    if (tid<2) temper[tid] = state[gid]; 
	temper[tid+2] = state[gid + 2];
    
    __syncthreads();

    #pragma unroll
	for (int k=mid; k>0; k--)
	{
		if (tidx < (base-k) && tidx >= k)
		{
            stepupdate(temper, tidx, tstep + k);
		}
		__syncthreads();
	}
    state[gid] = temper[tidx]
}


/**
    Builds an diamond using the swept rule after a left pass.

    Unsplit diamond using the swept rule.  wholeDiamond must apply boundary conditions only at it's center.
*/
__global__
void
wholeDiamond(states *state, int tstep)
{
	extern __shared__ states temper[];

    int tid = threadIdx.x; // Thread index
    int mid = (blockDim.x >> 1); // Half of block size
    int base = blockDim.x + 2; 
	int gid = blockDim.x * blockIdx.x + tid; 
    int tidx = tid + 1;

    if (tid<2) temper[tid] = state[gid]; 
	temper[tid + 2] = state[gid + 2];
    
    __syncthreads();

    #pragma unroll
	for (int k=mid; k>0; k--)
	{
		if (tidx < (base-k) && tidx >= k)
		{
            stepupdate(temper, tidx, tstep + k);
		}
		__syncthreads();
	}

    #pragma unroll
	for (int k=2; k<=mid; k++)
	{
		if (tidx < (base-k) && tidx >= k)
		{
            stepupdate(temper, tidx, tstep + k);
		}
		__syncthreads();
	}
    state[gid + 1] = temper[tidx]
}

//Parameters are straighforward and taken directly from inputs to program.  Wrapper that clls the classic procedure.
double
classicWrapper(const int bks, int tpb, const int dv, const double dt, const double t_end,
    REALthree *IC, REALthree *T_f, const double freq, ofstream &fwr)
{
    REALthree *dEuler_in, *dEuler_out;

    //Allocate device arrays.
    cudaMalloc((void **)&dEuler_in, sizeof(REALthree)*dv);
    cudaMalloc((void **)&dEuler_out, sizeof(REALthree)*dv);

    // Copy the initial conditions to the device array.
    cudaMemcpy(dEuler_in,IC,sizeof(REALthree)*dv,cudaMemcpyHostToDevice);

    //Print to make sure we're here
    cout << "Classic scheme" << endl;

    //Start the timer (simulation timer that is.)
    double t_eq = 0.0;
    double twrite = freq - QUARTER*dt;

    //Call the kernel to step forward alternating global arrays with each call.
    while (t_eq < t_end)
    {
        classicEuler <<< bks,tpb >>> (dEuler_in, dEuler_out, false);
        classicEuler <<< bks,tpb >>> (dEuler_out, dEuler_in, true);
        t_eq += dt;

        //If multiple timesteps should be written out do so here with the file provided.
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

//The wrapper that enacts the swept rule.
double
sweptWrapper(REALthree *IC, REALthree *T_f, const double freq, ofstream &fwr)
{
    const size_t smem = (2*dimz.base)*sizeof(REALthree); //Amt of shared memory to request
    const int cpuLoc = dv-tpb; //Where to write the cpu values back to device memory to make swap if hybrid.

    // CPU mask values for boundary.
    int htcpu[5];
    for (int k=0; k<5; k++) htcpu[k] = dimz.hts[k]+2;

	REALthree *d_IC, *d0_right, *d0_left, *d2_right, *d2_left;

    // Allocate device global memory
	cudaMalloc((void **)&d_IC, sizeof(REALthree)*dv);
	cudaMalloc((void **)&d0_right, sizeof(REALthree)*dv);
	cudaMalloc((void **)&d0_left, sizeof(REALthree)*dv);
    cudaMalloc((void **)&d2_right, sizeof(REALthree)*dv);
	cudaMalloc((void **)&d2_left, sizeof(REALthree)*dv);

    // Transfer over the initial conditions.
	cudaMemcpy(d_IC,IC,sizeof(REALthree)*dv,cudaMemcpyHostToDevice);

	// Start the simulation time counter and start the clock.
	const double t_fullstep = 0.25*dt*(double)tpb;

    //Call up first out of loop with right and left 0
	upTriangle <<<bks, tpb, smem>>> (d_IC, d0_right, d0_left);

    double t_eq;
    double twrite = freq - QUARTER*dt;

	// Call the kernels until you reach the final time

    // The hybrid version.
    if (cpu)
    {
        // Tell us what it is and allocate the host arrays.  cudaHostAlloc is pinned memory that transfers faster.
        cout << "Hybrid Swept scheme" << endl;

        REALthree *h_right, *h_left;
        REALthree *tmpr = (REALthree *) malloc(smem);
        cudaHostAlloc((void **) &h_right, tpb*sizeof(REALthree), cudaHostAllocDefault);
        cudaHostAlloc((void **) &h_left, tpb*sizeof(REALthree), cudaHostAllocDefault);

        t_eq = t_fullstep; 

        // Start 3 cuda streams to overlap memory transfer and kernel launch with cpu computation.
        cudaStream_t st1, st2, st3;
        cudaStreamCreate(&st1);
        cudaStreamCreate(&st2);
        cudaStreamCreate(&st3);

        // Split Diamond Begin------

        // Call wholeDiamond on first non-default stream and launch with one missing block.
        wholeDiamond <<<bks-1, tpb, smem, st1>>> (d0_right, d0_left, d2_right, d2_left, true);
        // Simultaneously transfer the right and left values for node1 to the CPU.
        cudaMemcpyAsync(h_left, d0_left, tpb*sizeof(REALthree), cudaMemcpyDeviceToHost, st2);
        cudaMemcpyAsync(h_right, d0_right, tpb*sizeof(REALthree), cudaMemcpyDeviceToHost, st3);

        // Wait for memory to arrive. and read the edges into the working array: tmpr.
        cudaStreamSynchronize(st2);
        cudaStreamSynchronize(st3);

        // CPU Part Start -----

        for (int k=0; k<tpb; k++)  readIn(tmpr, h_right, h_left, k, k);
        // Step forward with temper which now contains only edges 
        CPU_diamond(tmpr, htcpu);
        // And write those values back to the edges.
        for (int k=0; k<tpb; k++)  writeOutLeft(tmpr, h_right, h_left, k, k, 0);
        // Simultaneously write those edges back out to the device passing the left one to the end.
        cudaMemcpyAsync(d2_right, h_right, tpb*sizeof(REALthree), cudaMemcpyHostToDevice, st2);
        cudaMemcpyAsync(d2_left + cpuLoc, h_left, tpb*sizeof(REALthree), cudaMemcpyHostToDevice, st3);

        // No need to explicitly synchronize.  This is implicitly synchronized by calling the next kernel in the default stream.

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
    // Shared (GPU only) version of the swept scheme.
    else
    {
        cout << "GPU only Swept scheme" << endl;
        splitDiamond <<<bks, tpb, smem>>> (d0_right, d0_left, d2_right, d2_left);
        t_eq = t_fullstep;

        while(t_eq < t_end)
        {

            wholeDiamond <<<bks, tpb, smem>>> (d2_right, d2_left, d0_right, d0_left, false);

            splitDiamond <<<bks, tpb, smem>>> (d0_right, d0_left, d2_right, d2_left);
            //It always ends on a left pass since the down triangle is a right pass.
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
    // The last call is down so call it and pass the relevant data to the host with memcpy.
    downTriangle <<<bks, tpb, smem>>> (d_IC, d2_right, d2_left);

	cudaMemcpy(T_f, d_IC, sizeof(REALthree)*dv, cudaMemcpyDeviceToHost);

	cudaFree(d_IC);
	cudaFree(d0_right);
	cudaFree(d0_left);
    cudaFree(d2_right);
	cudaFree(d2_left);

    return t_eq;
}
