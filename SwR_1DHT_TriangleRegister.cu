// Function file for swept rule, triangles, each triangle is one thread,
// no shared memory.

// Could try to do it all on the GPU so if you can do it with 32 threads,
// it could be nothing but shuffle and register.  Also could try doing shared
// memory and keeping it all on the gpu never touching global.


__global__ void upTriangle(REAL *IC, REAL *right, REAL *left)
{

	//Need pieces of information:
	//sum(2:2:base) (matlabese) of triangle (tA)
	//base of triangle (tB)

	REAL temper[tA];

	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	int tid = threadIdx.x; //Warp or node ID

	//First row of register
	#pragma unroll
	for (int k = 0; k<tB; k++)
	{
		temper[k] = (IC[gid*tB+k]);
	}

	//The initial conditions are timeslice 0 so start k at 1.
	#pragma unroll
	for (int k = 1; k<16; k++)
	{
		//Well there's definitely still some math here.
		for(int n = tB*k; n<tB-(2*k)); n++)
		//Bitwise even odd. On even iterations write to first row.
		shft_wr = (k & 1);
		//On even iterations write to second row (starts at element 32)
		shft_rd = 32*((shft_wr+1) & 1);

		//Each iteration the triangle narrows.  When k = 1, 30 points are
		//computed, k = 2, 28 points.
		if (tid <= (31-k) && tid >= k)
		{
			temper[tid + (32*shft_wr)] = fo * (temper[tid+shft_rd-1] + temper[tid+shft_rd+1]) + (1-2.*fo) * temper[tid+shft_rd];
		}

		//Make sure the threads are synced
		__syncthreads();

		//Now thread 0 in each block (which never computes a value) is used to
		//fill the shared right and left arrays with the relevant values.
		//This grabs the top and bottom edges on the iteration when the top
		//row is written.
		if (shft_wr && tid == 0)
		{
			sL[k+itr] = temper[k-1];
			sL[k+itr+1] = temper[k];
			sL[k+itr+2] = temper[32+k];
			sL[k+itr+3] = temper[33+k];
			sR[k+itr] = temper[31-k];
			sR[k+itr+1] = temper[32-k];
			sR[k+itr+2] = temper[62-k];
			sR[k+itr+3] = temper[63-k];
			itr += 2;
		}

	}

	//After the triangle has been computed, the right and left shared arrays are
	//stored in global memory by the global thread ID since (conveniently),
	//they're the same size as a warp!
	right[gid] = sR[tid];
	left[gid] = sL[tid];

}

//The upside down triangle.  This function essentially takes right and left and
//returns IC.

//IMPORTANT note: k and tid were in sync in the first function, but here they're
//out of sync in the loop.  This is because we can't use tid = 33 or 32 and the
//shared temperature array is that long.  BUT in order to fill the arrays, these
//elements must be accessed.  So each element in each row is shifted by +1.
//For instance, thread tid = 16 refers to temper[17].  That being said, tid is
//unique and k is NOT so the index must be referenced by tid.

__global__ void downTriangle(REAL *IC, REAL *right, REAL *left)
{

	//Now temper needs to accommodate a longer row by 2, one on each side.
	//since it has two rows that's 4 extra floats.  The last row will still be
	//32 numbers long.
	__shared__ REAL temper[68];
	__shared__ REAL sR[32];
	__shared__ REAL sL[32];

	//Same as upTriangle
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	int shft_rd;
	int shft_wr;

	// Pass to the left so all checks are for block 0 (this reduces arithmetic).
	// The left ridge is always kept by the block.
	sR[tid] = left[gid];

	// The right ridge is passed, each block 1-end gets the right of 0-end-1
	// Block 0 gets the right of the last block.
	if (blockIdx.x > 0)
	{
		sL[tid] = right[gid-blockDim.x];
	}
	else
	{
		sL[tid] = right[blockDim.x*(gridDim.x-1) + tid];
	}

	__syncthreads();

	// Initialize temper. Kind of an unrolled for loop.  This is actually at
	// Timestep 0.
	temper[15] = sL[0];
	temper[16] = sL[1];
	temper[17] = sR[0];
	temper[18] = sR[1];

	//Now we need two counters since we need to use sL and sR EVERY iteration
	//instead of every other iteration and instead of growing smaller with every
	//iteration this grows larger.
	int itr = 2;
	int itr2 = 18;
	//k needs to insert the relevant left right values around the computed values
	//every timestep.  Since it grows larger the loop is reversed.
	for (int k = 17; k>1; k--)
	{
		// This tells you if the current row is the first or second.
		shft_wr = (k & 1);
		// Read and write are opposite rows.
		shft_rd = 34*((shft_wr+1) & 1);

		//Block 0 is split so it needs a different algorithm.  This algorithm
		//is slightly different than top triangle as described in the note above.
		if (blockIdx.x > 0)
		{
			if (tid <= (33-k) && tid >= (k-2))
			{
				temper[tid + 1 + (34*shft_wr)] = fo * (temper[tid+shft_rd] + temper[tid+shft_rd+2]) + (1-2.*fo) * temper[tid+shft_rd+1];
			}

		}

		//Split part.  This exhibits thread divergence and is suboptimal.
		//So it's ripe to be improved.

		else
		{
			if (tid <= (33-k) && tid >= (k-2))
			{
				if (tid == 15)
				{
					temper[tid + 1 + (34*shft_wr)] = 2. * fo * (temper[tid+shft_rd]-temper[tid+shft_rd+1]) + temper[tid+shft_rd+1];
				}
				else if (tid == 16)
				{
					temper[tid + 1 + (34*shft_wr)] = 2. * fo * (temper[tid+shft_rd+2]-temper[tid+shft_rd+1]) + temper[tid+shft_rd+1];
				}
				else
				{
					temper[tid + 1 + (34*shft_wr)] = fo * (temper[tid+shft_rd] + temper[tid+shft_rd+2]) + (1-2.*fo) * temper[tid+shft_rd+1];
				}
			}

		}

		//Fill edges.  Thread 0 never gets used for both operations so the
		//calculation and the filling are conceptually coincident.
		//Threads are synced afterward here because the next timestep is
		//reliant on the entire loop.
		if (k>2 && tid == 0)
		{
			temper[(k-3)+(34*shft_wr)] = sL[itr];
			temper[(k-2)+(34*shft_wr)] = sL[itr+1];
			temper[itr2+(34*shft_wr)] = sR[itr];
			itr2++;
			temper[itr2+(34*shft_wr)] = sR[itr+1];
			itr+=2;

		}
		__syncthreads();

	}

	//Now fill the global unified timestep variable with the final calculated
	//temperatures.

	//Blocks 1 to end hold values 16 to end-16.
	if (blockIdx.x > 0)
	{
		//True if it ends on the first row! The first and last of temper on the final row are empty.
		IC[gid - 16] = temper[tid+1];
	}
	//Block 0 holds values 0 to 15 and end-15 to end.  In that order.
	else
	{
		if (tid>15)
		{
			IC[gid - 16] = temper[tid+1];
		}
		else
		{
			IC[(blockDim.x * gridDim.x) + (tid - 16) ] = temper[tid+1];
		}
	}
}
