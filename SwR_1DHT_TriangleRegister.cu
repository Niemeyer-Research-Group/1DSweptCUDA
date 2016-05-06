// Function file for swept rule, triangles, each triangle is one thread,
// no shared memory.

// Could try to do it all on the GPU so if you can do it with 32 threads,
// it could be nothing but shuffle and register.  Also could try doing shared
// memory and keeping it all on the gpu never touching global.


__global__ void upTriangle(REAL *IC, REAL *right, REAL *left)
{

	//Need pieces of information:
	//base of triangle (tB)  Ok we'll use 2D

	REAL temper[tArea];

	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
	int tid = threadIdx.x; //Warp or node ID

	//First row of register
	#pragma unroll
	for (int k = 0; k<tB; k++)
	{
		temper[k] = (IC[gid*tB+k]);
	}

	//Global to global
	right[gid*tB] = IC[gid*tB];
	right[gid*tB+1] = IC[gid*tB+1];
	left[gid*tB] = IC[ ((gid+1) * tB) - 2];
	left[gid*tB+1] = IC[ ((gid+1) * tB) - 1];

	int iter = tB;
	int iter2 = tB;

	//The initial conditions are timeslice 0 so start k at 1.
	#pragma unroll
	for (int k = 1; k<tB/2; k++)
	{

		for(int n = 0; n < (tB-2*k); n++)
		{
		//Each iteration the triangle narrows.  When k = 1, 30 points are
		//computed, k = 2, 28 points.
			temper[iter2+n] = fo * (temper[(iter2-iter)+n]  + temper[(iter2-iter)+2+n] ) + (1.-2.*fo) * temper[(iter2-iter)+1+n];
		}

		right[gid*tB+2*k] = temper[iter2];
		right[gid*tB+(2*k+1)] = temper[(iter2+1];

		iter -= 2
		iter2 += iter

		// Global memory version
		// Index math is kinda out of contol.
		// Could template this.

		left[gid*tB+2*k] =  temper[iter2-2];
		left[gid*tB+(2*k+1)] =  temper[iter2-1];

	}

	// Ok, let's use shfl next time

}

//The upside down triangle.  This function essentially takes right and left and
//returns IC.

//IMPORTANT note: k and tid were in sync in the first function, but here they're
//out of sync in the loop.  This is because we can't use tid = 33 or 32 and the
//shared temperature array is that long.  BUT in order to fill the arrays, these
//elements must be accessed.  So each element in each row is shifted by +1.
//For instance, thread tid = 16 refers to temper[17].  That being said, tid is
//unique and k is NOT so the index must be referenced by tid.
//
//tArea = tB*(tB/4+.5).  Which is an interesting identity.

__global__ void downTriangle(REAL *IC, REAL *right, REAL *left)
{

	//Hopefully all these will go to the registers and not local memory.
	REAL temper[tArea];
	REAL shR[tB];
	REAL shL[tB];

	//Same as upTriangle
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	// Pass to the left so all checks are for block 0 (this reduces arithmetic).
	// The left ridge is always kept by the block. Max gid = block*grid-1
	#pragma unroll
	for (int k = 0; k<tB/2; k++)
	{
		shR[k] = left[gid*tB+k];
		if (gid > 0)
		{
			shL[k] = right[(gid-1)*tB+k];
		}
		else
		{
			shL[k] = right[(blockDim.x*gridDim.x-1)*tB+k];
		}
	}
	// The right ridge is passed, each block 1-end gets the right of 0-end-1
	// Block 0 gets the right of the last block.
	// Damn thread divergence!

	// Initialize temper. Kind of an unrolled for loop.  This is actually at
	// Timestep 1 and 2.
	if (gid>0)
	{
		temper[0] = fo * (shL[0]  + shR[0]) + (1.-2.*fo) * shL[1];
		temper[1] = fo * (shR[1]  + shL[1]) + (1.-2.*fo) * shR[0];
		temper[2] = fo * (shL[2]  + temper[0]) + (1.-2.*fo) * shL[3];
		temper[3] = fo * (shL[3]  + temper[1]) + (1.-2.*fo) * temper[0];
		temper[4] = fo * (shR[2]  + temper[0]) + (1.-2.*fo) * temper[1];
		temper[5] = fo * (shR[3]  + temper[1]) + (1.-2.*fo) * shR[2];
	}
	else
	{
		temper[0] = 2 * fo * (shL[0]  - shL[1]) +  shL[1];
		temper[1] = 2 * fo * (shR[1]  - shR[0]) +  shR[0];
		temper[2] = fo * (shL[2]  + temper[0]) + (1.-2.*fo) * shL[3];
		temper[3] = 2 * fo * (shL[3]  - temper[0]) +  temper[0];
		temper[4] = 2 * fo * (shR[2]  - temper[1]) +  temper[1];
		temper[5] = fo * (shR[3]  + temper[1]) + (1.-2.*fo) * shR[2];
	}

	//Now we need two counters since we need to use sL and sR EVERY iteration
	//instead of every other iteration and instead of growing smaller with every
	//iteration this grows larger.

	int iter = 6;

	//k needs to insert the relevant left right values around the computed values
	//every timestep.  Since it grows larger the loop is reversed.
	for (int k = 2; k<(tB/2); k++)
	{

		temper[iter] = fo * (shL[2*k]  + temper[iter-2*k]) + (1.-2.*fo) * shL[2*k+1];;
		temper[iter+1] = fo * (shL[2*k+1]  + temper[(iter+1)-2*k]) + (1.-2.*fo) * temper[iter-(2*k)];

		//10                           R4         T4                                           T5
		temper[iter+(2*k)] = fo * (shR[2*k]  + temper[iter-2]) + (1.-2.*fo) * temper[(iter-1)];
		temper[iter+1+(2*k)] = fo * (shR[2*k+1]  + temper[iter-1]) + (1.-2.*fo) * shL[(2*k)];

		for (int n = 2; n < (iter+1); n++)
		{
		//Block 0 is split so it needs a different algorithm.  This algorithm
		//is slightly different than top triangle as described in the note above.
			if (gid > 0)
			{

				temper[iter+n] = fo * (temper[iter-(2*k+2)] + temper[iter-(2*k)] ) + (1-2.*fo) * temper[iter-(2*k+1)];

			}

			else
			{
				//This is a stunning identity.
				if (n == k)
				{
					temper[iter+n]= 2. * fo * (temper[[iter-(2*k+2)]-temper[iter-(2*k+1)]) + temper[iter-(2*k+1)];
				}
				else if (n == k+1)
				{
					temper[iter+n] = 2. * fo * (temper[iter-(2*k)];-temper[iter-(2*k+1)]) + temper[iter-(2*k+1)];
				}
				else
				{
					temper[iter+n] = fo * (temper[iter-(2*k+2)] + temper[iter-(2*k)]) + (1-2.*fo) * temper[iter-(2*k+1)];
				}

			}

		}

		iter += 2*k;

	}
	//Now fill the global unified timestep variable with the final calculated
	//temperatures.

	//Blocks 1 to end hold values 16 to end-16.
	if (gid > 0)
	{
		#pragma unroll
		for (int k = 0; k<tB; k++)
		{
			IC[(gid*tB)+k] = temper[(tArea-tB)+k];
		{

	}
	//Block 0 holds values 0 to 15 and end-15 to end.  In that order.
	else
	{
		#pragma unroll
		for (int k = 0; k<tB/2; k++)
		{
			IC[(blockDim.x*gridDim.x-1)*tB+k]; = temper[(tArea-tB)+k];
		{
		#pragma unroll
		for (int k = tB/2; k<tB; k++)
		{
			IC[k-tB/2] = temper[(tArea-tB)+k];
		}

	}
}
