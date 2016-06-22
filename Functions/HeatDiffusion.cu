//

#include <cuda.h>

#include <cmath>
#include <cstdlib>

//fo needs to be in constant memory.

__host__ void initFun(int xnode, REAL ds, REAL lx,REAL result)
{

    result = 500.f*expf((-ds*(REAL)xnode)/lx);

}

__device__ REAL execFunc(REAL tLeft, REAL tRight, REAL tCenter)
{

    return fo*(tLeft+tRight) + (1.f-2.f*fo)*tCenter;

}
