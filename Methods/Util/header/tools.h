#ifndef __TOOLS_H__
#define __TOOLS_H__

#include<cmath>
#include<cuda_runtime.h>

#define averageCeil(total,unit) (int)ceil((double)total/unit-0.001)

void initDeviceTriNum(void** d_triNum);
__device__ void sumTriangle(int *triNum, int threadTriNum[]);

#endif
