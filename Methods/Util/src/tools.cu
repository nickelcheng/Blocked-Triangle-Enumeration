#include "tools.h"

void initDeviceTriNum(int* d_triNum){
    int zero = 0;
    cudaMemcpy(d_triNum, &zero, sizeof(int), cudaMemcpyHostToDevice);
}

__device__ void sumTriangle(int *triNum, int threadTriNum[]){
    int tmp = 0;
    for(int i = 0; i < blockDim.x; i++){
        tmp += threadTriNum[i];
    }
    atomicAdd(triNum, tmp);
}
