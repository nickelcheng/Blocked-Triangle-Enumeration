#include<cstdio>

int main(){
    int deviceNum;
    cudaGetDeviceCount(&deviceNum);
    printf("total %d cards\n", deviceNum);
    for(int i = 0; i < deviceNum; i++){
        struct cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("card %d: %s\n", i, prop.name);
    }

    return 0;
}
