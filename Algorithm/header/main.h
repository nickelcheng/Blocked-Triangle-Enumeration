#ifndef __MAIN_H__
#define __MAIN_H__

#include<vector>

#ifdef __NVCC__
#define DECORATE __host__ __device__
#else
#define DECORATE
#endif

#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost

#define GPU_BLOCK_NUM 16000
#define GPU_THREAD_NUM 1024
#define BIT_NUM_TABLE_SIZE 65536
#define BIT_SHIFT_AMT 16

const int EDGE_NUM_LIMIT = 100*1024*1024;

using std::vector;

typedef unsigned char UC;
typedef unsigned int UI;

typedef struct Node{
    vector< int > nei;
    int degree(void) const{
        return (int)nei.size();
    }
} Node;

typedef struct Edge{
    int u, v;
    DECORATE bool operator < (const Edge &a) const{
        if(u != a.u) return u < a.u;
        return v < a.v;
    }
} Edge;

#endif
