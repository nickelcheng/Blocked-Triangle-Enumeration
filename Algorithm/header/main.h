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

const int EDGE_NUM_LIMIT = 250*1024*1024;

using namespace std;

typedef struct Node{
    vector< int > nei;
    int degree(void) const{
        return (int)nei.size();
    }
} Node;

typedef struct Edge{
    int u, v;
    Edge(int _u, int _v){
        u = _u, v = _v;
    }
    bool operator < (const Edge &a) const{
        if(u != a.u) return u < a.u;
        return v < a.v;
    }
} Edge;

typedef struct vector< Edge > Row;
typedef struct vector< Row > Matrix;

#endif
