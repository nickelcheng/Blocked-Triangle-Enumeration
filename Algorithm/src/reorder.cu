#include "reorder.h"
#include "tool.h"
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <omp.h>

void gForwardReorder(int nodeNum, vector< Edge > &edge){
    thrust::device_vector< ForwardNode > d_node(nodeNum);
    ForwardNode *pd_node = thrust::raw_pointer_cast(d_node.data());
    int nodeBlock = nodeNum/1024;
    int nodeThread = (nodeNum<1024) ? nodeNum : 1024;
    if(nodeNum % 1024 != 0) nodeBlock++;

    initNode<<< nodeBlock, nodeThread >>>(nodeNum, pd_node);

    cudaStream_t cs[STREAM_NUM];
    Edge *d_edge[STREAM_NUM];
    for(int i = 0; i < STREAM_NUM; i++){
        cudaStreamCreate(&cs[i]);
        cudaMalloc((void**)&d_edge[i], sizeof(Edge)*EDGE_UNIT);
    }

    int totalEdge = (int)edge.size(), edgeNum = 0;
    for(int usedEdge = 0, i = 0; totalEdge > 0; usedEdge+=edgeNum, totalEdge-=edgeNum, i++){
        edgeNum = (totalEdge > EDGE_UNIT)? EDGE_UNIT : totalEdge;

        int edgeBlock = edgeNum/1024;
        int edgeThread = (edgeNum<1024) ? edgeNum : 1024;
        if(edgeNum % 1024 != 0) edgeBlock++;

        cudaMemcpyAsync(d_edge[i%STREAM_NUM], &edge[usedEdge], sizeof(Edge)*edgeNum, H2D, cs[i%STREAM_NUM]);
        countDeg<<< edgeBlock, edgeThread, 0, cs[i%STREAM_NUM] >>>(d_edge[i%STREAM_NUM], edgeNum, pd_node);
    }

    for(int i = 0; i < STREAM_NUM; i++){
        cudaStreamDestroy(cs[i]);
        cudaFree(d_edge[i]);
    }

    thrust::sort(d_node.begin(), d_node.end());

    int newOrder[nodeNum], *d_newOrder;
    cudaMalloc((void**)&d_newOrder, sizeof(int)*nodeNum);
    setNewOrder<<< nodeBlock, nodeThread >>>(pd_node, nodeNum, d_newOrder);
    cudaMemcpy(newOrder, d_newOrder, sizeof(int)*nodeNum, D2H);
    cudaFree(d_newOrder);

    edgeNum = (int)edge.size();
    #pragma omp parallel for
    for(int i = 0; i < edgeNum; i++){
        int newU = newOrder[edge[i].u];
        int newV = newOrder[edge[i].v];
        if(newU < newV) edge[i].u=newU, edge[i].v=newV;
        else edge[i].u=newV, edge[i].v=newU;
    }
}

__global__ void initNode(int nodeNum, ForwardNode *node){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int threads = blockDim.x * gridDim.x;

    for(int i = idx; i < nodeNum; i+=threads){
        node[i].order = i;
        node[i].realDeg = 0;
    }
}

__global__ void countDeg(Edge *edge, int edgeNum, ForwardNode *node){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int threads = blockDim.x * gridDim.x;

    for(int i = idx; i < edgeNum; i+=threads){
        atomicAdd(&(node[edge[i].u].realDeg), 1);
        atomicAdd(&(node[edge[i].v].realDeg), 1);
    }
}

__global__ void setNewOrder(const ForwardNode *node, int nodeNum, int *newOrder){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int threads = blockDim.x * gridDim.x;

    for(int i = idx; i < nodeNum; i+=threads){
        newOrder[node[i].order] = i;
    }
}
