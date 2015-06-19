#include "reorder.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <omp.h>

void forwardReorder(int nodeNum, vector< Edge > &edge){
    thrust::host_vector< ForwardNode > node(nodeNum);
    #pragma omp parallel for
    for(int i = 0; i < nodeNum; i++){
        node[i].oriOrder = i;
        node[i].realDeg = 0;
    }

    vector< Edge >::iterator e = edge.begin();
    for(; e != edge.end(); ++e){
        node[e->u].realDeg++;
        node[e->v].realDeg++;
    }

    thrust::device_vector< ForwardNode > d_node = node;
    thrust::sort(d_node.begin(), d_node.end());
    thrust::copy(d_node.begin(), d_node.end(), node.begin());

    int *newOrder = new int[nodeNum];
    #pragma omp parallel for
    for(int i = 0; i < nodeNum; i++)
        newOrder[node[i].oriOrder] = i;

    int edgeNum = (int)edge.size();
    #pragma omp parallel for
    for(int i = 0; i < edgeNum; i++){
        int newU = newOrder[edge[i].u];
        int newV = newOrder[edge[i].v];
        if(newU < newV) edge[i].u=newU, edge[i].v=newV;
        else edge[i].u=newV, edge[i].v=newU;
    }
}

