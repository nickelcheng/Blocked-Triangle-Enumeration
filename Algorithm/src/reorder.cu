#include "reorder.h"
#include <thrust/sort.h>
#include <omp.h>

void gForwardReorder(int nodeNum, vector< Edge > &edge){
    ForwardNode *node = new ForwardNode[nodeNum];
    #pragma omp parallel for
    for(int i = 0; i < nodeNum; i++){
        node[i].order = i;
        node[i].realDeg = 0;
    }

    vector< Edge >::iterator e = edge.begin();
    for(; e != edge.end(); ++e){
        node[e->u].realDeg++;
        node[e->v].realDeg++;
    }

    thrust::sort(node, node+nodeNum);

    int *newOrder = new int[nodeNum];
    #pragma omp parallel for
    for(int i = 0; i < nodeNum; i++)
        newOrder[node[i].order] = i;

    int edgeNum = (int)edge.size();
    #pragma omp parallel for
    for(int i = 0; i < edgeNum; i++){
        int newU = newOrder[edge[i].u];
        int newV = newOrder[edge[i].v];
        if(newU < newV) edge[i].u=newU, edge[i].v=newV;
        else edge[i].u=newV, edge[i].v=newU;
    }

    delete [] node;
    delete [] newOrder;
}

