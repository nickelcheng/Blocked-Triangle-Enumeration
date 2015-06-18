#include "reorder.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <omp.h>

void forwardReorder(int nodeNum, vector< Edge > &edge){
    thrust::host_vector< forwardNode > node(nodeNum);
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

    thrust::device_vector< forwardNode > d_node = node;
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

/*void forwardReorderSeq(int nodeNum, vector< Edge > &edge){
    vector< vector< int > > degList(nodeNum);
    vector< forwardNode > node(nodeNum);

    vector< Edge >::iterator e = edge.begin();
    for(; e != edge.end(); ++e){
        node[e->u].realDeg++;
        node[e->v].realDeg++;
    }

    for(int i = 0; i < (int)node.size(); i++){
        degList[node[i].realDeg].push_back(i);
    }

    vector< vector< int > >::iterator deg = degList.begin();
    for(int idx = 0; deg != degList.end(); ++deg){
        vector< int >::iterator n = deg->begin();
        for(; n != deg->end(); ++n){
            node[*n].newOrder = idx++;
        }
    }

    for(e = edge.begin(); e != edge.end(); ++e){
        int newU = node[e->u].newOrder;
        int newV = node[e->v].newOrder;
        if(newU < newV) e->u=newU, e->v=newV;
        else e->u=newV, e->v=newU;
    }
}*/

