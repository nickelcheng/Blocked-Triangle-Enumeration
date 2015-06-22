#include "reorder.h"
#include "main.h"
#include <omp.h>

void cForwardReorder(int nodeNum, vector< Edge > &edge){
    ForwardNode *node = new ForwardNode[nodeNum];
    vector< vector< int > > degList(nodeNum);

    #pragma omp parallel for
    for(int i = 0; i < nodeNum; i++)
        node[i].realDeg = 0;

    vector< Edge >::iterator e = edge.begin();
    for(; e != edge.end(); ++e){
        node[e->u].realDeg++;
        node[e->v].realDeg++;
    }

    for(int i = 0; i < nodeNum; i++){
        degList[node[i].realDeg].push_back(i);
    }

    vector< vector< int > >::iterator deg = degList.begin();
    for(int idx = 0; deg != degList.end(); ++deg){
        vector< int >::iterator n = deg->begin();
        for(; n != deg->end(); ++n){
            node[*n].order = idx++;
        }
    }

    int edgeNum = (int)edge.size();
    #pragma omp parallel for
    for(int i = 0; i < edgeNum; i++){
        int newU = node[edge[i].u].order;
        int newV = node[edge[i].v].order;
        if(newU < newV) edge[i].u=newU, edge[i].v=newV;
        else edge[i].u=newV, edge[i].v=newU;
    }

    delete [] node;
}

