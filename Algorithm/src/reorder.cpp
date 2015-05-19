#include "reorder.h"
#include "struct.h"
#include<cstring>
#include<cstdio>

void forwardReorder(int nodeNum, vector< Edge > &edge){
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
}

