#include "vertexCover.h"

void vertexCover(vector< Node > &node, vector< Edge > &edge){
    int edgeNum = (int)edge.size();
    for(int i = 0; i < edgeNum; i++){
        int smaller = edge[i].u;
        if(edge[i].u > edge[i].v)
            smaller = edge[i].v;
        node[smaller].inCoverSet = true;
    }
}
