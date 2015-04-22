#include "vertexCover.h"

void vertexCover(vector< Node > &node, vector< Edge > &edge){
    int edgeNum = (int)edge.size();
    for(int i = 0; i < edgeNum; i++){
        node[edge[i].u].inCoverSet = true;
    }
}
