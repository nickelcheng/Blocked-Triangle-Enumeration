#include "forward.h"

void reorderByDegree(vector< Node > &node, vector< Edge > &edge){
    int nodeNum = (int)node.size();
    int edgeNum = (int)edge.size();
    vector< vector< int > > degList(nodeNum);

    // count degree for each node
    for(int i = 0; i < edgeNum; i++){
        node[edge[i].u].realDeg++;
        node[edge[i].v].realDeg++;
    }
    // reorder by counting sort
    for(int i = 0; i < nodeNum; i++){
        degList[node[i].realDeg].push_back(i);
    }
    for(int i = 0, deg = 0; deg < nodeNum; deg++){
        for(int j = 0; j < (int)degList[deg].size(); j++){
            node[degList[deg][j]].newOrder = i++;
        }
    }
}

