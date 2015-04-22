#include "reorder.h"
#include "io.h"
#include "forward.h"
#include "edgeIterator.h"
#include "vertexCover.h"
#include<cstring>
#include<algorithm>

int reorder(int algo, vector< Node > &node, vector< Edge > &edge){
    if(algo == FORWARD || algo == COVER_FORWARD)
        reorderByDegree(node, edge);
    else if(algo == EDGE_ITERATOR || algo == COVER_EDGE_ITERATOR)
        reorderByDegeneracy(node, edge);
    updateIndex(node, edge);

    if(algo == COVER_FORWARD || algo == COVER_EDGE_ITERATOR || algo == COVER){
        vertexCover(node, edge);
    }
    buildNeiList(node, edge);

    return getMaxDeg(node);
}

void updateIndex(vector< Node > &node, vector< Edge > &edge){
    int edgeNum = (int)edge.size();
    for(int i = 0; i < edgeNum; i++){
        edge[i].u = node[edge[i].u].newOrder;
        edge[i].v = node[edge[i].v].newOrder;
    }
}

void buildNeiList(vector< Node > &node, vector< Edge > &edge){
    int edgeNum = (int)edge.size();
    int nodeNum = (int)node.size();
    for(int i = 0; i < edgeNum; i++){
        int u = edge[i].u, v = edge[i].v;
        if(u < v) node[u].addNei(v);
        else node[v].addNei(u);
    }
    for(int i = 0; i < nodeNum; i++){
        sort(node[i].nei.begin(), node[i].nei.end());
    }
}

int getMaxDeg(vector< Node > &node){
    int nodeNum = (int)node.size();
    int maxDeg = 0;
    for(int i = 0; i < nodeNum; i++){
        if(node[i].degree() > maxDeg)
            maxDeg = node[i].degree();
    }
    return maxDeg;
}
