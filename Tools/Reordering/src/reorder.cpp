#include "reorder.h"
#include "tool.h"

void initNodeNei(int nodeRange, int edgeRange, vector< Node > &node, vector< Edge > &edge){
    for(int i = 0; i < nodeRange; i++){
        node[i].nei.clear();
    }
    for(int i = 0; i < edgeRange; i++){
        int u = edge[i].u, v = edge[i].v;
        node[u].addNei(v);
        node[v].addNei(u);
    }
}

void reorder(int nodeRange, int edgeRange, vector< Node > &node, vector< Edge > &edge){
    DegInfoPQ pq;
    initDegInfoPQ(nodeRange, node, pq);
    for(int i = 0; i < nodeRange; i++){
        int minDegNode = findMinDegNode(pq, node);
        node[minDegNode].nextOrder = i;
        removeNode(minDegNode, node, pq);
    }
    updateEdgeEnd(edgeRange, node, edge);
    updateNodeOrder(nodeRange, node);
}

void initDegInfoPQ(int nodeRange, vector< Node > &node, DegInfoPQ &pq){
    while(!pq.empty()) pq.pop();
    for(int i = 0; i < nodeRange; i++){
        int deg = (int)node[i].nei.size();
        pq.push(DegInfo(deg, i));
    }
}

int findMinDegNode(DegInfoPQ &pq, vector< Node > &node){
    while(!pq.empty() && node[pq.top().nodeID].nextOrder != UNDEF)
        pq.pop();

    if(pq.empty()) throw("Error: pq emtpy\n");

    int tar = pq.top().nodeID;
    pq.pop();
    return tar;
}

void removeNode(int v, vector< Node > &node, DegInfoPQ &pq){
    set< int >::iterator it;
    for(it = node[v].nei.begin(); it != node[v].nei.end(); ++it){
        node[*it].removeNei(v);
        updateDegInfoPQ(*it, node, pq);
    }
    node[v].nei.clear();
}

void updateDegInfoPQ(int v, vector< Node > &node, DegInfoPQ &pq){
    int deg = (int)node[v].nei.size();
    pq.push(DegInfo(deg, v));
}



int removeOutRangeEdge(int nodeRange, int edgeRange, vector< Edge > &edge){
    for(int i = 0; i < edgeRange; i++){
        if(edge[i].outOfRange(nodeRange)){
            edgeRange--;
            int tmp = edge[edgeRange].u;
            edge[edgeRange].u = edge[i].u;
            edge[i].u = tmp;
            tmp = edge[edgeRange].v;
            edge[edgeRange].v = edge[i].v;
            edge[i].v = tmp;
            i--;
        }
    }
    return edgeRange;
}

