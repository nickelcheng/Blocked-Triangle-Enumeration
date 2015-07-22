#include "degeneracy.h"
#include <cstdio>

void reorderByDegeneracy(int nodeNum, vector< Edge > &edge, bool reorder){
    int edgeNum = (int)edge.size();
    if(reorder){
        vector< DegeneracyNode > node(nodeNum);
        vector< DegList > degList;

        // build adj list for each node
        for(int i = 0; i < edgeNum; i++){
            node[edge[i].u].addNei(edge[i].v);
            node[edge[i].v].addNei(edge[i].u);
        }
        // reordering
        degList.resize(nodeNum);
        buildDegList(node, degList);
        reordering(node, degList);
        for(int i = 0; i < edgeNum; i++){
            int u = edge[i].u, v = edge[i].v;
            edge[i].u = node[u].newOrder;
            edge[i].v = node[v].newOrder;
        }
    }
    for(int i = 0; i < edgeNum; i++){
        int u = edge[i].u, v = edge[i].v;
        if(u > v) edge[i].u = v, edge[i].v = u;
    }
}

void buildDegList(vector< DegeneracyNode > &node, vector< DegList > &degList){
    int nodeNum = (int)node.size();
    for(int i = 0; i < nodeNum; i++){
        node[i].inListPos = (int)degList[node[i].degree()].mem.size();
        degList[node[i].degree()].mem.push_back(i);
    }
}

void reordering(vector< DegeneracyNode > &node, vector< DegList > &degList){
    int nodeNum = (int)node.size();
    int currPos = 0;

    for(int i = 0; i < nodeNum; i++){
        node[i].leftDeg = node[i].degree();
    }

    for(int i = 0; i < nodeNum; i++){
        int v = findMinDegNode(currPos, degList);
        removeNode(v, node, degList);
        node[v].newOrder = i;
    }
}

int findMinDegNode(int &currPos, vector< DegList > &degList){
    if(currPos > 0 && !degList[currPos-1].mem.empty())
        currPos--;
    while(degList[currPos].mem.empty()) currPos++;
    return degList[currPos].mem.back();
}

void removeNode(int v, vector< DegeneracyNode > &node, vector< DegList > &degList){
    int deg = (int)node[v].nei.size();
    for(int i = 0; i < deg; i++){
        decDegByOne(node[v].nei[i], node, degList);
    }
    node[v].nei.clear();
    removeNodeInList(v, node, degList);
}

void decDegByOne(int v, vector< DegeneracyNode > &node, vector< DegList > &degList){
    if(node[v].inListPos == -1){
        return;
    }
    removeNodeInList(v, node, degList);
    node[v].leftDeg--;
    node[v].inListPos = (int)degList[node[v].leftDeg].mem.size();
    degList[node[v].leftDeg].mem.push_back(v);
}

void removeNodeInList(int v, vector< DegeneracyNode > &node, vector< DegList > &degList){
    int last = degList[node[v].leftDeg].mem.back();
    degList[node[v].leftDeg].mem[node[v].inListPos] = last;
    node[last].inListPos = node[v].inListPos;
    degList[node[v].leftDeg].mem.pop_back();
    node[v].inListPos = -1;
    
}
