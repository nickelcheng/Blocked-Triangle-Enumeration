#include "edgeIterator.h"

void reorderByDegeneracy(vector< Node > &node, vector< Edge > &edge){
    int nodeNum = (int)node.size();
    int edgeNum = (int)edge.size();
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
}

void buildDegList(vector< Node > &node, vector< DegList > &degList){
    int nodeNum = (int)node.size();
    for(int i = 0; i < nodeNum; i++){
        node[i].inListPos = (int)degList[node[i].degree()].mem.size();
        degList[node[i].degree()].mem.push_back(i);
    }
}

void reordering(vector< Node > &node, vector< DegList > &degList){
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

void removeNode(int v, vector< Node > &node, vector< DegList > &degList){
    int deg = (int)node[v].nei.size();
    for(int i = 0; i < deg; i++){
        decDegByOne(node[v].nei[i], node, degList);
    }
    node[v].nei.clear();
    removeNodeInList(v, node, degList);
}

void decDegByOne(int v, vector< Node > &node, vector< DegList > &degList){
    if(node[v].inListPos == -1){
        return;
    }
    removeNodeInList(v, node, degList);
    node[v].leftDeg--;
    node[v].inListPos = (int)degList[node[v].leftDeg].mem.size();
    degList[node[v].leftDeg].mem.push_back(v);
}

void removeNodeInList(int v, vector< Node > &node, vector< DegList > &degList){
    int last = degList[node[v].leftDeg].mem.back();
    degList[node[v].leftDeg].mem[node[v].inListPos] = last;
    node[last].inListPos = node[v].inListPos;
    degList[node[v].leftDeg].mem.pop_back();
    node[v].inListPos = -1;
    
}
