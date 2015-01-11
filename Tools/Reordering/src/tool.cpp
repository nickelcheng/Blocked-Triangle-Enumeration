#include "tool.h"
#include <cstdio>
#include <cstdlib>
#include <algorithm>

void initSetting(
    int argc, char *argv[],
    char *srcFile, char *tarFile,
    char *nodeMapFile,
    int &blockSize
){
    checkArgs(argc);
    sprintf(srcFile, "%s", argv[1]);
    sprintf(tarFile, "%s", argv[2]);
    sprintf(nodeMapFile, "%s", argv[3]);
    blockSize = atoi(argv[4]);
}

void checkArgs(int argc){
    if(argc != 5)
        throw("usage: reorder <src_file> <tar_file> <node_map_file> <block_size>\n");
}

void initNodeOriOrder(int nodeNum, vector< Node > &node){
    node.resize(nodeNum);
    for(int i = 0; i < nodeNum; i++){
        node[i].oriOrder = i;
    }
}

void sortNewEdge(vector< Edge > &edge){
    int edgeNum = (int)edge.size();
    for(int i = 0; i < edgeNum; i++){
        if(edge[i].u > edge[i].v){
            int tmp = edge[i].u;
            edge[i].u = edge[i].v;
            edge[i].v = tmp;
        }
    }
    sort(edge.begin(), edge.end());
}

void updateNodeOrder(int nodeRange, vector< Node > &node){
    for(int i = 0; i < nodeRange; i++){
        int tar = node[i].nextOrder;
        int currOri = node[i].oriOrder;
        while(tar != UNDEF){
            int tmp = node[tar].oriOrder;
            node[tar].oriOrder = currOri;
            currOri = tmp;

            tmp = node[tar].nextOrder;
            node[tar].nextOrder = UNDEF;
            tar = tmp;
        }
    }
}

void updateEdgeEnd(int edgeRange, vector< Node > &node, vector< Edge > &edge){
    int edgeNum = (int)edge.size();
    for(int i = 0; i < edgeNum; i++){
        if(node[edge[i].u].nextOrder != -1)
            edge[i].u = node[edge[i].u].nextOrder;
        if(node[edge[i].v].nextOrder != -1)
            edge[i].v = node[edge[i].v].nextOrder;
    }
}

