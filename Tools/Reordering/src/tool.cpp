#include "tool.h"
#include <cstdio>
#include <cstdlib>
#include <algorithm>

void checkArgs(int argc){
    try{
        if(argc != 5){
            throw("usage: reorder <src_file> <tar_file> <node_map_file> <block_size>\n");
        }
    } catch (const char *msg){
        fprintf(stderr, "%s", msg);
    }
}

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

void sortNewEdge(vector< Node > &node, vector< Edge > &edge){
    updateEdgeEnd((int)edge.size(), node, edge);
    sort(edge.begin(), edge.end());
}

void updateNodeOrder(int nodeRange, vector< Node > &node){
    for(int i = 0; i < nodeRange; i++){
        node[i].updateOrder();
    }
}

void updateEdgeEnd(int range, vector< Node > &node, vector< Edge > &edge){
    for(int i = 0; i < range; i++){
        edge[i].u = node[edge[i].u].currOrder;
        edge[i].v = node[edge[i].v].currOrder;
    }
}

