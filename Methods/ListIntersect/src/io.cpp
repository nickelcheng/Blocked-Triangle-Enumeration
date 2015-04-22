#include "io.h"
#include<cstdio>
#include<cstring>

int getAlgo(const char *algo){
    if(strcmp(algo, "forward") == 0)
        return FORWARD;
    if(strcmp(algo, "cover_forward") == 0)
        return COVER_FORWARD;
    if(strcmp(algo, "edge") == 0)
        return EDGE_ITERATOR;
    if(strcmp(algo, "cover_edge") == 0)
        return COVER_EDGE_ITERATOR;
    if(strcmp(algo, "cover") == 0)
        return COVER;
    return -1;
}

void inputList(const char *inFile, vector< Edge > &edge){
    FILE *fp = fopen(inFile, "r");
    int u, v;
    while(fscanf(fp, "%d%d", &u, &v) != EOF){
        edge.push_back(Edge(u,v));
    }
    fclose(fp);
}

void initNode(int nodeNum, vector< Node > &node, int algo){
    bool useVertexCover = false;
    if(algo == COVER_FORWARD || algo == COVER_EDGE_ITERATOR || algo == COVER)
        useVertexCover = true;

    for(int i = 0; i < nodeNum; i++)
        node.push_back(Node(i, useVertexCover));
}
