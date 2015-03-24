#include "io.h"
#include<cstdio>
#include<cstring>

int getAlgo(const char *algo){
    if(strcmp(algo, "forward") == 0)
        return FORWARD;
    if(strcmp(algo, "edge") == 0)
        return EDGE_ITERATOR;
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

