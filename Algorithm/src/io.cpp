#include "io.h"
#include<cstdio>

void inputEdge(const char *inFile, vector< Edge > &edge){
    FILE *fp = fopen(inFile, "r");
    int u, v;
    while(fscanf(fp, "%d%d", &u, &v) != EOF){
        edge.push_back(Edge(u,v));
    }
    fclose(fp);
}

void printBlock(vector< Edge > &edge, int x, int y){
    vector< Edge >::iterator e = edge.begin();
    printf("block[%d][%d]:", x, y);
    for(; e != edge.end(); ++e){
        printf(" (%d,%d)", e->u, e->v);
    }
    printf("\n");
}
