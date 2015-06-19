#include "io.h"
#include<cstdio>

int inputEdge(const char *inFile, vector< Edge > &edge){
    FILE *fp = fopen(inFile, "r");
    int u, v, n;
    fscanf(fp, "%d", &n);
    while(fscanf(fp, "%d%d", &u, &v) != EOF){
        edge.push_back((Edge){u,v});
    }
    fclose(fp);
    return n;
}

void printEdge(const vector< Edge > &edge){
    vector< Edge >::const_iterator e = edge.begin();
    for(; e != edge.end(); ++e){
        printf(" (%d,%d)", e->u, e->v);
    }
    printf("\n");
}
