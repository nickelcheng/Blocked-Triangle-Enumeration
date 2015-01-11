#include "io.h"

int input(const char *srcFile, vector< Edge > &edge){
    FILE *fp = fopen(srcFile, "r");
    int nodeNum, edgeNum;

    fscanf(fp, "%d%d", &nodeNum, &edgeNum);
    edge.resize(edgeNum);

    for(int i = 0; i < edgeNum; i++){
        int u, v;
        fscanf(fp, "%d%d", &u, &v);
        edge[i] = Edge(u, v);
    }

    fclose(fp);
    return nodeNum;
}

void output(
    const char *tarFile, const char *nodeMapFile, 
    vector< Node > &node, vector< Edge > &edge
){
    outputTar(tarFile, (int)node.size(), edge);
    outputNodeMap(nodeMapFile, node);
}

void outputTar(const char *file, int nodeNum, vector< Edge > &edge){
    FILE *fp = fopen(file, "w");
    int edgeNum = (int)edge.size();

    fprintf(fp, "%d %d\n", nodeNum, edgeNum);
    for(int i = 0; i < edgeNum; i++){
        fprintf(fp, "%d %d\n", edge[i].u, edge[i].v);
    }

    fclose(fp);
}

void outputNodeMap(const char *file, vector< Node > &node){
    FILE *fp = fopen(file, "w");
    int nodeNum = (int)node.size();

    for(int i = 0; i < nodeNum; i++){
        fprintf(fp, "%d <- %d\n", i, node[i].oriOrder);
    }

    fclose(fp);
}

