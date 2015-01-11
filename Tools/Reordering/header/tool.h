#ifndef __MY_TOOL_H__
#define __MY_TOOL_H__

#include "main.h"

void initSetting(
    int argc, char *argv[],
    char *srcFile, char *tarFile,
    char *nodeMapFile,
    int &blockSize
);
void checkArgs(int argc);

void initNodeOriOrder(int nodeNum, vector< Node > &node);
void sortNewEdge(vector< Edge > &edge);
void updateNodeOrder(int nodeRange, vector< Node > &node);
void updateEdgeEnd(int edgeRange, vector< Node > &node, vector< Edge > &edge);

#endif
