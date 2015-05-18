#ifndef __SOLVE_H__
#define __SOLVE_H__

#include "struct.h"

enum{FORWARD = 0, G_FORWARD, MAT, G_MAT, UNDEF};
enum{CPU = 0, GPU};

long long solveBlock(int blockSize, vector< Edge > &edge, int algo, int blockNum=1024, int threadNum=256);
long long mergeBlock(vector< Matrix > &block, int x, int y, int blockSize);
long long intersectBlock(vector< Matrix > &block, int x, int y, int z, int blockSize);
int scheduler(int nodeNum, int edgeNum);

#endif
