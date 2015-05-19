#ifndef __SOLVE_H__
#define __SOLVE_H__

#include "struct.h"

enum{LIST = 0, G_LIST, MAT, G_MAT, UNDEF};
enum{CPU = 0, GPU};

long long solveBlock(
    const vector< Edge > &edge, int blockSize,
    int assignProc, int blockNum=1024, int threadNum=256
);

long long mergeBlock(
    const vector< Matrix > &block, int x, int y, int blockSize,
    int assignProc, int blockNum=1024, int threadNum=256
);

long long intersectBlock(
    const vector< Matrix > &block, int x, int y, int z, int blockSize,
    int assignProc, int blockNum=1024, int threadNum=256
);

long long depatch(
    int proc,
    const vector< Edge > &edge, int edgeRange,
    const vector< Edge > &target, int nodeNum,
    int blockNum, int threadNum
);

int scheduler(int nodeNum, int edgeNum, int assignProc);
#endif
