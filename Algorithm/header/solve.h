#ifndef __SOLVE_H__
#define __SOLVE_H__

#include "struct.h"

enum{LIST = 0, G_LIST, MAT, G_MAT, UNDEF};
enum{CPU = 0, GPU};

long long solveBlock(const vector< Edge > &edge, int blockSize);

long long mergeBlock(const vector< Matrix > &block, int x, int y, int blockSize);

long long intersectBlock(const vector< Matrix > &block, int x, int y, int z, int blockSize);

long long scheduler(
    const vector< Edge > &edge, int edgeRange,
    const vector< Edge > &target, int nodeNum, int entryNum
);

#endif
