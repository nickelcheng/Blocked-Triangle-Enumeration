#ifndef __SOLVE_H__
#define __SOLVE_H__

#include "main.h"

enum{LIST = 0, G_LIST, MAT, G_MAT, UNDEF};
enum{CPU = 0, GPU};

void solveBlock(const vector< Edge > &edge, int blockSize);

void mergeBlock(const vector< Matrix > &block, int x, int y, int blockSize);

void intersectBlock(const vector< Matrix > &block, int x, int y, int z, int blockSize);

void scheduler(
    const vector< Edge > &edge, int edgeRange,
    const vector< Edge > &target, int nodeNum, int entryNum
);

void getStrategy(int nodeNum, int edgeNum, int &proc);

#endif
