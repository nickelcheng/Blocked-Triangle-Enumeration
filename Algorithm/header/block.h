#ifndef __BLOCK_H__
#define __BLOCK_H__

#include "main.h"
#include "listArray.h"

typedef vector< Edge > Entry;
typedef vector< Entry > EdgeRow;
typedef vector< EdgeRow > EdgeMatrix;

typedef vector< ListArray > ListArrRow;
typedef vector< ListArrRow > ListArrMatrix;

int initEdgeBlock(
    const vector< Edge > &edge, int nodeNum, int blockSize,
    EdgeMatrix &block, vector< int > &rowWidth
);

void countBlockEdgeNum(
    const vector< Edge > &edge, int blockDim, int blockSize,
    vector< int* > &blockEdgeNum
);

int integrateBlock(
    const vector< int* > &blockEdgeNum, int blockDim,
    int *newID, vector< int > &rowWidth
);

void splitBlock(
    const vector< Edge > &edge, const int* newID, int blockSize, int blockDim,
    EdgeMatrix &block
);

/*void sortBlock(vector< Matrix > &block, int blockDim);
void relabelBlock(vector< Edge > &edge, int blockSize, int uOffset, int vOffset);*/

void initListArrBlock(
    const EdgeMatrix &edgeBlock, const vector< int > &rowWidth, int blockDim, int blockSize,
    ListArrMatrix &listArrBlock
);

#ifdef __NVCC__
__global__ void relabelBlock(int edgeNum, int uOffset, int vOffset, Edge *edge);
#endif

#endif
