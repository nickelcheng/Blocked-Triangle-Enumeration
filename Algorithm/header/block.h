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
    const vector< int* > &blockEdgeNum, int blockDim, int blockSize,
    int *newID, vector< int > &rowWidth
);

int integrateBlock2(
    const vector< int* > &blockEdgeNum, int blockDim,
    int *newID, vector< int > &rowWidth
);
void splitBlock(
    const vector< Edge > &edge, const int* newID, int blockSize, int blockDim,
    EdgeMatrix &block
);

void cTransBlock(vector< Edge > &edge, int nodeNum, int uOffset, int vOffset, ListArray &listArr);
void cRelabelBlock(int uOffset, int vOffset, vector< Edge > &edge);
void cEdge2ListArr(const vector< Edge > &edge, ListArray &listArr);

void initListArrBlock(
    EdgeMatrix &edgeBlock, const vector< int > &rowWidth, int blockDim, int blockSize,
    ListArrMatrix &listArrBlock
);

void gTransBlock(
    const vector< Edge > &edge, int nodeNum, int uOffset, int vOffset,
    ListArray &listArr, ListArray *d_listArr
);

void setEmptyArray(int nodeNum, int *arr);

#ifdef __NVCC__
__global__ void relabelBlock(int edgeNum, int uOffset, int vOffset, Edge *edge);
#endif

#endif
