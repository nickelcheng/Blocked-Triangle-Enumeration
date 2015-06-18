#ifndef __BLOCK_H__
#define __BLOCK_H__

#include "main.h"

int initBlock(const vector< Edge > &edge, int nodeNum, int blockSize, vector< Matrix > &block);
void countBlockEdgeNum(const vector< Edge > &edge, int blockDim, int blockSize, vector< int* > &blockEdgeNum);
int integrateBlock(const vector< int* > &blockEdgeNum, int blockDim, int *newBlockID);
void splitBlock(const vector< Edge > &edge, const int* newBlockID, int blockSize, int blockDim, vector< Matrix > &block);
void sortBlock(vector< Matrix > &block, int blockDim);
void relabelBlock(vector< Edge > &edge, int blockSize, int uOffset, int vOffset);

#endif
