#ifndef __IO_H__
#define __IO_H__

#include<vector>
#include "struct.h"

using namespace std;

void inputEdge(const char *inFile, vector< Edge > &edge);
void initBlock(int blockDim, vector< Matrix > &block);
void splitBlock(int blockSize, vector< Matrix > &block, vector< Edge > &edge);
void relabelBlock(int blockSize, int blockDim, vector< Matrix > &block);
void relabel(int offset, vector< Edge > &edge);

#endif

