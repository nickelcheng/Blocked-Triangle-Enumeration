#ifndef __BLOCK_H__
#define __BLOCK_H__

#include "struct.h"

void initBlock(int blockDim, vector< Matrix > &block);
void splitBlock(int blockSize, vector< Matrix > &block, vector< Edge > &edge);
void sortBlock(vector< Matrix > &block, int blockDim);
void relabelBlock(vector< Edge > &edge, int blockSize, int uOffset, int vOffset);

#endif
