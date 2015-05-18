#ifndef __IO_H__
#define __IO_H__

#include<vector>
#include "struct.h"

using namespace std;

void inputEdge(const char *inFile, vector< Edge > &edge);
void printBlock(vector< Edge > &edge, int x, int y);

#endif

