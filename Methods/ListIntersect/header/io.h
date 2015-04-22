#ifndef __IO_H__
#define __IO_H__

#include "ds.h"

enum{FORWARD = 0, COVER_FORWARD, EDGE_ITERATOR, COVER_EDGE_ITERATOR, COVER};

int getAlgo(const char *algo);
void inputList(const char *inFile, vector< Edge > &edge);
bool useVertexCover(int algo);
void initNode(int nodeNum, vector< Node > &node, bool useVC);

#endif
