#ifndef __IO_H__
#define __IO_H__

#include "ds.h"

enum{FORWARD = 0, EDGE_ITERATOR};

int getAlgo(const char *algo);
void inputList(const char *inFile, vector< Edge > &edge);

#endif
