#ifndef __REORDER_H__
#define __REORDER_H__


#include "ds.h"
#include<vector>

using namespace std;

int reorder(int algo, vector< Node > &node, vector< Edge > &edge);
void updateGraph(vector< Node > &node, vector< Edge > &edge);
int getMaxDeg(vector< Node > &node);

#endif
