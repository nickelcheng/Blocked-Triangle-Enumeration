#ifndef __MY_REORDER_H__
#define __MY_REORDER_H__

#include "main.h"

void initNodeNei(int nodeRange, int edgeRange, vector< Node > &node, vector< Edge > &edge);

void reorder(int nodeRange, int edgeRange, vector< Node > &node, vector< Edge > &edge);
void initDegInfoPQ(int nodeRange, vector< Node > &node, DegInfoPQ &pq);
int findMinDegNode(DegInfoPQ &pq, vector< Node > &node);
void removeNode(int v, vector< Node > &node, DegInfoPQ &pq);
void updateDegInfoPQ(int v, vector< Node > &node, DegInfoPQ &pq);

int removeOutRangeEdge(int nodeRange, int edgeRange, vector< Edge > &edge);

#endif
