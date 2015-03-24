#ifndef __EDGE_ITERATOR_H__
#define __EDGE_ITERAROR_H__

#include "ds.h"

void reorderByDegeneracy(vector< Node > &node, vector< Edge > &edge);
void buildDegList(vector< Node > &node, vector< DegList > &degList);
void reordering(vector< Node > &node, vector< DegList > &degList);
int findMinDegNode(int &currPos, vector< DegList > &degList);
void removeNode(int v, vector< Node > &node, vector< DegList > &degList);
void decDegByOne(int v, vector< Node > &node, vector< DegList > &degList);
void removeNodeInList(int v, vector< Node > &node, vector< DegList > &degList);


#endif
