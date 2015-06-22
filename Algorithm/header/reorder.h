#ifndef __REORDER_H__
#define __REORDER_H__

#include "main.h"
#include <vector>

using std::vector;

typedef struct ForwardNode{
    int order; // ori order for g reorder, new order for c reorder
    int realDeg;
    DECORATE bool operator < (const ForwardNode &a) const{
        return realDeg < a.realDeg;
    }
} ForwardNode;

void gForwardReorder(int nodeNum, vector< Edge > &edge);
void cForwardReorder(int nodeNum, vector< Edge > &edge);

#endif
