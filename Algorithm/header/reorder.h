#ifndef __REORDER_H__
#define __REORDER_H__

#include "main.h"
#include <vector>

using std::vector;

const int STREAM_NUM = 2;
const int EDGE_UNIT = 1024*1024;

typedef struct ForwardNode{
    int order; // ori order for g reorder, new order for c reorder
    int realDeg;
    DECORATE bool operator < (const ForwardNode &a) const{
        return realDeg < a.realDeg;
    }
} ForwardNode;

void forwardReorder(int nodeNum, vector< Edge > &edge);
void cForwardReorder(int nodeNum, vector< Edge > &edge);

void gForwardReorder(int nodeNum, vector< Edge > &edge);
#ifdef __NVCC__
__global__ void initNode(int nodeNum, ForwardNode *node);
__global__ void countDeg(Edge *edge, int edgeNum, ForwardNode *node);
__global__ void setNewOrder(const ForwardNode *node, int nodeNum, int *newOrder);
#endif

#endif
