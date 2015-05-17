#ifndef __REORDER_H__
#define __REORDER_H__

#include "struct.h"

typedef struct forwardNode{
    int newOrder;
    int realDeg;
    forwardNode(){
        realDeg = 0;
    }
} ForwardNode;

void forwardReorder(int nodeNum, vector< Edge > &edge);

#endif
