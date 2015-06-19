#ifndef __REORDER_H__
#define __REORDER_H__

#include "main.h"

typedef struct ForwardNode{
    int oriOrder;
    int realDeg;
    DECORATE bool operator < (const ForwardNode &a) const{
        return realDeg < a.realDeg;
    }
} ForwardNode;

void forwardReorder(int nodeNum, vector< Edge > &edge);

#endif
