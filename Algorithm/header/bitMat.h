#ifndef __BIT_MAT_H__
#define __BIT_MAT_H__

#include "struct.h"

typedef unsigned int UI;

#define BIT_PER_ENTRY (sizeof(UI)*8)

class BitMat{
    public:
        BitMat(int node, int entry);
        ~BitMat();
        void initMat(const vector< Edge > &edge);
        UI getContent(int x, int y) const;
        void setEdge(int u, int v, const UI *mask);
//    protected:
        UI *mat;
        int nodeNum, entryNum; // width and height
};

#endif
