#ifndef __BIT_MAT_H__
#define __BIT_MAT_H__

#include "struct.h"

typedef unsigned int UI;

#define BIT_PER_ENTRY (sizeof(UI)*8)

class BitMat{
    public:
        BitMat(int node);
        ~BitMat();
        void initMat(const vector< Edge > &edge);
        virtual UI getContent(int x, int y) const = 0;
        virtual void setEdge(int u, int v, const UI *mask) = 0;
//    protected:
        UI *mat;
        int nodeNum, entryNum;
};

class EdgeMat: public BitMat{
    public:
        EdgeMat(int node);
        void setEdge(int u, int v, const UI *mask);
        UI getContent(int node, int entry) const;
};

class TarMat: public BitMat{
    public:
        TarMat(int node);
        void setEdge(int u, int v, const UI *mask);
        UI getContent(int node, int entry) const;
};

#endif
