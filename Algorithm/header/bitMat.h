#ifndef __BIT_MAT_H__
#define __BIT_MAT_H__

#include "listArray.h"

typedef unsigned int UI;

#define BIT_PER_ENTRY (sizeof(UI)*8)

class BitMat{
    public:
        ~BitMat();
        static void createMask(void);
        void initMat(const ListArray &edge, int entry);
        UI getContent(int x, int y) const;
        void setEdge(int u, int v);
//    protected:
        static UI mask[BIT_PER_ENTRY];
        UI *mat;
        int nodeNum, entryNum; // width and height
};

#endif
