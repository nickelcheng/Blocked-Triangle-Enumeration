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

#ifdef __NVCC__
void gListArr2BitMat(const ListArray &src, BitMat **tar, UI **mat, int entryNum);
void createMask(UI *mask, UI **d_mask);
__global__ void initMat(UI *mat, int nodeNum, int entryNum);
__global__ void listArr2BitMat(const ListArray *src, const UI *mask, BitMat *tar, UI *mat, int entryNum);
#endif

void cListArr2BitMat(const ListArray &src, BitMat **tar, UI **mat, int entryNum);

#endif
