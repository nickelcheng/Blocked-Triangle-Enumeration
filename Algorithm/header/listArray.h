#ifndef __LIST_ARRAY_H__
#define __LIST_ARRAY_H__

#include "main.h"

class ListArray{
    public:
        ~ListArray(); // destructor
        void initArray(int n, int e);
        int getMaxDegree() const;
        DECORATE int getDeg(int v) const;
        DECORATE int getNodeNum() const;
        DECORATE const int* neiStart(int v) const;
        void integrate(const ListArray &a, ListArray &res) const;
        DECORATE void print() const;
//    private:
        int *nodeArr, nodeNum;
        int *edgeArr, edgeNum;
};

#ifdef __NVCC__
__global__ void edge2listArr(const Edge *edge, int nodeNum, int edgeNum, ListArray *listArr);
#endif

#endif
