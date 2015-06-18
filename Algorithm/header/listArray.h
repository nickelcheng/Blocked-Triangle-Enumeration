#ifndef __LIST_ARRAY_H__
#define __LIST_ARRAY_H__

#include "main.h"

class ListArray{
    public:
        ~ListArray(); // destructor
        void initArray(const vector< Edge > &edge, int n);
        int getMaxDegree() const;
        DECORATE int getDeg(int v) const;
        DECORATE int getNodeNum() const;
        DECORATE const int* neiStart(int v) const;
        DECORATE void print() const;
//    private:
        int *nodeArr, nodeNum;
        int *edgeArr, edgeNum;
};

#endif
