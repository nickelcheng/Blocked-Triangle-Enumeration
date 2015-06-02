#ifndef __THREAD_HANDLER_H__
#define __THREAD_HANDLER_H__

#include "listArray.h"
#include "bitMat.h"

const int MAX_THREAD_NUM = 20;

typedef struct listArg{
    ListArray edge, target;
    int maxDeg, threadNum, blockNum;
    int device;
} ListArg;

typedef struct matArg{
    ListArray edge;
    BitMat target;
    int threadNum, blockNum;
    int device;
} MatArg;

void waitAndAddTriNum(int tid);
void *callList(void *arg);
void *callMat(void *arg);

#endif
