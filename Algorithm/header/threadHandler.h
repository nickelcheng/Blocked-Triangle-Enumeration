#ifndef __THREAD_HANDLER_H__
#define __THREAD_HANDLER_H__

#include "listArray.h"
#include "bitMat.h"
#include <pthread.h>

const int MAX_THREAD_NUM = 48;

typedef struct ListArg{
    const ListArray *edge, *target;
    int maxDeg;
    int device;
    bool delTar;
} ListArg;

typedef struct MatArg{
    const ListArray *edge;
    const BitMat *target;
    int device;
} MatArg;

void waitThread(int tid);
void *callList(void *arg);
void *callMat(void *arg);

#endif
