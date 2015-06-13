#include "threadHandler.h"
#include "solve.h"
#include "list.h"
#include "mat.h"
#include <pthread.h>

extern pthread_t threads[MAX_THREAD_NUM];
extern bool threadUsed[MAX_THREAD_NUM];

void waitAndAddTriNum(int tid){
    if(!threadUsed[tid]) return;
    extern long long triNum;
    void *ans;
    pthread_join(threads[tid], &ans);
    triNum += *(long long*)ans;
    delete (long long*)ans;
    threadUsed[tid] = false;
}

void *callList(void *arg){
    long long *triNum = new long long;
    if(((ListArg*)arg)->device == CPU)
        *triNum = cpuCountList(*(ListArg*)arg);
    else
        *triNum = gpuCountTriangle(*(ListArg*)arg);
        
    delete (ListArg*)arg;
    pthread_exit((void*)triNum);
}

void *callMat(void *arg){
    long long *triNum = new long long;
    if(((MatArg*)arg)->device == CPU)
        *triNum = cpuCountMat(*(MatArg*)arg);
    else
        *triNum = gpuCountTriangleMat(*(MatArg*)arg);

    delete (MatArg*)arg;
    pthread_exit((void*)triNum);
}

