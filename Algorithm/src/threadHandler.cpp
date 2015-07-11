#include "threadHandler.h"
#include "solve.h"
#include "list.h"
#include "mat.h"
#include <pthread.h>

extern pthread_t threads[MAX_THREAD_NUM];
extern bool threadUsed[MAX_THREAD_NUM];

void waitThread(int tid){
    if(!threadUsed[tid]) return;
    pthread_join(threads[tid], NULL);
    threadUsed[tid] = false;
}

void *callList(void *arg){
    if(((ListArg*)arg)->device == CPU){
        cpuCountList(*(ListArg*)arg);
        if(((ListArg*)arg)->delTar)
            delete ((ListArg*)arg)->target;
    }
    else
        gpuCountTriangle(*(ListArg*)arg);

    delete (ListArg*)arg;
    pthread_exit(NULL);
}

void *callMat(void *arg){
    if(((MatArg*)arg)->device == CPU){
        cpuCountMat(*(MatArg*)arg);
        delete ((MatArg*)arg)->target;
    }
    else
        gpuCountTriangleMat(*(MatArg*)arg);

    delete (MatArg*)arg;
    pthread_exit(NULL);
}

