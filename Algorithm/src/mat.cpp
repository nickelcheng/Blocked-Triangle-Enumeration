#include "mat.h"
#include "tool.h"
#include "solve.h"
#include <pthread.h>

extern vector< pthread_t* > threads;

void mat(
    int device,
    const vector< Edge > &edge, int edgeRange,
    const vector< Edge > &target, int nodeNum, int entryNum,
    int threadNum, int blockNum
){
    MatArg *matArg = new MatArg;

    matArg->edge.initArray(edge, edgeRange);
    matArg->target.initMat(target, nodeNum, entryNum);
    
    threads.push_back(new pthread_t);

    if(device == CPU || matArg->target.nodeNum > MAX_NODE_NUM_LIMIT){
        pthread_create(threads.back(), NULL, callCpuMat, (void*)matArg);
    }

    else{
        matArg->threadNum = threadNum;
        matArg->blockNum = blockNum;

        pthread_create(threads.back(), NULL, callGpuMat, (void*)matArg);
    }
}

void *callCpuMat(void *arg){
    long long *triNum = new long long;
    *triNum = cpuCountMat(*(MatArg*)arg);
    delete (MatArg*)arg;
    pthread_exit((void*)triNum);
}

void *callGpuMat(void *arg){
    long long *triNum = new long long;
    *triNum = gpuCountTriangleMat(*(MatArg*)arg);
    delete (MatArg*)arg;
    pthread_exit((void*)triNum);
}

long long cpuCountMat(const MatArg &matArg){
    const ListArray &edge = matArg.edge;
    const BitMat &target = matArg.target;
    long long triNum = 0;
    for(int e = 0; e < target.entryNum; e++){
    // iterator through each edge
        int range = edge.getNodeNum();
        for(int u = 0; u < range; u++){

            const int *uNei = edge.neiStart(u);
            int uDeg = edge.getDeg(u);
            for(int i = 0; i < uDeg; i++){
                int v = uNei[i];
                UI e1 = target.getContent(u, e);
                UI e2 = target.getContent(v, e);
                long long tmp = countOneBits(e1 & e2);
                if(countOneBits(e1&e2)>0)
                triNum += tmp;
            }
        }
    }
    return triNum;
}

