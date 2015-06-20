#include "mat.h"
#include "tool.h"
#include "solve.h"
#include "threadHandler.h"
#include <pthread.h>
#include <cstdio>

void mat(
    int device,
    const vector< Edge > &edge, int edgeRange,
    const vector< Edge > &target, int nodeNum, int entryNum,
    int threadNum, int blockNum
){
/*    MatArg *matArg = new MatArg;

    extern pthread_t threads[MAX_THREAD_NUM];
    extern bool threadUsed[MAX_THREAD_NUM];
    extern int currTid;

    matArg->edge.initArray(edge, edgeRange);
    matArg->target.initMat(target, nodeNum, entryNum);
    
    currTid %= 10;
    waitAndAddTriNum(currTid);
    threadUsed[currTid] = true;

    if(device == CPU || matArg->target.nodeNum > MAX_NODE_NUM_LIMIT){
        matArg->device = CPU;
        pthread_create(&threads[currTid++], NULL, callMat, (void*)matArg);
    }

    else{
        matArg->threadNum = threadNum;
        matArg->blockNum = blockNum;
        matArg->device = GPU;

        pthread_create(&threads[currTid++], NULL, callMat, (void*)matArg);
    }*/
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

