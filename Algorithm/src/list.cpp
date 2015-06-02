#include "list.h"
#include "listArray.h"
#include "reorder.h"
#include "solve.h"
#include <cstring>
#include <algorithm>
#include <pthread.h>

void list(
    int device,
    const vector< Edge > &edge, int edgeRange,
    const vector< Edge > &target, int nodeNum,
    int threadNum, int blockNum
){
    ListArg *listArg = new ListArg;

    extern pthread_t threads[MAX_THREAD_NUM];
    extern int currTid;

    listArg->edge.initArray(edge, edgeRange);
    listArg->target.initArray(target, nodeNum);

    currTid %= 10;
    waitAndAddTriNum(currTid);

    if(device == CPU || listArg->maxDeg > MAX_DEG_LIMIT){
        listArg->device = CPU;
        pthread_create(&threads[currTid++], NULL, callList, (void*)listArg);
    }

    else{
        listArg->maxDeg = listArg->target.getMaxDegree();
        listArg->threadNum = threadNum;
        listArg->blockNum = blockNum;
        listArg->device = GPU;

        pthread_create(&threads[currTid++], NULL, callList, (void*)listArg);
    }
}

long long cpuCountList(const ListArg &listArg){
    const ListArray &edge = listArg.edge;
    const ListArray &target = listArg.target;
    long long triNum = 0;
    // iterator through each edge (u, v)
    int range = edge.getNodeNum();
    for(int u = 0; u < range; u++){
        int uLen = target.getDeg(u);
        const int *uList = target.neiStart(u);

        const int *uNei = edge.neiStart(u);
        int uDeg = edge.getDeg(u);
        for(int i = 0; i < uDeg; i++){
            int v = uNei[i];
            int vLen = target.getDeg(v);
            const int *vList = target.neiStart(v);
            // intersect u list and v list in target
            triNum += intersectList(uLen, vLen, uList, vList);
        }
    }
    return triNum;
}

