#include "mat.h"
#include "solve.h"

void mat(int device, const ListArray &edge, const BitMat &target){
    extern pthread_t threads[MAX_THREAD_NUM];
    extern bool threadUsed[MAX_THREAD_NUM];
    extern int currTid;

    MatArg *matArg = new MatArg; // delete in callMat
    matArg->device = device;
    matArg->edge = &edge, matArg->target = &target;
    if(device == GPU && target.nodeNum > MAX_NODE_NUM_LIMIT){
        matArg->device = CPU;
    }

    currTid %= MAX_THREAD_NUM;
    waitThread(currTid);
    threadUsed[currTid] = true;
    pthread_create(&threads[currTid++], NULL, callMat, (void*)matArg);
}

void cpuCountMat(const MatArg &matArg){
    const ListArray &edge = *(matArg.edge);
    const BitMat &target = *(matArg.target);

    long long ans = 0;

    for(int e = 0; e < target.entryNum; e++){
    // iterator through each edge
        int range = edge.nodeNum;
        for(int u = 0; u < range; u++){

            int uDeg = edge.getDeg(u);
            if(uDeg > 0){
                const int *uNei = edge.neiStart(u);
                for(int i = 0; i < uDeg; i++){
                    int v = uNei[i];
                    UI e1 = target.getContent(u, e);
                    UI e2 = target.getContent(v, e);
                    ans += countOneBits(e1 & e2);
                }
            }
        }
    }
    extern long long triNum;
    extern pthread_mutex_t lock;
    pthread_mutex_lock(&lock);
    triNum += ans;
    pthread_mutex_unlock(&lock);
}

