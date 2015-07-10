#include "list.h"
#include "solve.h"
#include <cstdio>

void list(int device, const ListArray &edge, const ListArray &target, bool delTar){
    extern pthread_t threads[MAX_THREAD_NUM];
    extern bool threadUsed[MAX_THREAD_NUM];
    extern int currTid;

    if(edge.edgeNum <= 0 || target.edgeNum <= 0) return;

    ListArg *listArg = new ListArg; // delete in callList
    listArg->device = device;
    listArg->edge = &edge, listArg->target = &target;
    listArg->maxDeg = target.getMaxDegree();
    if(device == GPU && listArg->maxDeg > MAX_DEG_LIMIT){
        listArg->device = CPU;
/*        delete listArg;
        return;*/
    }
    listArg->delTar = delTar;
    
    currTid %= MAX_THREAD_NUM;
    waitThread(currTid);
    threadUsed[currTid] = true;
    pthread_create(&threads[currTid++], NULL, callList, (void*)listArg);
}

void cpuCountList(const ListArg &listArg){
    const ListArray &edge = *(listArg.edge);
    const ListArray &target = *(listArg.target);

    long long ans = 0;

    // iterator through each edge (u, v)
    int range = edge.nodeNum;
    for(int u = 0; u < range; u++){
        int uLen = target.getDeg(u);
        if(uLen > 0){
            const int *uList = target.neiStart(u);
            int uDeg = edge.getDeg(u);
            if(uDeg > 0){
                const int *uNei = edge.neiStart(u);
                for(int i = 0; i < uDeg; i++){
                    int v = uNei[i];
                    int vLen = target.getDeg(v);
                    if(vLen > 0){
                        const int *vList = target.neiStart(v);
                        // intersect u list and v list in target
                        ans += intersectList(uLen, vLen, uList, vList);
                    }
                }
            }
        }
    }
    printf("clist %lld\n", ans);
    extern long long triNum;
    extern pthread_mutex_t lock;
    pthread_mutex_lock(&lock);
    triNum += ans;
    pthread_mutex_unlock(&lock);
}

