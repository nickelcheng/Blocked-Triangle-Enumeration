#include "list.h"
#include "solve.h"
#include <cstdio>

void list(int device, const ListArray &edge, const ListArray &target){
    if(edge.edgeNum <= 0 || target.edgeNum <= 0) return;

    int maxDeg = target.getMaxDegree();
    printf("edge region max degree = %d (shared memory size)\n", maxDeg);
    printf("edge region avg degree = %d (work load)\n", edge.edgeNum/edge.nodeNum);
    if(device == GPU && maxDeg > MAX_DEG_LIMIT)
        device = CPU;
    
    if(device == CPU)
        cpuCountList(edge, target);
    else
        gpuCountTriangle(edge, target, maxDeg);
}

void cpuCountList(const ListArray &edge, const ListArray &target){
    printf("\033[1;33mcpu list intersection!!!\n\033[m");
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
    extern long long triNum;
    triNum += ans;
}

