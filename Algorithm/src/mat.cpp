#include "mat.h"
#include "solve.h"
#include "tool.h"
#include "timer.h"

void mat(int device, const ListArray &edge, const ListArray &target, int width){
    int entry = averageCeil(width, BIT_PER_ENTRY);
    if(device == GPU && target.nodeNum > MAX_NODE_NUM_LIMIT)
        device = CPU;

    if(device == CPU){
        timerInit(1)
        timerStart(0)
        BitMat *tarMat = new BitMat;
        tarMat->initMat(target, entry);
        timerEnd("cpu list->bitmat", 0)
        timerStart(0)
        cpuCountMat(edge, *tarMat);
        delete tarMat;
        timerEnd("mat count", 0)
    }
    else
        gpuCountTriangleMat(edge, target, entry);
}

void cpuCountMat(const ListArray &edge, const BitMat &target){
    extern unsigned char oneBitNum[BIT_NUM_TABLE_SIZE];
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
                    UC e1 = target.getContent(u, e);
                    UC e2 = target.getContent(v, e);
                    ans += countOneBits(e1 & e2, oneBitNum);
                }
            }
        }
    }
    extern long long triNum;
    triNum += ans;
}

