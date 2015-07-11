#include "solve.h"
#include "list.h"
#include "mat.h"
#include "tool.h"

long long findTriangle(const ListArrMatrix &block, const vector< int > &rowWidth, int blockDim){
    for(int b = 0; b < blockDim; b++){
        const ListArray &base = block[b][b];

        // solve block
        scheduler(base, base, rowWidth[b], false);

        for(int i = b+1; i < blockDim; i++){
            const ListArray &ext = block[b][i];
            ListArray *target;

            target = new ListArray; // delete in callList or gpuCountTriangle or scheduler
            ext.relabel(*target);
            // 2-way merge-1
            scheduler(base, *target, rowWidth[i], true);

            target = new ListArray; // delete in callList or gpuCountTriangle or scheduler
            ext.integrate(block[i][i], true, *target);
            // 2-way merge-2
            scheduler(ext, *target, rowWidth[i], true);

            for(int j = i+1; j < blockDim; j++){
                target = new ListArray; // delete in callList or gpuCountTriangle or scheduler
                block[b][j].integrate(block[i][j], false, *target);
                // 3-way merge
                scheduler(ext, *target, rowWidth[j], true);
            }
        }
    }

    for(int i = 0; i < MAX_THREAD_NUM; i++)
        waitThread(i);

    return 0;
}

void scheduler(
    const ListArray &edge, const ListArray &target, int width, bool delTar
){
    int device, proc;
    getStrategy(edge, target, device, proc);

    if(proc == LIST)
        list(device, edge, target, delTar);
    else{
        BitMat *tarMat = new BitMat; // delete in callMat or gpuCountTriangleMat
        int entry = averageCeil(width, BIT_PER_ENTRY);
        tarMat->initMat(target, entry);
        if(delTar) delete &target;
        mat(device, edge, *tarMat);
    }
}

void getStrategy(const ListArray &edge, const ListArray &target, int &device, int &proc){
    device = GPU, proc = LIST; // default
    int nodeNum = edge.nodeNum;
    double density = (double)target.edgeNum/(((double)(target.nodeNum)*target.nodeNum)/2);

    // suppose all nodeNum <= 10240
    // split line:
    //     N <= 4096: density = 7.902e-8*nodeNum^2 + 7.16e-4*nodeNum + 2.025
    //     N <= 10240: density = 5.407e-*nodeNum^2 + 1.12e-4*nodeNum + 0.857
    // >: MAT, <=: LIST
    double x = nodeNum, lhs;
    if(nodeNum <= 4096)
        lhs = 7.902e-8*x*x - 7.16e-4*x + 2.025;
    else
        lhs = 5.407e-9*x*x - 1.12e-4*x + 0.857;

    if(density > lhs) proc = MAT;
    else proc = LIST;

    extern int assignProc;
    switch(assignProc){
        case LIST: device = CPU, proc = LIST; break;
        case G_LIST: device = GPU, proc = LIST; break;
        case MAT: device = CPU, proc = MAT; break;
        case G_MAT: device = GPU, proc = MAT; break;
    }
}
