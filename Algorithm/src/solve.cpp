#include "solve.h"
#include "list.h"
#include "mat.h"
#include "tool.h"

long long findTriangle(const ListArrMatrix &block, const vector< int > &rowWidth, int blockDim){
    for(int b = 0; b < blockDim; b++){
        const ListArray &base = block[b][b];

        // solve block
        fprintf(stderr, "solve block %d\n", b);
        scheduler(base, base, rowWidth[b], false);

        for(int i = b+1; i < blockDim; i++){
            const ListArray &ext = block[b][i];
            ListArray *target;

            // 2-way merge-1
            fprintf(stderr, "2-1merge %d %d\n", b, i);
            scheduler(base, ext, rowWidth[i], false);

            target = new ListArray; // delete in callList or gpuCountTriangle or scheduler
            ext.integrate(block[i][i], true, *target);
            // 2-way merge-2
            fprintf(stderr, "2-2merge %d %d\n", b, i);
            scheduler(ext, *target, rowWidth[i], true);

            for(int j = i+1; j < blockDim; j++){
                target = new ListArray; // delete in callList or gpuCountTriangle or scheduler
                block[b][j].integrate(block[i][j], false, *target);
                // 3-way merge
                printf("3merge %d %d %d\n", b, i, j);
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
    device = CPU, proc = LIST; // default
    extern int assignProc;
    switch(assignProc){
        case LIST: device = CPU, proc = LIST; break;
        case G_LIST: device = GPU, proc = LIST; break;
        case MAT: device = CPU, proc = MAT; break;
        case G_MAT: device = GPU, proc = MAT; break;
    }
}
