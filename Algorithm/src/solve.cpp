#include "solve.h"
#include "list.h"
#include "mat.h"
#include "tool.h"
#include "timer.h"

long long findTriangle(const ListArrMatrix &block, const vector< int > &rowWidth, int blockDim){
    for(int b = 0; b < blockDim; b++){
        const ListArray &base = block[b][b];

        // solve block
        timerInit(1)
        printf("\nsolve subgraph %d\n", b);
        timerStart(0)
        scheduler(base, base, rowWidth[b], true);
        timerEnd("time", 0)

        for(int i = b+1; i < blockDim; i++){
            const ListArray &ext = block[b][i];
            if(ext.edgeNum == 0) continue;
            ListArray *target;

            target = new ListArray;
            ext.relabel(*target);
            // 2-way merge-1
            printf("\n2-merge-1 %d and %d\n", b, i);
            timerStart(0)
            scheduler(base, *target, rowWidth[i], false);
            timerEnd("time", 0)
            delete target;

            target = new ListArray;
            ext.integrate(block[i][i], true, *target);
            // 2-way merge-2
            printf("\n2-merge-2 %d and %d\n", b, i);
            timerStart(0)
            scheduler(ext, *target, rowWidth[i], false);
            timerEnd("time", 0)
            delete target;

            for(int j = i+1; j < blockDim; j++){
                target = new ListArray;
                block[b][j].integrate(block[i][j], false, *target);
                // 3-way merge
                printf("\n3-merge %d, %d, and %d\n", b, i, j);
                timerStart(0)
                scheduler(ext, *target, rowWidth[j], false);
                timerEnd("time", 0)
                delete target;
            }
        }
    }

    return 0;
}

void scheduler(
    const ListArray &edge, const ListArray &target, int width, bool isDiagonal
){
    if(edge.edgeNum == 0 || target.edgeNum == 0){
        printf("\033[1;31medge or target empty, end the procedure\033[m\n");
        return;
    }
    int device, proc;
    getStrategy(edge, target, width, isDiagonal, device, proc);

    if(proc == LIST)
        list(device, edge, target);
    else
        mat(device, edge, target, width);
}

void getStrategy(const ListArray &edge, const ListArray &target, int width, bool isDiagonal, int &device, int &proc){
    device = GPU, proc = LIST; // default
    double possibleEdge;
    if(isDiagonal) possibleEdge = (double)target.nodeNum*(target.nodeNum-1)/2.0;
    else possibleEdge = (double)target.nodeNum * width;
    double density = (double)target.edgeNum/possibleEdge;
    printf("list area: %d nodes * %d width\n", target.nodeNum, width);
    printf("list area: %d edges (possible %.0lf)\n", target.edgeNum, possibleEdge);
    printf("density = %lf%%\n", density*100.0);
/*    if(target.nodeNum >= MAX_NODE_NUM_LIMIT)
        printf("\033[1;31mCAN NOT USE VECTOR INTERSECTION\033[m\n");*/

    extern double densityBoundary;
    if(density > densityBoundary && edge.nodeNum < MAX_NODE_NUM_LIMIT) proc = MAT;
    else proc = LIST;

    extern int assignProc;
    switch(assignProc){
//        case LIST: device = CPU, proc = LIST; break;
        case G_LIST: device = GPU, proc = LIST; break;
//        case MAT: device = CPU, proc = MAT; break;
        case G_MAT: device = GPU, proc = MAT; break;
    }
    if(edge.nodeNum >= MAX_NODE_NUM_LIMIT) proc = LIST;
}
