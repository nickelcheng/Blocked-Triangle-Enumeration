#include "solve.h"
#include "list.h"
#include "mat.h"
#include "tool.h"
#include "timer.h"

long long findTriangle(const ListArrMatrix &block, const vector< int > &rowWidth, int blockDim){
    for(int b = 0; b < blockDim; b++){
        const ListArray &base = block[b][b];

        // solve block
        scheduler(base, base, rowWidth[b]);

        for(int i = b+1; i < blockDim; i++){
            const ListArray &ext = block[b][i];
            ListArray *target;

            target = new ListArray;
            ext.relabel(*target);
            // 2-way merge-1
            scheduler(base, *target, rowWidth[i]);
            delete target;

            target = new ListArray;
            ext.integrate(block[i][i], true, *target);
            // 2-way merge-2
            scheduler(ext, *target, rowWidth[i]);
            delete target;

            for(int j = i+1; j < blockDim; j++){
                target = new ListArray;
                block[b][j].integrate(block[i][j], false, *target);
                // 3-way merge
                scheduler(ext, *target, rowWidth[j]);
                delete target;
            }
        }
    }

    return 0;
}

void scheduler(
    const ListArray &edge, const ListArray &target, int width
){
    if(edge.nodeNum == 0 || target.nodeNum == 0){
        printf("\033[1;31medge or target empty, end the procedure\n\033[m");
        return;
    }
    int device, proc;
    getStrategy(edge, target, width, device, proc);

    if(proc == LIST)
        list(device, edge, target);
    else
        mat(device, edge, target, width);
}

void getStrategy(const ListArray &edge, const ListArray &target, int width, int &device, int &proc){
    device = GPU, proc = LIST; // default
    double possibleEdge = (double)target.nodeNum * width;
    double density = (double)target.edgeNum/possibleEdge;
    printf("list area: %d nodes * %d width\n", target.nodeNum, width);
    printf("list area: %d edges (possible %.0lf)\n", edge.edgeNum, possibleEdge);
    printf("density = %lf%%\n", density*100.0);
    if(target.nodeNum >= MAX_NODE_NUM_LIMIT)
        printf("\033[1;31mCAN NOT USE VECTOR INTERSECTION\n\033[m");

    if(density > 0.06 && edge.nodeNum < MAX_NODE_NUM_LIMIT) proc = MAT;
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
