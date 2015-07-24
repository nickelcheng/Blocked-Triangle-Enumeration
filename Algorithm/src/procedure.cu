#include<cstdio>
#include <cstdlib>
#include "io.h"
#include "reorder.h"
#include "solve.h"
#include "timer.h"
#include "mat.h"
#include "degeneracy.h"

int assignProc, threadNum, blockNum;
int edgeNumLimit;
double densityBoundary = 0;
long long triNum;
UI mask[BIT_PER_ENTRY], *d_mask;
UC oneBitNum[BIT_NUM_TABLE_SIZE], *d_oneBitNum;

int main(int argc, char *argv[]){
    if(argc < 3){
        fprintf(stderr, "arguments:\n");
        fprintf(stderr, "  assign_proc\t\t\t(required, 1(list) or 3(vector))\n");
        fprintf(stderr, "  input_path\t\t\t(required)\n");
        fprintf(stderr, "  reorder_or_not\t\t(default=true)\n");
        fprintf(stderr, "  max_allowed_gpu_block_num\t(default:16000)\n");
        fprintf(stderr, "  max_allowed_thread_per_block\t(default:1024)\n");
        return 0;
    }

    assignProc = atoi(argv[1]);
    if(assignProc < LIST || assignProc > G_MAT){
        fprintf(stderr, "algo choice\n0: forward\n1: g_forward\n2: mat\n3: g_mat\n");
        return 0;
    }

    bool reorder = true;
    blockNum = GPU_BLOCK_NUM;
    threadNum = GPU_THREAD_NUM;
    if(argc >= 4) reorder = (strcmp("true",argv[3])==0) ? true : false;
    if(argc >= 5) blockNum = atoi(argv[4]);
    if(argc >= 6) threadNum = atoi(argv[5]);

    vector< Edge > edge;
    int nodeNum = inputEdge(argv[2], edge);
    
    // resolve first cuda call slow timing issue
    cudaFree(0);

    timerInit(2)
//    forwardReorder(nodeNum, edge, reorder);
    reorderByDegeneracy(nodeNum, edge, reorder);

    timerStart(0)
    timerStart(1)
    ListArray listArr, *d_listArr;
    cudaMalloc((void**)&d_listArr, sizeof(ListArray));
    if(assignProc == 0)
        cTransBlock(edge, nodeNum, 0, 0, listArr);
    else
        gTransBlock(edge, nodeNum, 0, 0, listArr, d_listArr);
    cudaFree(d_listArr);
    timerEnd("edge->list", 1)

//    timerStart(1)
    createMask(mask, &d_mask);
    createOneBitNumTable(oneBitNum, &d_oneBitNum);
    triNum = 0;
//    timerEnd("init", 1)
    timerStart(1)
    scheduler(listArr, listArr, nodeNum, true);
    timerEnd("count", 1)
    
    cudaFree(d_oneBitNum);
    cudaFree(d_mask);

    int edgeNum = (int)edge.size();
    double density = (double)edgeNum/((double)nodeNum*nodeNum/2.0) * 100.0;
    printf("\n\033[1;44m=== Graph Information ===\033[m\n");
    timerEnd("total", 0)
    printf("%d node, %d edge\ndensity = %lf%%\n", nodeNum, edgeNum, density);
    printf("total triangle: %lld\n", triNum);
    printf("\033[1;44m=== End of the Program ===\033[m\n\n");
    return 0;
}
