#include<cstdio>
#include<cstdlib>
#include "io.h"
#include "reorder.h"
#include "solve.h"
#include "block.h"
#include "timer.h"
#include "mat.h"
#include "degeneracy.h"

int assignProc, blockNum, threadNum;
double densityBoundary;
int edgeNumLimit;
long long triNum;
UI mask[BIT_PER_ENTRY], *d_mask;
UC oneBitNum[BIT_NUM_TABLE_SIZE], *d_oneBitNum;

int main(int argc, char *argv[]){
    if(argc < 2){
        fprintf(stderr, "arguments:\n");
        fprintf(stderr, "  input_path\t\t\t(required)\n");
        fprintf(stderr, "  assign_proc\t\t\t(default:auto)\n");
        fprintf(stderr, "  density_boundary\t\t(default=%lf)\n", DENSITY_BOUNDARY);
        fprintf(stderr, "  block_size\t\t\t(default=%d)\n", DEFAULT_BLOCK_SIZE);
        fprintf(stderr, "  edge_num_limit\t\t(default=%d)\n", EDGE_NUM_LIMIT);
        fprintf(stderr, "  reorder_or_not\t\t(default=true)\n");
        fprintf(stderr, "  max_allowed_gpu_block_num\t(default=%d)\n", GPU_BLOCK_NUM);
        fprintf(stderr, "  max_allowed_thread_per_block\t(default=%d\n", GPU_THREAD_NUM);
        return 0;
    }

    assignProc = UNDEF;
    densityBoundary = DENSITY_BOUNDARY;
    int blockSize = DEFAULT_BLOCK_SIZE;
    edgeNumLimit = EDGE_NUM_LIMIT;
    bool reorder = true;
    blockNum = GPU_BLOCK_NUM;
    threadNum = GPU_THREAD_NUM;
    if(argc >= 3) assignProc = atoi(argv[2]);
    if(argc >= 4) densityBoundary = atof(argv[3]);
    if(argc >= 5) blockSize = atoi(argv[4]);
    if(argc >= 6) edgeNumLimit = atoi(argv[5]);
    if(argc >= 7) reorder = (strcmp("true",argv[6])==0) ? true : false;
    if(argc >= 8) blockNum = atoi(argv[7]);
    if(argc >= 9) threadNum = atoi(argv[8]);

    vector< Edge > edge;

    int nodeNum = inputEdge(argv[1], edge);

    // resolve first cuda call slow timing issue
    cudaFree(0);

    timerInit(2)

    int edgeNum = (int)edge.size();

//    timerStart(1)
//    forwardReorder(nodeNum, edge, reorder);
    reorderByDegeneracy(nodeNum, edge, reorder);
//    timerEnd("reorder", 1)

    timerStart(0)
    timerStart(1)
    createMask(mask, &d_mask);
    createOneBitNumTable(oneBitNum, &d_oneBitNum);
    triNum = 0;
    timerEnd("initial", 1)

    timerStart(1)
    EdgeMatrix edgeBlock;
    vector< int > rowWidth;
    int remain = nodeNum % blockSize;
    if(remain == 0) remain = blockSize;
    int blockDim = initEdgeBlock(edge, nodeNum, blockSize, remain, edgeBlock, rowWidth);
    rowWidth.resize(blockDim);
    timerEnd("split block", 1)
    printf("\033[1;32mdivide into %d subgraph(s)\033[m\n", blockDim);
    printf("subgraph size:");
    for(int i = 0; i < (int)rowWidth.size(); i++)
        printf(" %d", rowWidth[i]);
    printf("\n");

    if(blockDim == 1){
        timerStart(1)
        ListArray listArr, *d_listArr;
        cudaMalloc((void**)&d_listArr, sizeof(ListArray));
        gTransBlock(edge, nodeNum, 0, 0, listArr, d_listArr);
        cudaFree(d_listArr);
        timerEnd("edge->list", 1)

        timerStart(1)
        scheduler(listArr, listArr, nodeNum, true);
        timerEnd("count", 1)
    }

    else{
        timerStart(1)
        ListArrMatrix listArrBlock(blockDim);
        initListArrBlock(edgeBlock, rowWidth, blockDim, blockSize, listArrBlock);
        timerEnd("edge->list", 1)

        timerStart(1)
        findTriangle(listArrBlock, rowWidth, blockDim);
        timerEnd("count", 1)
    }

    cudaFree(d_oneBitNum);
    cudaFree(d_mask);

    double density = (double)edgeNum/((double)nodeNum*nodeNum/2.0) * 100.0;
    printf("\n\033[1;44m=== Graph Information ===\033[m\n");
    timerEnd("total time", 0)
    printf("%d node, %d edge\ndensity = %lf%%\n", nodeNum, edgeNum, density);
    printf("total triangle: %lld\n", triNum);
    printf("\033[1;44m=== End of the Program ===\033[m\n\n");
    return 0;
}
