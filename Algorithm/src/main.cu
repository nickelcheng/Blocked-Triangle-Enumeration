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
        fprintf(stderr, "  density_boundary\t\t(default=0.06)\n");
        fprintf(stderr, "  block_size\t\t\t(default=3072)\n");
        fprintf(stderr, "  edge_num_limit\t\t(default=20M)\n");
        fprintf(stderr, "  reorder_or_not\t\t(default=true)\n");
        fprintf(stderr, "  assign_proc\t\t\t(default:auto)\n");
        fprintf(stderr, "  max_allowed_gpu_block_num\t(default=16000)\n");
        fprintf(stderr, "  max_allowed_thread_per_block\t(default=1024\n");
        return 0;
    }

    densityBoundary = DENSITY_BOUNDARY;
    int blockSize = DEFAULT_BLOCK_SIZE;
    edgeNumLimit = EDGE_NUM_LIMIT;
    bool reorder = true;
    assignProc = UNDEF;
    blockNum = GPU_BLOCK_NUM;
    threadNum = GPU_THREAD_NUM;
    if(argc >= 3) densityBoundary = atof(argv[2]);
    if(argc >= 4) blockSize = atoi(argv[3]);
    if(argc >= 5) edgeNumLimit = atoi(argv[4]);
    if(argc >= 6) reorder = (strcmp("true",argv[5])==0) ? true : false;
    if(argc >= 7) assignProc = atoi(argv[6]);
    if(argc >= 8) blockNum = atoi(argv[7]);
    if(argc >= 9) threadNum = atoi(argv[8]);

    vector< Edge > edge;

    int nodeNum = inputEdge(argv[1], edge);

    // resolve first cuda call slow timing issue
    cudaFree(0);

    timerInit(2)
    timerStart(0)

    int edgeNum = (int)edge.size();
    double density = (double)edgeNum/((double)nodeNum*nodeNum/2.0) * 100.0;

    timerStart(1)
    //forwardReorder(nodeNum, edge, reorder);
    reorderByDegeneracy(nodeNum, edge);
    timerEnd("reorder", 1)

    timerStart(1)
    createMask(mask, &d_mask);
    createOneBitNumTable(oneBitNum, &d_oneBitNum);
    triNum = 0;
    timerEnd("initial", 1)

/*    if(edgeNum <= EDGE_NUM_LIMIT){
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
    else{*/
        EdgeMatrix edgeBlock;
        vector< int > rowWidth;
        int remain = nodeNum % blockSize;
        if(remain == 0) remain = blockSize;
        int blockDim = initEdgeBlock(edge, nodeNum, blockSize, remain, edgeBlock, rowWidth);
        rowWidth.resize(blockDim);
        printf("divide into %d subgraph(s):", blockDim);
        for(int i = 0; i < (int)rowWidth.size(); i++){
            printf(" %d", rowWidth[i]);
        }

        timerStart(1)
        ListArrMatrix listArrBlock(blockDim);
        initListArrBlock(edgeBlock, rowWidth, blockDim, blockSize, listArrBlock);
        timerEnd("edge->list", 1)

        timerStart(1)
        findTriangle(listArrBlock, rowWidth, blockDim);
        timerEnd("count", 1)
//    }

    timerEnd("total", 0)
    cudaFree(d_oneBitNum);
    cudaFree(d_mask);

    fprintf(stderr, "%d node, %d edge, density = %lf%%\n", nodeNum, edgeNum, density);
    printf("total triangle: %lld\n", triNum);
    return 0;
}
