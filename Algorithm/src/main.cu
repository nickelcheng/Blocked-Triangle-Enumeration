#include<cstdio>
#include<cstdlib>
#include "io.h"
#include "reorder.h"
#include "solve.h"
#include "block.h"
#include "timer.h"
#include "mat.h"

int assignProc, blockNum, threadNum;
long long triNum;
UC mask[BIT_PER_ENTRY], *d_mask;
UC oneBitNum[BIT_NUM_TABLE_SIZE], *d_oneBitNum;

int main(int argc, char *argv[]){
    if(argc < 3){
        fprintf(stderr, "usage: count <input_path> <block_size> (<reorder_or_not> <assign_proc> <block_num> <thread_num>)\n");
        return 0;
    }

    int blockSize = atoi(argv[2]);
    bool reorder = true;
    assignProc = UNDEF;
    blockNum = GPU_BLOCK_NUM;
    threadNum = GPU_THREAD_NUM;
    if(argc >= 4) reorder = (strcmp("true",argv[3])==0) ? true : false;
    if(argc >= 5) assignProc = atoi(argv[4]);
    if(argc >= 6) blockNum = atoi(argv[5]);
    if(argc >= 7) threadNum = atoi(argv[6]);

    vector< Edge > edge;

    int nodeNum = inputEdge(argv[1], edge);

    // resolve first cuda call slow timing issue
    cudaFree(0);

    timerInit(2)
    timerStart(0)

    int edgeNum = (int)edge.size();
//    double density = (double)edgeNum/((double)nodeNum*nodeNum/2.0) * 100.0;

    //timerStart(1)
    forwardReorder(nodeNum, edge, reorder);
    //timerEnd("reorder", 1)

    timerStart(1)
    createMask(mask, &d_mask);
    createOneBitNumTable(oneBitNum, &d_oneBitNum);
    triNum = 0;
    timerEnd("initial", 1)

    if(edgeNum <= EDGE_NUM_LIMIT){
        timerStart(1)
        ListArray listArr, *d_listArr;
        cudaMalloc((void**)&d_listArr, sizeof(ListArray));
        gTransBlock(edge, nodeNum, 0, 0, listArr, d_listArr);
        cudaFree(d_listArr);
        timerEnd("edge->list", 1)

        timerStart(1)
        scheduler(listArr, listArr, nodeNum);
        timerEnd("count", 1)
    }
    else{
        EdgeMatrix edgeBlock;
        vector< int > rowWidth;
        int blockDim = initEdgeBlock(edge, nodeNum, blockSize, edgeBlock, rowWidth);
        rowWidth.resize(blockDim);
//        fprintf(stderr, "blockDim: %d\n", blockDim);
        for(int i = 0; i < (int)rowWidth.size(); i++){
            rowWidth[i] *= blockSize;
//            printf("%d %d\n", i, rowWidth[i]);
        }

        timerStart(1)
        ListArrMatrix listArrBlock(blockDim);
        initListArrBlock(edgeBlock, rowWidth, blockDim, blockSize, listArrBlock);
        timerEnd("edge->list", 1)

        timerStart(1)
        findTriangle(listArrBlock, rowWidth, blockDim);
        timerEnd("count", 1)
    }

    timerEnd("total", 0)
    cudaFree(d_oneBitNum);
    cudaFree(d_mask);

//    fprintf(stderr, "%d node, %d edge, density = %lf%%\n", nodeNum, edgeNum, density);
    printf("total triangle: %lld\n", triNum);
    return 0;
}
