#include<cstdio>
#include<cstdlib>
#include "io.h"
#include "reorder.h"
#include "solve.h"
#include "block.h"
#include "threadHandler.h"
#include "timer.h"
#include "mat.h"

int assignProc, currTid;
int blockNum, threadNum;
pthread_t threads[MAX_THREAD_NUM];
bool threadUsed[MAX_THREAD_NUM];
pthread_mutex_t lock;
long long triNum;
unsigned char oneBitNum[BIT_NUM_TABLE_SIZE];

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

    timerInit(1)
    timerStart(0)

    int edgeNum = (int)edge.size();
//    double density = (double)edgeNum/((double)nodeNum*nodeNum/2.0) * 100.0;

    //timerStart(1)
    forwardReorder(nodeNum, edge, reorder);
    //timerEnd("reorder", 1)

    BitMat::createMask();
    createOneBitNumTable(oneBitNum);
    currTid = 0;
    triNum = 0;
    memset(threadUsed, false, MAX_THREAD_NUM);

    pthread_mutex_init(&lock, NULL);

    if(edgeNum <= EDGE_NUM_LIMIT){
        ListArray listArr, *d_listArr;
        cudaMalloc((void**)&d_listArr, sizeof(ListArray));
        gTransBlock(edge, nodeNum, 0, 0, listArr, d_listArr);
        cudaFree(d_listArr);

        //timerStart(1)
        scheduler(listArr, listArr, nodeNum, false);
        for(int i = 0; i < MAX_THREAD_NUM; i++) waitThread(i);
        //timerEnd("count", 1)
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

        ListArrMatrix listArrBlock(blockDim);
        initListArrBlock(edgeBlock, rowWidth, blockDim, blockSize, listArrBlock);

        //timerStart(1)
        findTriangle(listArrBlock, rowWidth, blockDim);
        //timerEnd("count", 1)
    }

    pthread_mutex_destroy(&lock);
    timerEnd("total", 0)

//    fprintf(stderr, "%d node, %d edge, density = %lf%%\n", nodeNum, edgeNum, density);
    printf("total triangle: %lld\n", triNum);
    return 0;
}
