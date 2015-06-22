#include<cstdio>
#include<cstdlib>
#include<cstring>
#include "io.h"
#include "reorder.h"
#include "solve.h"
#include "tool.h"
#include "block.h"
#include "threadHandler.h"

int assignProc, currTid;
pthread_t threads[MAX_THREAD_NUM];
bool threadUsed[MAX_THREAD_NUM];
pthread_mutex_t lock;
long long triNum;

int main(int argc, char *argv[]){
    if(argc != 3 && argc != 4){
        fprintf(stderr, "usage: count <input_path> <block_size> (<assign_proc>)\n");
        return 0;
    }

    int blockSize = atoi(argv[2]);
    if(argc == 4) assignProc = atoi(argv[3]);
    else assignProc = UNDEF;

    vector< Edge > edge;

    int nodeNum = inputEdge(argv[1], edge);

    // resolve first cuda call slow timing issue
    cudaFree(0);

    int edgeNum = (int)edge.size();
    double density = (double)edgeNum/((double)nodeNum*nodeNum/2.0) * 100.0;

    if(density > 0.01)
        forwardReorder(nodeNum, edge);

    EdgeMatrix edgeBlock;
    vector< int > rowWidth;
    int blockDim = initEdgeBlock(edge, nodeNum, blockSize, edgeBlock, rowWidth);
    rowWidth.resize(blockDim);
    fprintf(stderr, "blockDim: %d\n", blockDim);
    for(int i = 0; i < (int)rowWidth.size(); i++){
        rowWidth[i] *= blockSize;
//        printf("%d %d\n", i, rowWidth[i]);
    }

    ListArrMatrix listArrBlock(blockDim);
    initListArrBlock(edgeBlock, rowWidth, blockDim, blockSize, listArrBlock);

    pthread_mutex_init(&lock, NULL);

    BitMat::createMask();
    currTid = 0;
    triNum = 0;
    memset(threadUsed, false, MAX_THREAD_NUM);
    findTriangle(listArrBlock, rowWidth, blockDim);

    pthread_mutex_destroy(&lock);

    fprintf(stderr, "%d node, %d edge, density = %lf%%\n", nodeNum, edgeNum, density);
    printf("total triangle: %lld\n", triNum);
    return 0;
}
