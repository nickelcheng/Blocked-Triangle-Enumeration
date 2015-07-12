#include<cstdio>
#include <cstdlib>
#include "io.h"
#include "reorder.h"
#include "solve.h"
#include "timer.h"
#include "threadHandler.h"


int assignProc, currTid, threadNum, blockNum;
pthread_t threads[MAX_THREAD_NUM];
bool threadUsed[MAX_THREAD_NUM];
pthread_mutex_t lock;
long long triNum;


int main(int argc, char *argv[]){
    if(argc != 3 && argc != 4 && argc != 6){
        fprintf(stderr, "usage: proc <assign_proc> <input_path> <reorder_or_not> <thread_per_block> <block_num>\n");
        return 0;
    }

    extern int assignProc, threadNum, blockNum;
    assignProc = atoi(argv[1]);
    bool reorder = true;

    if(argc >= 4) reorder = (strcmp("true",argv[3])==0) ? true : false;
    if(assignProc < LIST || assignProc > G_MAT){
        fprintf(stderr, "algo choice\n0: forward\n1: g_forward\n2: mat\n3: g_mat\n");
        return 0;
    }
    if(assignProc == G_LIST || assignProc == G_MAT){
        if(argc != 6){
            fprintf(stderr, "use default %d blocks, %d threads\n", GPU_BLOCK_NUM, GPU_THREAD_NUM);
            blockNum = GPU_BLOCK_NUM;
            threadNum = GPU_THREAD_NUM;
        }
        else{
            blockNum = atoi(argv[5]);
            threadNum = atoi(argv[4]);
        }
    }

    vector< Edge > edge;
    int nodeNum = inputEdge(argv[2], edge);
    
    // resolve first cuda call slow timing issue
    cudaFree(0);

    timerInit(1)
    timerStart(0)
    forwardReorder(nodeNum, edge, reorder);

    ListArray listArr, *d_listArr;
    cudaMalloc((void**)&d_listArr, sizeof(ListArray));
    if(assignProc == 0)
        cTransBlock(edge, nodeNum, 0, 0, listArr);
    else
        gTransBlock(edge, nodeNum, 0, 0, listArr, d_listArr);
    cudaFree(d_listArr);

    pthread_mutex_init(&lock, NULL);

    BitMat::createMask();
    currTid = 0;
    triNum = 0;
    memset(threadUsed, false, MAX_THREAD_NUM);
    scheduler(listArr, listArr, nodeNum, false);

    for(int i = 0; i < MAX_THREAD_NUM; i++) waitThread(i);

    pthread_mutex_destroy(&lock);
    timerEnd("total", 0)

    printf("total triangle: %lld\n", triNum);
    return 0;
}
