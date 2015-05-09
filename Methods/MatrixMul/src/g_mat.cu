#include<cstdio>
#include<cstdlib>
#include "tools.h"
#include "timer.h"
#include "matFunc.h"


int main(int argc, char *argv[]){
    if(argc != 6){
        fprintf(stderr, "usage: tiledBit <input_path> <node_num> <node_per_tile> <thread_per_block> <block_num>\n");
        return 0;
    }

    int nodeNum = atoi(argv[2]);
    int nodePerTile = atoi(argv[3]);
	int threadPerBlock = atoi(argv[4]);
    int blockNum = atoi(argv[5]);
    if(nodePerTile % BIT_PER_ENTRY != 0){
        fprintf(stderr, "node per tile must be multiple of %lu\n", BIT_PER_ENTRY);
        return 0;
    }
    int entryPerTile = nodePerTile / BIT_PER_ENTRY;

    int entryNum = averageCeil(nodeNum, BIT_PER_ENTRY);
    UI *mat = (UI*)malloc(entryNum*nodeNum*sizeof(UI));

    inputMat(argv[1], mat, entryNum*nodeNum*sizeof(UI), entryNum);

    int triNum, *d_triNum;
    UI *d_mat;
cudaSetDevice(1);
    cudaMalloc((void**)&d_triNum, sizeof(int));
    cudaMalloc((void**)&d_mat, entryNum*nodeNum*sizeof(UI));

    timerInit(1)
    timerStart(0)

    initDeviceTriNum(d_triNum);
    matCopyToDevice(nodeNum, mat, d_mat);

    int smSize = (entryPerTile*nodeNum + threadPerBlock) * sizeof(UI);
    gpuCountTriNum<<< blockNum, threadPerBlock, smSize >>>(d_mat, d_triNum, nodeNum, nodePerTile);
    cudaDeviceSynchronize();

    cudaMemcpy(&triNum, d_triNum, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    timerEnd("total", 0)

    cudaFree(d_triNum);
    cudaFree(d_mat);
    free(mat);

    printf("total triangle: %d\n", triNum/6);

    return 0;
}

