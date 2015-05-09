#include<cstdio>
#include<cstdlib>
#include "tools.h"
#include "timer.h"
#include "reorder.h"
#include "triangle.h"
#include "io.h"


int main(int argc, char *argv[]){
    if(argc != 6){
        fprintf(stderr, "usage: g_list <algorithm> <input_path> <node_num> <thread_per_block> <block_num>\n");
        return 0;
    }
    int algo = getAlgo(argv[1]);
    if(algo == -1){
        fprintf(stderr, "algorithm should be forward or edge\n");
        return 0;
    }

    int nodeNum = atoi(argv[3]);
    int threadNum = atoi(argv[4]);
    int blockNum = atoi(argv[5]);
    vector< Node > node(nodeNum);
    vector< Edge > edge;

    inputList(argv[2], edge);

    int edgeNum = (int)edge.size();
    int *d_triNum, *d_offset, *d_edgeV;
cudaSetDevice(1);
    cudaMalloc((void**)&d_triNum, sizeof(int));
    cudaMalloc((void**)&d_offset, sizeof(int)*(nodeNum+1));
    cudaMalloc((void**)&d_edgeV, sizeof(int)*edgeNum);

    timerInit(1)
    timerStart(0)

    int maxDeg = reorder(algo, node, edge);

    initDeviceTriNum(d_triNum);
    listCopyToDevice(node, edgeNum, d_offset, d_edgeV);

    int smSize = (threadNum+maxDeg) * sizeof(int);
    gpuCountTriNum<<< blockNum, threadNum, smSize >>>(d_offset, d_edgeV, d_triNum, nodeNum);
    cudaDeviceSynchronize();

    int triNum;
    cudaMemcpy(&triNum, d_triNum, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    timerEnd("total", 0)

    cudaFree(d_triNum);
    cudaFree(d_offset);
    cudaFree(d_edgeV);

    printf("total triangle: %d\n", triNum);
    return 0;
}

