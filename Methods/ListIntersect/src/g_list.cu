#include<cstdio>
#include<cstdlib>
#include "tools.h"
#include "timer.h"
#include "reorder.h"
#include "triangle.h"
#include "io.h"


int main(int argc, char *argv[]){
    if(argc != 6){
        fprintf(stderr, "usage: listIntersect <algorithm> <input_path> <node_num> <thread_per_block> <block_num>\n");
        return 0;
    }
    int algo = getAlgo(argv[1]);
    if(algo == -1){
        fprintf(stderr, "algorithm should be forward or edge\n");
        return 0;
    }

    timerInit(2)
    timerStart(0)

    int nodeNum = atoi(argv[3]);
    int threadNum = atoi(argv[4]);
    int blockNum = atoi(argv[5]);
    vector< Node > node(nodeNum);
    vector< Edge > edge;

    timerStart(1)
    inputList(argv[2], edge);
    timerEnd("input", 1)

    timerStart(1)
    int maxDeg = reorder(algo, node, edge);
    timerEnd("reordering", 1)

    int edgeNum = (int)edge.size();
    int triNum, *h_offset = NULL, *h_edgeV = NULL;
    int *d_triNum, *d_offset, *d_edgeV;
    
    listCopy(h_offset, h_edgeV, edgeNum, node);

    timerStart(1)
    initDeviceTriNum((void**)&d_triNum);
    listCopyToDevice(nodeNum, edgeNum, h_offset, (void**)&d_offset, h_edgeV, (void**)&d_edgeV);
    timerEnd("cuda copy", 1)

    timerStart(1)
    int smSize = (threadNum+maxDeg) * sizeof(int);
    gpuCountTriNum<<< blockNum, threadNum, smSize >>>(d_offset, d_edgeV, d_triNum, nodeNum);
    cudaDeviceSynchronize();
    timerEnd("intersection", 1)

    cudaMemcpy(&triNum, d_triNum, sizeof(int), cudaMemcpyDeviceToHost);
    printf("total triangle: %d\n", triNum);

    cudaFree(d_triNum);
    cudaFree(d_offset);
    cudaFree(d_edgeV);
    free(h_offset);
    free(h_edgeV);

    timerEnd("total", 0)

    return 0;
}

