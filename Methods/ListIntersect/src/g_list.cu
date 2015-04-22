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
        fprintf(stderr, "algorithm should be forward, cover_forward, edge, cover_edge or cover\n");
        return 0;
    }

    timerInit(2)
    timerStart(0)

    int nodeNum = atoi(argv[3]);
    int threadNum = atoi(argv[4]);
    int blockNum = atoi(argv[5]);
    bool useVC = useVertexCover(algo);
    vector< Node > node;
    vector< Edge > edge;

//    timerStart(1)
    initNode(nodeNum, node, useVC);
    inputList(argv[2], edge);
//    timerEnd("input", 1)

//    timerStart(1)
    int maxDeg = reorder(algo, node, edge);
//    timerEnd("reordering", 1)

    int edgeNum = (int)edge.size();
    int *d_triNum, *d_offset, *d_edgeV;
    bool *d_inCoverSet;

//    timerStart(1)
    initDeviceTriNum((void**)&d_triNum);
    if(useVC)
        listCopyToDevice(node, edgeNum, (void**)&d_offset, (void**)&d_edgeV, (void**)&d_inCoverSet);
    else
        listCopyToDevice(node, edgeNum, (void**)&d_offset, (void**)&d_edgeV);
//    timerEnd("cuda copy", 1)

//    timerStart(1)
    int smSize = (threadNum+maxDeg) * sizeof(int);
    if(useVC)
        gpuCountTriNum<<< blockNum, threadNum, smSize >>>(d_offset, d_edgeV, d_triNum, nodeNum, d_inCoverSet);
    else
        gpuCountTriNum<<< blockNum, threadNum, smSize >>>(d_offset, d_edgeV, d_triNum, nodeNum);
    cudaDeviceSynchronize();
//    timerEnd("intersection", 1)

    int triNum;
    cudaMemcpy(&triNum, d_triNum, sizeof(int), cudaMemcpyDeviceToHost);
    printf("total triangle: %d\n", triNum);

    cudaFree(d_triNum);
    cudaFree(d_offset);
    cudaFree(d_edgeV);

    timerEnd("total", 0)

    return 0;
}

