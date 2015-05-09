#include "matFunc.h"
#include "tools.h"
#include<cstdio>

void inputMat(const char *inFile, unsigned int *mat, int edgeSize, int entryNum){
    FILE *fp = fopen(inFile, "r");
    int u, v;
    memset(mat, 0, edgeSize);
    while(fscanf(fp, "%d%d", &u, &v) != EOF){
        setEdge(u, v);
        setEdge(v, u);
    }
    fclose(fp);
}

void matCopyToDevice(int nodeNum, void* mat, void* d_mat){
    int entryNum = averageCeil(nodeNum, BIT_PER_ENTRY);
    cudaMemcpy(d_mat, mat, entryNum*nodeNum*sizeof(UI), cudaMemcpyHostToDevice);
}

int cpuCountTriNum(int nodeNum, int nodePerTile, UI *mat){
    int triNum = 0;
    int round = averageCeil(nodeNum, nodePerTile);
    int entryNum = averageCeil(nodeNum, BIT_PER_ENTRY);
    int entryPerTile = nodePerTile / BIT_PER_ENTRY;
    for(int t = 0; t < round; t++){
        int st = t * entryPerTile;
        int ed = st + entryPerTile;
        if(ed > entryNum) ed = entryNum;
        for(int i = 0; i < nodeNum; i++){
            for(int j = i+1; j < nodeNum; j++){
                if(!getEdge(i, j)) continue;
                for(int k = st; k < ed; k++){
                    UI result = mat[i*entryNum+k] & mat[j*entryNum+k];
                    triNum += countOneBits(result);
                }
            }
        }
    }
    return triNum/3;
}

__global__ void gpuCountTriNum(UI *mat, int *triNum, int nodeNum, int nodePerTile){
    int entryNum = averageCeil(nodeNum, BIT_PER_ENTRY);
    int entryPerTile = nodePerTile / BIT_PER_ENTRY;
    int nodePerThread = averageCeil(nodeNum, blockDim.x);

    int tileNum = averageCeil(nodeNum, nodePerTile);
    int tilePerBlock = averageCeil(tileNum, gridDim.x);
    for(int r = 0; r < tilePerBlock; r++){
        int tileID = blockIdx.x*tilePerBlock + r;
        if(tileID >= tileNum) continue;
        int offset = tileID * entryPerTile;
        int tileLen = entryPerTile;
        if(offset+tileLen > entryNum) tileLen = entryNum - offset;

        // move adj matrix tiled area to shared memory
        for(int i = 0; i < nodePerThread; i++){
            int idx = threadIdx.x*nodePerThread + i;
            if(idx >= nodeNum) continue;
            for(int j = 0; j < tileLen; j++){
                shared[idx*entryPerTile+j] = mat[idx*entryNum+j+offset]; // adjMat[idx][j]
            }
        }
        __syncthreads();

        // counting triangle number
        int tileSize = entryPerTile*nodeNum;
        int tid = tileSize + threadIdx.x;
        shared[tid] = 0; // threadTriNum[tid]
        for(int i = 0; i < nodePerThread; i++){
            int idx = threadIdx.x*nodePerThread + i;
            if(idx >= nodeNum) continue;
            for(int j = 0; j < nodeNum; j++){
                if(idx == j || !getEdge(idx, j)) continue;
                for(int k = 0; k < tileLen; k++){
                    UI result = shared[idx*entryPerTile+k] & shared[j*entryPerTile+k];
                    shared[tid] += countOneBits(result); //threadTriNum[tid]
                }
            }
        }
        __syncthreads();

        if(threadIdx.x == 0){
            sumTriangle(triNum, (int*)&shared[tileSize]);
        }
    }
}

__host__ __device__ int countOneBits(UI tar){
    int ones = 0;
    for(; tar; tar/=2)
        ones += tar % 2;
    return ones;
}

