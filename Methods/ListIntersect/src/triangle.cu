#include "triangle.h"
#include "tools.h"

int cpuCountTriNum(vector< Node > &node){
    int nodeNum = (int)node.size();
    int triNum = 0;
    for(int i = 0; i < nodeNum; i++){
        int deg = node[i].degree();
        for(int j = 0; j < deg; j++){
            int tar = node[i].nei[j];
            int sz1 = (int)node[i].nei.size();
            int sz2 = (int)node[tar].nei.size();
            triNum += intersectList(sz1, sz2, &node[i].nei[0], &node[tar].nei[0]);
        }
    }
    return triNum;
}

void listCopy(int **offset, int **edgeV, int edgeNum, vector< Node > &node){
    int nodeNum = (int)node.size();
    (*offset) = (int*)malloc(sizeof(int)*(nodeNum+1));
    (*edgeV) = (int*)malloc(sizeof(int)*edgeNum);

    (*offset)[0] = 0;
    for(int i = 0; i < nodeNum; i++){
        int deg = node[i].degree();
        (*offset)[i+1] = (*offset)[i] + deg;
        for(int j = 0; j < deg; j++){
            int idx = (*offset)[i] + j;
            (*edgeV)[idx] = node[i].nei[j];
        }
    }
}

void listCopyToDevice(vector< Node > &node, int edgeNum, int* d_offset, int* d_edgeV){
    int nodeNum = (int)node.size();
    int *h_offset, *h_edgeV;

    listCopy(&h_offset, &h_edgeV, edgeNum, node);

    cudaMemcpy(d_offset, h_offset, sizeof(int)*(nodeNum+1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeV, h_edgeV, sizeof(int)*edgeNum, cudaMemcpyHostToDevice);

    free(h_offset);
    free(h_edgeV);
}

__global__ void gpuCountTriNum(int *offset, int *edgeV, int *triNum, int nodeNum){
    int nodePerBlock = averageCeil(nodeNum, gridDim.x);
    for(int r = 0; r < nodePerBlock; r++){
        int nodeID = blockIdx.x*nodePerBlock + r;
        if(nodeID >= nodeNum) continue;
        int myOffset = offset[nodeID];
        int nextOffset = offset[nodeID+1];
        int deg = nextOffset - myOffset;
        int jobPerThread = averageCeil(deg, blockDim.x);

        // move node u's adj list to shared memory
        for(int i = 0; i < jobPerThread; i++){
            int idx = threadIdx.x*jobPerThread + i;
            if(idx < deg){
                shared[idx] = edgeV[myOffset+idx]; // adj[idx]
            }
        }
        __syncthreads();

        // counting triangle number
        shared[deg+threadIdx.x] = 0;
        for(int i = 0; i < jobPerThread; i++){
            int idx = threadIdx.x*jobPerThread + i;
            if(idx < deg){
                int v = shared[idx]; // adj[idx]
                int vNeiLen = offset[v+1] - offset[v];
                shared[deg+threadIdx.x] += intersectList(deg, vNeiLen, shared, &edgeV[offset[v]]); // threadTriNum[threadIdx.x]
            }
        }
        __syncthreads();

        if(threadIdx.x == 0){
           sumTriangle(triNum, &shared[deg]); 
        }
    }
}

__host__ __device__ int intersectList(int sz1, int sz2, int *l1, int *l2){
    int triNum = 0;
    for(int i = sz1-1, j = sz2-1; i >= 0 && j >= 0;){
        if(l1[i] > l2[j]) i--;
        else if(l1[i] < l2[j]) j--;
        else{
            i--, j--;
            triNum++;
        }
    }
    return triNum;
}

