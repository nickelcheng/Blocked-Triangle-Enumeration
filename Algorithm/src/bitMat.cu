#include "bitMat.h"

void createMask(UI *mask, UI **d_mask){
    cudaMalloc((void**)d_mask, sizeof(UI)*BIT_PER_ENTRY);
    for(int i = 0; i < (int)BIT_PER_ENTRY; i++)
        mask[i] = (UI)1 << i;
    cudaMemcpy(*d_mask, mask, sizeof(UI)*BIT_PER_ENTRY, H2D);
}

void cListArr2BitMat(const ListArray &src, BitMat **tar, UI **mat, int entryNum){
    BitMat *tarMat = new BitMat;
    tarMat->initMat(src, entryNum);
    cudaMalloc((void**)tar, sizeof(BitMat));
    cudaMemcpy(*tar, tarMat, sizeof(BitMat), H2D);
    cudaMalloc((void**)mat, sizeof(UI)*src.nodeNum*entryNum);
    cudaMemcpy(*mat, tarMat->mat, sizeof(UI)*src.nodeNum*entryNum, H2D);
    cudaMemcpy(&((*tar)->mat), mat, sizeof(UI*), H2D);
}
void gListArr2BitMat(const ListArray &src, BitMat **tar, UI **mat, int entryNum){
    ListArray *d_src;
    int *d_edgeArr, *d_nodeArr;

    // copy src to device
    cudaMalloc((void**)&d_src, sizeof(ListArray));
    cudaMemcpy(d_src, &src, sizeof(ListArray), H2D);
    // nodeArr
    cudaMalloc((void**)&d_nodeArr, sizeof(int)*(src.nodeNum+1));
    cudaMemcpy(d_nodeArr, src.nodeArr, sizeof(int)*(src.nodeNum+1), H2D);
    cudaMemcpy(&(d_src->nodeArr), &d_nodeArr, sizeof(int*), H2D);
    // edgeArr
    cudaMalloc((void**)&d_edgeArr, sizeof(int)*src.edgeNum);
    cudaMemcpy(d_edgeArr, src.edgeArr, sizeof(int)*src.edgeNum, H2D);
    cudaMemcpy(&(d_src->edgeArr), &d_edgeArr, sizeof(int*), H2D);

    // create mat on device
    cudaMalloc((void**)tar, sizeof(BitMat));
    cudaMalloc((void**)mat, sizeof(UI)*src.nodeNum*entryNum);
    cudaMemcpy(&((*tar)->mat), mat, sizeof(UI*), H2D);

    int threadNum = (entryNum<1024) ? entryNum : 1024;
    initMat<<< src.nodeNum, threadNum >>>(*mat, src.nodeNum, entryNum);

    extern UI *d_mask;
    int maxDeg = src.getMaxDegree();
    threadNum = (maxDeg<1024) ? maxDeg : 1024;
    listArr2BitMat<<< src.nodeNum, threadNum >>>(d_src, d_mask, *tar, *mat, entryNum);
    cudaFree(d_src);
    cudaFree(d_edgeArr);
    cudaFree(d_nodeArr);
}

__global__ void initMat(UI *mat, int nodeNum, int entryNum){
    for(int i = blockIdx.x; i < nodeNum; i+=gridDim.x){
        for(int j = threadIdx.x; j < entryNum; j+=blockDim.x){
            mat[j*nodeNum+i] = 0;
        }
    }
}

__global__ void listArr2BitMat(const ListArray *src, const UI *mask, BitMat *tar, UI *mat, int entryNum){
    tar->nodeNum = src->nodeNum;
    tar->entryNum = entryNum;
    int nodeNum = src->nodeNum;
    for(int u = blockIdx.x; u < nodeNum; u+=gridDim.x){
        int deg = src->getDeg(u);
        if(deg > 0){
            const int *nei = src->neiStart(u);
            for(int i = threadIdx.x; i < deg; i+=blockDim.x){
                int v = nei[i];
                int row = v / BIT_PER_ENTRY, col = u;
                int bit = v % BIT_PER_ENTRY;
                atomicOr(&mat[row*nodeNum+col], mask[bit]);
            }
        }
    }
}
