#include "bitMat.h"
#include <omp.h>

void createMask(UC *mask, UC **d_mask){
    cudaMalloc((void**)d_mask, sizeof(UC)*BIT_PER_ENTRY);
    for(int i = 0; i < (int)BIT_PER_ENTRY; i++)
        mask[i] = (UC)1 << i;
    cudaMemcpy(*d_mask, mask, sizeof(UC)*BIT_PER_ENTRY, H2D);
}

void cListArr2BitMat(const ListArray &src, BitMat **tar, UC **mat, int entryNum){
    BitMat *tarMat = new BitMat;
    tarMat->initMat(src, entryNum);
    cudaMalloc((void**)tar, sizeof(BitMat));
    cudaMemcpy(*tar, tarMat, sizeof(BitMat), H2D);
    cudaMalloc((void**)mat, sizeof(UC)*src.nodeNum*entryNum);
    cudaMemcpy(*mat, tarMat->mat, sizeof(UC)*src.nodeNum*entryNum, H2D);
    cudaMemcpy(&((*tar)->mat), mat, sizeof(UC*), H2D);
    delete tarMat;
}

void pListArr2BitMat(const ListArray &src, BitMat **tar, UC **mat, int entryNum){
    BitMat *tarMat = new BitMat;
    tarMat->nodeNum = src.nodeNum;
    tarMat->entryNum = entryNum;
    tarMat->mat = new UC[sizeof(UC)*entryNum*src.nodeNum];
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < src.nodeNum; i++){
        for(int j = 0; j < entryNum; j++){
            tarMat->mat[j*src.nodeNum+i] = 0;
        }
    }

    extern UC mask[BIT_PER_ENTRY];
    #pragma omp parallel for
    for(int i = 0; i < src.nodeNum; i++){
        for(int j = src.nodeArr[i]; j < src.nodeArr[i+1]; j++){
            int u = i, v = src.edgeArr[j];
            int row = v / BIT_PER_ENTRY, col = u;
            int bit = v % BIT_PER_ENTRY;
            tarMat->mat[row*src.nodeNum+col] |= mask[bit];
        }
    }

    cudaMalloc((void**)tar, sizeof(BitMat));
    cudaMemcpy(*tar, tarMat, sizeof(BitMat), H2D);
    cudaMalloc((void**)mat, sizeof(UC)*entryNum*src.nodeNum);
    cudaMemcpy(*mat, tarMat->mat, sizeof(UC)*entryNum*src.nodeNum, H2D);
    cudaMemcpy(&((*tar)->mat), mat, sizeof(UC*), H2D);
    delete tarMat;
}

void gListArr2BitMat(const ListArray &src, BitMat **tar, UC **mat, int entryNum){
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
    cudaMalloc((void**)mat, sizeof(UC)*src.nodeNum*entryNum);
    cudaMemcpy(&((*tar)->mat), mat, sizeof(UC*), H2D);

    int threadNum = (entryNum<1024) ? entryNum : 1024;
    initMat<<< src.nodeNum, threadNum >>>(src.nodeNum, entryNum, *tar, *mat);

    extern UC *d_mask;
    listArr2BitMat<<< src.nodeNum/32+1, 32 >>>(d_src, d_mask, *mat);

    cudaFree(d_src);
    cudaFree(d_edgeArr);
    cudaFree(d_nodeArr);
}

__global__ void initMat(int nodeNum, int entryNum, BitMat *tar, UC *mat){
    tar->nodeNum = nodeNum;
    tar->entryNum = entryNum;
    for(int i = blockIdx.x; i < nodeNum; i+=gridDim.x){
        for(int j = threadIdx.x; j < entryNum; j+=blockDim.x){
            mat[j*nodeNum+i] = 0;
        }
    }
}

__global__ void listArr2BitMat(const ListArray *src, const UC *mask, UC *mat){
    int u = threadIdx.x + blockIdx.x*blockDim.x;
    int nodeNum = src->nodeNum;
    if(u < nodeNum){
        int deg = src->getDeg(u);
        if(deg > 0){
            const int *nei = src->neiStart(u);
            for(int i = 0; i < deg; i++){
                int v = nei[i];
                int row = v / BIT_PER_ENTRY, col = u;
                int bit = v % BIT_PER_ENTRY;
                mat[row*nodeNum+col] |= mask[bit];
            }
        }
    }
}
