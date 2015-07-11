#include "block.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>

void initListArrBlock(
    EdgeMatrix &edgeBlock, const vector< int > &rowWidth, int blockDim, int blockSize,
    ListArrMatrix &listArrBlock
){
    for(int i = 0; i < blockDim; i++){
        listArrBlock[i] = ListArrRow(blockDim);
    }
    ListArray *d_listArr;
    cudaMalloc((void**)&d_listArr, sizeof(ListArray));

    int *offset = new int[blockDim];
    offset[0] = 0;
    for(int i = 1; i < blockDim; i++){
        offset[i] = offset[i-1] + rowWidth[i-1];
    }

    for(int i = 0; i < blockDim; i++){
        for(int j = i; j < blockDim; j++){
            int vOffset = offset[j];
            if(i != j) vOffset -= rowWidth[i];
            int nodeNum = rowWidth[i];
            gTransBlock(edgeBlock[i][j], nodeNum, offset[i], vOffset, listArrBlock[i][j], d_listArr);
        }
    }

    cudaFree(d_listArr);
    delete [] offset;
}

void gTransBlock(
    const vector< Edge > &edge, int nodeNum, int uOffset, int vOffset,
    ListArray &listArr, ListArray *d_listArr
){
    int edgeNum = (int)edge.size();
    listArr.initArray(nodeNum, edgeNum);
    if(edgeNum == 0){
        setEmptyArray(nodeNum, listArr.nodeArr);
        return;
    }

    thrust::device_vector< Edge > d_edge = edge;
    thrust::sort(d_edge.begin(), d_edge.end());

    Edge *pd_edge = thrust::raw_pointer_cast(d_edge.data());
    int edgeBlock = edgeNum/1024;
    int edgeThread = (edgeNum<1024) ? edgeNum : 1024;
    if(edgeNum % 1024 != 0) edgeBlock++;
    int nodeBlock = nodeNum/1024;
    int nodeThread = (nodeNum<1024) ? nodeNum : 1024;
    if(nodeNum % 1024 != 0) nodeBlock++;

    relabelBlock<<< edgeBlock, edgeThread >>>(edgeNum, uOffset, vOffset, pd_edge);

    int *d_nodeArr, *d_edgeArr;
    cudaMalloc((void**)&d_nodeArr, sizeof(int)*(nodeNum+1));
    cudaMalloc((void**)&d_edgeArr, sizeof(int)*edgeNum);
    cudaMemcpy(&(d_listArr->nodeArr), &d_nodeArr, sizeof(int*), H2D);
    cudaMemcpy(&(d_listArr->edgeArr), &d_edgeArr, sizeof(int*), H2D);

    initNodeArr<<< nodeBlock, nodeThread >>>(nodeNum, d_listArr);
    edge2listArr<<< edgeBlock, edgeThread >>>(pd_edge, nodeNum, edgeNum, d_listArr);
    removeEmptyFlag<<< nodeBlock, nodeThread >>>(nodeNum, d_listArr);

    cudaMemcpy(listArr.nodeArr, d_nodeArr, sizeof(int)*(nodeNum+1), D2H);
    cudaMemcpy(listArr.edgeArr, d_edgeArr, sizeof(int)*edgeNum, D2H);
    cudaFree(d_nodeArr);
    cudaFree(d_edgeArr);
}

__global__ void relabelBlock(int edgeNum, int uOffset, int vOffset, Edge *edge){
    if(uOffset == 0 && vOffset == 0) return;
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    int threads = blockDim.x * gridDim.x;

    for(int i = idx; i < edgeNum; i+=threads){
        edge[i].u -= uOffset;
        edge[i].v -= vOffset;
    }
}

