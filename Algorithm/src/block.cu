#include "block.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>

void initListArrBlock(
    const EdgeMatrix &edgeBlock, const vector< int > &rowWidth, int blockDim, int blockSize,
    ListArrMatrix &listArrBlock
){
    for(int i = 0; i < blockDim; i++){
        listArrBlock[i] = ListArrRow(blockDim);
    }
    ListArray *d_listArr;
    int *d_nodeArr, *d_edgeArr;
    cudaMalloc((void**)&d_listArr, sizeof(ListArray));

    int *offset = new int[blockDim];
    offset[0] = 0;
    for(int i = 1; i < blockDim; i++){
        offset[i] = offset[i-1] + rowWidth[i-1];
        offset[i] *= blockSize;
    }

    for(int i = 0; i < blockDim; i++){
        for(int j = i; j < blockDim; j++){
            thrust::device_vector< Edge > d_edge = edgeBlock[i][j];
            thrust::sort(d_edge.begin(), d_edge.end());

            int nodeNum = rowWidth[i]*blockSize;
            int edgeNum = (int)edgeBlock[i][j].size();
            Edge *pd_edge = thrust::raw_pointer_cast(d_edge.data());
            relabelBlock<<< 1, 1 >>>(edgeNum, offset[i], offset[j], pd_edge);

            cudaMalloc((void**)&d_nodeArr, sizeof(int)*(nodeNum+1));
            cudaMalloc((void**)&d_edgeArr, sizeof(int)*edgeNum);
            cudaMemcpy(&(d_listArr->nodeArr), &d_nodeArr, sizeof(int*), H2D);
            cudaMemcpy(&(d_listArr->edgeArr), &d_edgeArr, sizeof(int*), H2D);
            edge2listArr<<< 1, 1 >>>(pd_edge, nodeNum, edgeNum, d_listArr);

            listArrBlock[i][j].initArray(nodeNum, edgeNum);
            cudaMemcpy(listArrBlock[i][j].nodeArr, d_nodeArr, sizeof(int)*(nodeNum+1), D2H);
            cudaMemcpy(listArrBlock[i][j].edgeArr, d_edgeArr, sizeof(int)*edgeNum, D2H);

            cudaFree(d_nodeArr);
            cudaFree(d_edgeArr);
        }
    }

    cudaFree(d_listArr);
    delete [] offset;
}

__global__ void relabelBlock(int edgeNum, int uOffset, int vOffset, Edge *edge){
    if(edgeNum == 0) return;
    if(uOffset == 0 && vOffset == 0) return;
    for(int i = 0; i < edgeNum; i++){
        edge[i].u -= uOffset;
        edge[i].v -= vOffset;
    }
}

