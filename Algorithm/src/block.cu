#include "block.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>

void initListArrBlock(
    const EdgeMatrix &edgeBlock, const vector< int > &rowWidth, int blockDim,
    ListArrMatrix &listArrBlock
){
    for(int i = 0; i < blockDim; i++){
        for(int j = 1; j < blockDim; j++){
            thrust::device_vector< Edge > d_edge = edgeBlock[i][j];
            thrust::sort(d_edge.begin(), d_edge.end());
        }
    }
/*    for(int i = 0; i < blockDim; i++){
        listArrBlock[i] = ListArrRow(blockDim);
    }*/
}
