#include "list.h"
#include "binaryTree.h"
#include "solve.h"

void gpuCountTriangle(const ListArg &listArg){
    const ListArray &edge = *(listArg.edge);
    const ListArray &target = *(listArg.target);
    int maxDeg = listArg.maxDeg;

    long long *d_triNum, ans;
    ListArray *d_edge, *d_target;
    int *d_edge_edgeArr, *d_edge_nodeArr, *d_target_edgeArr, *d_target_nodeArr;
    
    int blockNum = GPU_BLOCK_NUM;
    if(blockNum > edge.nodeNum) blockNum = edge.nodeNum;
    cudaMalloc((void**)&d_triNum, sizeof(long long)*blockNum);

    // copy edge to device
    cudaMalloc((void**)&d_edge, sizeof(ListArray));
    cudaMemcpy(d_edge, &edge, sizeof(ListArray), H2D);
    // edge.nodeArr
    cudaMalloc((void**)&d_edge_nodeArr, sizeof(int)*(edge.nodeNum+1));
    cudaMemcpy(d_edge_nodeArr, edge.nodeArr, sizeof(int)*(edge.nodeNum+1), H2D);
    cudaMemcpy(&(d_edge->nodeArr), &d_edge_nodeArr, sizeof(int*), H2D);
    // edge.edgeArr
    cudaMalloc((void**)&d_edge_edgeArr, sizeof(int)*edge.edgeNum);
    cudaMemcpy(d_edge_edgeArr, edge.edgeArr, sizeof(int)*edge.edgeNum, H2D);
    cudaMemcpy(&(d_edge->edgeArr), &d_edge_edgeArr, sizeof(int*), H2D);

    // copy target to device
    cudaMalloc((void**)&d_target, sizeof(ListArray));
    cudaMemcpy(d_target, &target, sizeof(ListArray), H2D);
    // target.nodeArr
    cudaMalloc((void**)&d_target_nodeArr, sizeof(int)*(target.nodeNum+1));
    cudaMemcpy(d_target_nodeArr, target.nodeArr, sizeof(int)*(target.nodeNum+1), H2D);
    cudaMemcpy(&(d_target->nodeArr), &d_target_nodeArr, sizeof(int*), H2D);
    // target.edgeArr
    cudaMalloc((void**)&d_target_edgeArr, sizeof(int)*target.edgeNum);
    cudaMemcpy(d_target_edgeArr, target.edgeArr, sizeof(int)*target.edgeNum, H2D);
    cudaMemcpy(&(d_target->edgeArr), &d_target_edgeArr, sizeof(int*), H2D);

    if(listArg.delTar) delete &target;

    int smSize = maxDeg*sizeof(int);
    gpuCountList<<< blockNum, GPU_THREAD_NUM, smSize >>>(d_edge, d_target, d_triNum);
    sumTriangle<<< 1, 1 >>>(d_triNum, blockNum);
    cudaMemcpy(&ans, d_triNum, sizeof(long long), D2H);

    cudaFree(d_triNum);
    cudaFree(d_edge);
    cudaFree(d_edge_edgeArr);
    cudaFree(d_edge_nodeArr);
    cudaFree(d_target);
    cudaFree(d_target_edgeArr);
    cudaFree(d_target_nodeArr);

    extern long long triNum;
    extern pthread_mutex_t lock;
    pthread_mutex_lock(&lock);
    triNum += ans;
    pthread_mutex_unlock(&lock);
}

__global__ void gpuCountList(const ListArray *edge, const ListArray *target, long long *triNum){
    __shared__ long long threadTriNum[1024];
    int bound = nearestLessPowOf2(blockDim.x);

    triNum[blockIdx.x] = 0;
    // iterator through each edge (u, v)
    int range = edge->nodeNum;
    for(int u = blockIdx.x; u < range; u += gridDim.x){
        int uLen = target->getDeg(u);
        if(uLen > 0){
            const int *uList = target->neiStart(u);
        
            // move node u's adj list (in target) to shared memory
            int uDeg = edge->getDeg(u);
            for(int i = threadIdx.x; i < uLen; i += blockDim.x){
                uAdj[i] = uList[i];
            }
            __syncthreads();

            // counting triangle number
            threadTriNum[threadIdx.x] = 0;
            const int *uNei = edge->neiStart(u);
            for(int i = threadIdx.x; i < uDeg; i += blockDim.x){
                int v = uNei[i];
                int vLen = target->getDeg(v);
                const int *vList = target->neiStart(v);
                // intersect u list and v list in target
                threadTriNum[threadIdx.x] += intersectList(uLen, vLen, uAdj, vList);
            }
        }
        __syncthreads();

        binaryTreeSum(threadTriNum, blockDim.x, bound);
        if(threadIdx.x==0)
            triNum[blockIdx.x] += threadTriNum[0];

//        if(threadIdx.x==0)
//            triNum[blockIdx.x] += linearSum(threadTriNum, blockDim.x);

        __syncthreads();
    }
}

DECORATE long long intersectList(int sz1, int sz2, const int *l1, const int *l2){
    long long triNum = 0;
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

