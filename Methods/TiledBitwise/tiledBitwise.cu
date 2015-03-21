#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<cmath>
#include<sys/time.h>

#define BIT_PER_ENTRY (sizeof(UI)*8)

#define getEdge(u,v) (((UI)1<<(v%BIT_PER_ENTRY))&edge[u*entryNum+(v/BIT_PER_ENTRY)])>0
#define setEdge(u,v) edge[u*entryNum+(v/BIT_PER_ENTRY)]|=((UI)1<<(v%BIT_PER_ENTRY))

#define cntTime(st,ed)\
((double)ed.tv_sec*1000000+ed.tv_usec-(st.tv_sec*1000000+st.tv_usec))/1000

#define timerInit(n)\
struct timeval st[n], ed[n];

#define timerStart(n)\
gettimeofday(&st[n], NULL);

#define timerEnd(tar, n)\
gettimeofday(&ed[n], NULL);\
//fprintf(stderr, " %.3lf", cntTime(st[n],ed[n]));
//fprintf(stderr, "%s: %.3lf ms\n", tar, cntTime(st[n],ed[n]));

typedef unsigned int UI;

__global__ void countTriNum(UI *edge, int *triNum, int nodeNum, int nodePerTile);
__device__ int countOneBits(UI tar);

extern __shared__ UI shared[]; // adjMat[entryPerTile*nodeNum], threadTriNum[threadPerBlock]

int main(int argc, char *argv[]){
    if(argc != 6){
        fprintf(stderr, "usage: tiledBit <input_path> <node_num> <node_per_tile> <thread_per_block> <block_num>\n");
        return 0;
    }

    timerInit(2)
    timerStart(0)

    int nodeNum = atoi(argv[2]);
    int nodePerTile = atoi(argv[3]);
    if(nodePerTile % BIT_PER_ENTRY != 0){
        fprintf(stderr, "node per tile must be multiple of %lu\n", BIT_PER_ENTRY);
        return 0;
    }

    int entryNum = (int)ceil((double)nodeNum/BIT_PER_ENTRY-0.001);
    UI *edge = (UI*)malloc(entryNum*nodeNum*sizeof(UI));

    timerStart(1)
    FILE *fp = fopen(argv[1], "r");
    int u, v;
    memset(edge, 0, entryNum*nodeNum*sizeof(UI));
    while(fscanf(fp, "%d%d", &u, &v) != EOF){
        setEdge(u, v);
        setEdge(v, u);
    }
    fclose(fp);
    timerEnd("input", 1)

    int triNum = 0, *d_triNum;
    UI *d_edge;

    timerStart(1)
    cudaMalloc((void**)&d_edge, entryNum*nodeNum*BIT_PER_ENTRY);
    cudaMalloc((void**)&d_triNum, sizeof(int));
    cudaMemcpy(d_edge, edge, entryNum*nodeNum*BIT_PER_ENTRY, cudaMemcpyHostToDevice);
    cudaMemcpy(d_triNum, &triNum, sizeof(int), cudaMemcpyHostToDevice);
    timerEnd("cuda copy", 1)

    timerStart(1)
	int threadPerBlock = atoi(argv[4]);
    int blockNum = atoi(argv[5]);
	int tileNum = (int)ceil((double)nodeNum/nodePerTile-0.001);
    int entryPerTile = nodePerTile / BIT_PER_ENTRY;
    int smSize = (entryPerTile*nodeNum + threadPerBlock) * sizeof(UI);
    countTriNum<<< blockNum, threadPerBlock, smSize >>>(d_edge, d_triNum, nodeNum, nodePerTile);
    cudaDeviceSynchronize();
    timerEnd("find triangle", 1)

    cudaMemcpy(&triNum, d_triNum, sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
    printf("total triangle: %d\n", triNum/6);

    cudaFree(d_triNum);
    cudaFree(d_edge);

    free(edge);

    timerEnd("total", 0)

    return 0;
}

__global__ void countTriNum(UI *edge, int *triNum, int nodeNum, int nodePerTile){
    int tileNum = (int)ceil((double)nodeNum/nodePerTile-0.001);
    int tilePerBlock = (int)ceil((double)tileNum/gridDim.x-0.001);
    for(int r = 0; r < tilePerBlock; r++){
        int tileID = blockIdx.x*tilePerBlock + r;
        if(tileID >= tileNum) continue;
        int entryNum = (int)ceil((double)nodeNum/BIT_PER_ENTRY-0.001);
        int entryPerTile = nodePerTile / BIT_PER_ENTRY;
    	int nodePerThread = (int)ceil((double)nodeNum/blockDim.x-0.001);
        int offset = tileID * entryPerTile;
        int bound = entryPerTile;
        if(offset+bound > nodeNum) bound = nodeNum - offset;

        // move adj matrix tiled area to shared memory
        for(int i = 0; i < nodePerThread; i++){
            int idx = threadIdx.x*nodePerThread + i;
            for(int j = 0; j < bound; j++){
                shared[idx*entryPerTile+j] = edge[idx*entryNum+j+offset]; // adjMat[idx][j]
            }
        }
        __syncthreads();

        // counting triangle number
        int tileSize = entryPerTile*nodeNum;
        int tid = tileSize + threadIdx.x;
        shared[tid] = 0; // threadTriNum[tid]
        for(int i = 0; i < nodePerThread; i++){
            int idx = threadIdx.x*nodePerThread + i;
            if(idx < nodeNum){
	            for(int j = 0; j < nodeNum; j++){
                    if(idx == j || !getEdge(idx, j)) continue;
               	    for(int k = 0; k < bound; k++){
                        UI result = shared[idx*entryPerTile+k] & shared[j*entryPerTile+k];
                        shared[tid] += countOneBits(result); //threadTriNum[tid]
                    }
                }
            }
        }
        __syncthreads();

        // sum triangle number
        if(threadIdx.x == 0){
            int tmp = 0;
            for(int i = 0; i < blockDim.x; i++){
                tmp += shared[tileSize+i]; // threadTriNum[i]
            }
            atomicAdd(triNum, tmp);
        }
    }
}

__device__ int countOneBits(UI tar){
    int ones = 0;
    for(; tar; tar/=2)
        ones += tar % 2;
    return ones;
}
