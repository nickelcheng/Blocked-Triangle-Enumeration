#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<cmath>

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
fprintf(stderr, " %.3lf", cntTime(st[n],ed[n]));
//fprintf(stderr, "%s: %.3lf ms\n", tar, cntTime(st[n],ed[n]));

typedef unsigned int UI;

long long countOneBits(UI tar);

int main(int argc, char *argv[]){
    if(argc != 5){
        fprintf(stderr, "usage: tiledBit <input_path> <node_num> <node_per_tile> <thread_per_block\n");
        return 0;
    }

    timerInit(2)
    timerStart(0)

    int nodeNum = atoi(argv[2]);
    int nodePerTile = atoi(argv[3]);
    if(nodePerTile % BIT_PER_ENTRY != 0){
        fprintf(stderr, "node per tile must be multiple of %d\n", BIT_PER_ENTRY);
        return 0;
    }

    int entryNum = nodeNum / BIT_PER_ENTRY + 1;
    UI *edge = (UI*)malloc(entryNum*nodeNum*BIT_PER_ENTRY);

    timerStart(1)
    FILE *fp = fopen(argv[1], "r");
    int u, v;
    memset(edge, 0, entryNum*nodeNum*BIT_PER_ENTRY);
    while(fscanf(fp, "%d%d", &u, &v) != EOF){
        setEdge(u, v);
        setEdge(v, u);
    }
    fclose(fp);
    timerEnd("input", 1)

    long long h_triNum = 0, *d_triNum;
    UI *d_edge;

    timerStart(1)
    cudaMalloc((void**)&d_edge, entryNum*nodeNum*BIT_PER_ENTRY);
    cudaMalloc((void**)&d_triNum, sizeof(long long));
    cudaMemcyp(d_dege, edge, entryNum*nodeNum*BIT_PER_ENTRY, cudaMemcpyHostToDevice);
    cudaMemcpy(d_triNum, &h_triNum, sizeof(long long), cudaMemcpyHostToDevice);
    timerEnd("cuda copy", 1)

    timerStart(1)
    int thraedNum = atoi(4);
    int entryPerTile = nodePerTile / BIT_PER_ENTRY;
    int tileNum = entryNum / entryPerTile + 1;
    int smSize = entryPerTile * nodeNum * sizeof(UI);
    countTriNum<<< tileNum, threadNum, smSize >>>();
    cudaDeviceSynchronize();
    timerEnd("find triangle", 1)

    cudaMemcyp(&h_triNum, d_triNum, sizeof(long long), cudaMemcpyDeviceToHost);
/*    int round = (int)ceil((double)nodeNum/nodePerTile-0.001);
    for(int t = 0; t < round; t++){
        int st = t * entryPerTile;
        int ed = st + entryPerTile;
        if(ed > nodeNum) ed = nodeNum;
        for(int i = 0; i < nodeNum; i++){
            for(int j = i+1; j < nodeNum; j++){
                if(!getEdge(i, j)) continue;
                for(int k = st; k < ed; k++){
                    UI result = edge[i*entryNum+k] & edge[j*entryNum+k];
                    triNum += countOneBits(result);
                }
            }
        }
    }
    triNum /= 3;
    TimerEnd("find triangle", 1)*/
    printf("total triangle: %lld\n", h_triNum/3);

    cudaFree(d_triNum);
    cudaFree(d_edge);

    free(edge);

    timerEnd("total", 0)

    return 0;
}

__global__ void countTriNum(UI *edge, long long *triNum, int nodeNum, int nodePerTile){
    int round = (int)ceil((double)nodeNum/nodePerTile-0.001);
    int entryPerTile = nodePerTile / BIT_PER_ENTRY + 1;
    for(int t = 0; t < round; t++){
        int st = t * entryPerTile;
        int ed = st + entryPerTile;
        for(int i = 0; i < nodePerThread; i++){
            int idx = threadIdx.x*nodePerThread + i;
            for(int j = 0; j < nodeNum; j++){
                if(!getEdge(idx, j)) continue;
                for(int k = st; k < ed; k++){
                    UI result = edge[i*entryNum+k] & edge[j*entryNum+k];
                    atomicAdd(triNum, countOneBits(result));
                }
            }
        }
    }
}

__device__ long long countOneBits(UI tar){
    long long ones = 0;
    for(; tar; tar/=2)
        ones += tar % 2;
    return ones;
}
