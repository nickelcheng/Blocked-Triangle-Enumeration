#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<sys/time.h>

#define getEdge(u,v) (((ULL)1<<(v%(sizeof(ULL)*8)))&edge[u*entryNum+(v/(sizeof(ULL)*8))])>0
#define setEdge(u,v) edge[u*entryNum+(v/(sizeof(ULL)*8))]|=((ULL)1<<(v%(sizeof(ULL)*8)))

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

typedef unsigned long long ULL;

__global__ void countTriNum(ULL *edge, int *triNum);

int main(int argc, char *argv[]){
    if(argc != 3){
        fprintf(stderr, "usage: bitwise <input_path> <node_num>\n");
        return 0;
    }

    timerInit(2)

    timerStart(0)

    int nodeNum = atoi(argv[2]);
    int entryNum = nodeNum/(sizeof(ULL)*8) + 1;
    printf("%d nodes, %d entrys\n", nodeNum, entryNum);
    ULL *edge = (ULL*)malloc(entryNum*nodeNum*(sizeof(ULL)*8));

    timerStart(1)
    FILE *fp = fopen(argv[1], "r");
    int u, v;
    memset(edge, 0, entryNum*nodeNum*(sizeof(ULL)*8));
    while(fscanf(fp, "%d%d", &u, &v) != EOF){
        setEdge(u, v);
        setEdge(v, u);
    }
    fclose(fp);
    timerEnd("input", 1)


    int h_triNum = 0, *d_triNum;
    ULL *d_edge;

    timerStart(1)
    cudaMalloc((void**)&d_edge, entryNum*nodeNum*(sizeof(ULL)*8));
    cudaMalloc((void**)&d_triNum, sizeof(int));
    cudaMemcpy(d_edge, edge, entryNum*nodeNum*(sizeof(ULL)*8), cudaMemcpyHostToDevice);
    cudaMemcpy(d_triNum, &h_triNum, sizeof(int), cudaMemcpyHostToDevice);
    timerEnd("cuda copy", 1)

    timerStart(1)
    int nT = entryNum < 1024 ? entryNum : 1024;
    printf("%d blocks, %d threads/bolck\n", nodeNum, nT);
    countTriNum<<< nodeNum, nT >>>(d_edge, d_triNum);
    cudaDeviceSynchronize();
    timerEnd("find triangle", 1)

    cudaMemcpy(&h_triNum, d_triNum, sizeof(int), cudaMemcpyDeviceToHost);
    printf("total triangle: %d\n", h_triNum/6);

    cudaFree(d_triNum);
    cudaFree(d_edge);

    free(edge);

    timerEnd("total", 0)

    return 0;
}

__global__ void countTriNum(ULL *edge, int *triNum){
    int i = blockIdx.x;
    int nodeNum = gridDim.x;
    int entryNum = nodeNum/(sizeof(ULL)*8) + 1;
    int nodePerThread = nodeNum/blockDim.x + 1;
    int stPos = threadIdx.x*nodePerThread;
    int tmp = 0, j, k;
    for(j = stPos; j < stPos+nodePerThread; j++){
        if(!getEdge(i, j)) continue;
        for(k = 0; k < nodeNum; k++){
            if(getEdge(j, k) && getEdge(k, i))
                tmp++;
        }
    }
    atomicAdd(triNum, tmp);
}

