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

long long countOneBits(UI tar);

int main(int argc, char *argv[]){
    if(argc != 4){
        fprintf(stderr, "usage: tiledBit <input_path> <node_num> <node_per_tile>\n");
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

    timerStart(1)
    long long triNum = 0;
    int entryPerTile = nodePerTile / BIT_PER_ENTRY;
    int round = (int)ceil((double)nodeNum/nodePerTile-0.001);
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
    timerEnd("find triangle", 1)
    printf("total triangle: %lld\n", triNum);

    free(edge);

    timerEnd("total", 0)

    return 0;
}

long long countOneBits(UI tar){
    long long ones = 0;
    for(; tar; tar/=2)
        ones += tar % 2;
    return ones;
}
