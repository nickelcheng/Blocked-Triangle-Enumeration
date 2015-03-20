#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<cmath>

#define BIT_PER_ENTRY (sizeof(UI)*8)

#define getEdge(u,v) (((UI)1<<(v%BIT_PER_ENTRY))&edge[u*entryNum+(v/BIT_PER_ENTRY)])>0
#define setEdge(u,v) edge[u*entryNum+(v/BIT_PER_ENTRY)]|=((UI)1<<(v%BIT_PER_ENTRY))

typedef unsigned int UI;

long long countOneBits(UI tar);

int main(int argc, char *argv[]){
    if(argc != 4){
        fprintf(stderr, "usage: tiledBit <input_path> <node_num> <node_per_tile>\n");
        return 0;
    }

    int nodeNum = atoi(argv[2]);
    int nodePerTile = atoi(argv[3]);
    if(nodePerTile % BIT_PER_ENTRY != 0){
        fprintf(stderr, "node per tile must be multiple of %d\n", BIT_PER_ENTRY);
        return 0;
    }

    int entryNum = nodeNum / BIT_PER_ENTRY + 1;
    UI *edge = (UI*)malloc(entryNum*nodeNum*BIT_PER_ENTRY);

    FILE *fp = fopen(argv[1], "r");
    int u, v;
    memset(edge, 0, entryNum*nodeNum*BIT_PER_ENTRY);
    while(fscanf(fp, "%d%d", &u, &v) != EOF){
        setEdge(u, v);
        setEdge(v, u);
    }
    fclose(fp);

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
    printf("total triangle: %lld\n", triNum);

    free(edge);

    return 0;
}

long long countOneBits(UI tar){
    long long ones = 0;
    for(; tar; tar/=2)
        ones += tar % 2;
    return ones;
}
