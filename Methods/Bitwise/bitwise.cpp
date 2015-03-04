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
fprintf(stderr, "%s: %.3lf ms\n", tar, cntTime(st[n],ed[n]));
//fprintf(stderr, " %.3lf", cntTime(st[n],ed[n]));

typedef unsigned long long ULL;

int main(int argc, char *argv[]){
    if(argc != 3){
        fprintf(stderr, "usage: bitwise <input_path> <node_num>\n");
        return 0;
    }

    timerInit(2)

    timerStart(0)

    int nodeNum = atoi(argv[2]);
    int entryNum = nodeNum/(sizeof(ULL)*8) + 1;
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

    timerStart(1)
    int triNum = 0;
    for(int i = 0; i < nodeNum; i++){
        for(int j = i+1; j < nodeNum; j++){
            if(!getEdge(i, j)) continue;
            for(int k = j+1; k < nodeNum; k++){
                if(getEdge(j, k) && getEdge(k, i)){
                    triNum++;
//                    printf("%d %d %d\n", i, j, k);
                }
            }
        }
    }
    printf("total triangle: %d\n", triNum);
    timerEnd("find triangle", 1)

    free(edge);

    timerEnd("total", 0)

    return 0;
}
