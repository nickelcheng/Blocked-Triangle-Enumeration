#include<cstdio>
#include<cstdlib>
#include "timer.h"

int main(int argc, char *argv[]){
    if(argc != 3){
        fprintf(stderr, "usage: bruteForce <input_path> <node_num>\n");
        return 0;
    }

    int nodeNum = atoi(argv[2]);

    FILE *fp = fopen(argv[1], "r");

    int u, v;
    bool *edge = (bool*)malloc(nodeNum*nodeNum*sizeof(bool));
    while(fscanf(fp, "%d%d", &u, &v) != EOF){
        edge[u*nodeNum+v] = edge[v*nodeNum+u] = true;
    }
    fclose(fp);

    timerInit(1)
    timerStart(0)
    int triNum = 0;
    for(int i = 0; i < nodeNum; i++){
        for(int j = i+1; j < nodeNum; j++){
            if(!edge[i*nodeNum+j]) continue;
            for(int k = j+1; k < nodeNum; k++){
                if(edge[j*nodeNum+k] && edge[i*nodeNum+k]){
                    triNum++;
//                    printf("%d %d %d\n", i, j, k);
                }
            }
        }
    }
    timerEnd("total", 0)

    free(edge);

    printf("total triangle: %d\n", triNum);

    return 0;
}
