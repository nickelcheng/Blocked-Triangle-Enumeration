#include<cstdio>
#include<cstdlib>
#include<ctime>

int main(int argc, char *argv[]){
    if(argc != 3){
        printf("argc = %d\n", argc);
        fprintf(stderr, "usage: randomGraph <size> <density>\n");
        return 0;
    }

    int nodeNum = atoi(argv[1]);
    double density = atof(argv[2]);
    int edgeNum = 0;

    printf("%d\n", nodeNum);
    srand(time(NULL));
    for(int i = 0; i < nodeNum-1; i++){
        for(int j = i+1; j < nodeNum; j++){
            double p = (double)rand() / (double)RAND_MAX;
            if(p < density){
                edgeNum++;
                printf("%d %d\n", i, j);
            }
        }
    }

    fprintf(stderr, "%d nodes, %d edges\n", nodeNum, edgeNum);
    fprintf(stderr, "target density = %lf\n", density);
    fprintf(stderr, "real density = %lf\n", (double)(edgeNum*2)/(nodeNum*nodeNum));

    return 0;
}
