#include "io.h"
#include "solve.h"

#include<cstdio>
#include <cstdlib>

int main(int argc, char *argv[]){
    if(argc != 4){
        fprintf(stderr, "usage: proc <input_path> <node_num> <algo>\n");
        return 0;
    }

    int nodeNum = atoi(argv[2]);
    int algo = atoi(argv[3]);
    vector< Edge > edge;

    inputEdge(argv[1], edge);

    long long triNum = solveBlock(nodeNum, edge, algo);
    printf("total triangle: %lld\n", triNum);

    return 0;
}
