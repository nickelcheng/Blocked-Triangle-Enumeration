#include "main.h"
#include "io.h"
#include "reorder.h"
#include "tool.h"

int main(int argc, char *argv[]){
    char srcFile[MAX_LEN], tarFile[MAX_LEN], nodeMapFile[MAX_LEN];
    int blockSize;
    
    vector< Node > node;
    vector< Edge > edge;
    int nodeRange, edgeRange;

    try{
        initSetting(argc, argv, srcFile, tarFile, nodeMapFile, blockSize);
    } catch(const char *msg){
        fprintf(stderr, "%s", msg);
        return 0;
    }

    nodeRange = input(srcFile, edge);
    edgeRange = (int)edge.size();
    initNodeOriOrder(nodeRange, node);

    while(nodeRange > 0){
        initNodeNei(nodeRange, edgeRange, node, edge);
        try{
            reorder(nodeRange, edgeRange, node, edge);
        } catch(const char *msg){
            fprintf(stderr, "%s", msg);
            return 0;
        }
        nodeRange -= blockSize;
        edgeRange = removeOutRangeEdge(nodeRange, edgeRange, edge);
    }

    sortNewEdge(edge);
    output(tarFile, nodeMapFile, node, edge);

    return 0;
}
