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

    initSetting(argc, argv, srcFile, tarFile, nodeMapFile, blockSize);
    nodeRange = input(srcFile, edge);
    node.resize(nodeRange);
    edgeRange = (int)edge.size();

    while(nodeRange > 0){
        initNodeNei(nodeRange, edgeRange, node, edge);
        reorder(nodeRange, edgeRange, node, edge);
        edgeRange = removeOutRangeEdge(nodeRange, edgeRange, edge);
        nodeRange -= blockSize;
    }

    sortNewEdge(node, edge);
    output(tarFile, nodeMapFile, node, edge);

    return 0;
}
