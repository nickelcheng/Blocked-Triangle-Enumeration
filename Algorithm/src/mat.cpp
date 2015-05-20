#include "mat.h"
#include "tool.h"
#include "solve.h"

#include <cstring>

#include<cstdio>

long long mat(
    int device,
    const vector< Edge > &edge, int edgeRange,
    const vector< Edge > &target, int nodeNum,
    int threadNum, int blockNum
){
    EdgeMat edgeMat(edgeRange);
    TarMat tarMat(nodeNum);
    long long triNum = 0;

    edgeMat.initMat(edge);
    tarMat.initMat(target);
    
    if(device == CPU || tarMat.nodeNum > MAX_NODE_NUM_LIMIT)
        triNum = cpuCountMat(edgeMat, tarMat);

/*    else
        triNum = gpuCountTriangleMat(edgeMat, tarMat, threadNum, blockNum);*/


    return triNum;
}

long long cpuCountMat(const EdgeMat &edge, const TarMat &target){
    long long triNum = 0;
    // iterator through each row in edge
    for(int i = 0; i < edge.nodeNum; i++){
        int entryOffset = i / BIT_PER_ENTRY;
        int bit = i % BIT_PER_ENTRY;

        // iterator through each entry of the row
        for(int j = entryOffset; j < edge.entryNum; j++){
            // iterator through each bit
            UI content = edge.getContent(i, j);
            int bitOffset = j*BIT_PER_ENTRY;
            if(j == entryOffset){ // first entry of this round
                // elimate some bits
                for(int s = bit; s >= 0; s--, content/=2);
                bitOffset = i+1; // start from next node of i
            }
            for(int k = bitOffset; content > 0; k++, content/=2){
                if(content % 2 == 1){ // edge(i, k) exists
                    for(int e = 0; e < target.entryNum; e++){
                        UI e1 = target.getContent(i, e);
                        UI e2 = target.getContent(k, e);
                        long long tmp;
                        tmp = countOneBits(e1 & e2);
                        triNum += tmp;
                    }
                }
            }
        }
    }
    return triNum;
}

