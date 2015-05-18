#include "solve.h"
#include "list.h"
#include "mat.h"
#include<cstdio>

long long solveBlock(int blockSize, vector< Edge > &edge, int algo, int blockNum, int threadNum){
    if(edge.empty()) return 0;
    int method;
    if(algo < FORWARD || algo > G_MAT)
        method = scheduler(blockSize, (int)edge.size());
    else method = algo;

    if(method == FORWARD){
//        printf("use forward\n");
        long long triNum = forward(CPU, blockSize, edge);
        return triNum;
    }
    else if(method == G_FORWARD){
//        printf("use g_forward\n");
        long long triNum = forward(GPU, blockSize, edge, threadNum, blockNum);
        return triNum;
    }
    else if(method == MAT){
//        printf("use mat\n");
        long long triNum = mat(CPU, blockSize, edge);
        return triNum;
    }
    else if(method == G_MAT){
//        printf("use g_mat\n");
        long long triNum = mat(GPU, blockSize, edge, threadNum, blockNum);
        return triNum;
    }
    return FORWARD;
}

int scheduler(int nodeNum, int edgeNum){
    return FORWARD;
}

long long mergeBlock(vector< Matrix > &block, int x, int y, int blockSize){
    vector< Edge > edge;
    edge.insert(edge.end(), block[x][x].begin(), block[x][x].end());
    edge.insert(edge.end(), block[y][y].begin(), block[y][y].end());
    edge.insert(edge.end(), block[x][y].begin(), block[x][y].end());
    if(edge.empty()) return 0;

    vector< Edge >::iterator e = block[x][y].begin();
    for(int i = 0; e != block[x][y].end(); ++e, i++){
        edge.push_back(Edge(e->v, e->u));
    }
    int nodeNum = 2 * blockSize;
    int edgeNum = (int)edge.size();

    int *nodeArr = new int[sizeof(int)*(nodeNum+1)];
    int *edgeArr = new int[sizeof(int)*edgeNum];

    initArray(nodeArr, edgeArr, edge, nodeNum, edgeNum);

    long long triNum = 0;
    for(e = block[x][y].begin(); e != block[x][y].end(); ++e){
        int u = e->u, v = e->v;
        int uDeg = getDeg(nodeArr, u);
        int vDeg = getDeg(nodeArr, v);
        long long tmp;
        tmp = intersectList(uDeg, vDeg, &edgeArr[nodeArr[u]], &edgeArr[nodeArr[v]]);
        triNum += tmp;
/*        int realU = u + blockSize*x;
        int realV = v + blockSize*(y-1);
        printf("2. intersect %d & %d => %lld\n", realU, realV, tmp);*/
    }

    delete [] nodeArr;
    delete [] edgeArr;

    return triNum;
}

long long intersectBlock(vector< Matrix > &block, int x, int y, int z, int blockSize){
    vector< Edge > edge;
    edge.insert(edge.end(), block[x][z].begin(), block[x][z].end());
    edge.insert(edge.end(), block[y][z].begin(), block[y][z].end());
    if(edge.empty()) return 0;

    int nodeNum = 2 * blockSize;
    int edgeNum = (int)edge.size();
    int *nodeArr = new int[sizeof(int)*(nodeNum+1)];
    int *edgeArr = new int[sizeof(int)*edgeNum];

    initArray(nodeArr, edgeArr, edge, nodeNum, edgeNum);
    
    long long triNum = 0;
    vector< Edge >::iterator e = block[x][y].begin();
    for(; e != block[x][y].end(); ++e){
        int u = e->u, v = e->v;
        int uDeg = getDeg(nodeArr, u);
        int vDeg = getDeg(nodeArr, v);
        long long tmp;
        tmp = intersectList(uDeg, vDeg, &edgeArr[nodeArr[u]], &edgeArr[nodeArr[v]]);
        triNum += tmp;
/*        int realU = u + blockSize*x;
        int realV = v + blockSize*(y-1);
        printf("3. intersect %d & %d => %lld\n", realU, realV, tmp);*/
    }

    delete [] nodeArr;
    delete [] edgeArr;

    return triNum;
}

