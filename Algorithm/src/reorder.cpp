#include "reorder.h"
#include "main.h"
#include <cmath>
#include <omp.h>

void forwardReorder(int nodeNum, vector< Edge > &edge, bool reorder){
    int edgeNum = (int)edge.size();
    double density = (double)edgeNum/((double)nodeNum*nodeNum/2.0) * 100.0;
    //if(density < 0.01) return;

    if(reorder){
        extern int assignProc;
        // split line: sqrt(nodeNum) = -24.175 * ln(density) + 142.456
        // >: GPU, <=: CPU
        double sqrtN = sqrt(nodeNum);
        double x = log(density);
        double lhs = -24.175*x + 142.456;
        if(assignProc != 0 && sqrtN > lhs) gForwardReorder(nodeNum, edge);
        else cForwardReorder(nodeNum, edge);
    }
    for(int i = 0; i < edgeNum; i++){
        int u = edge[i].u;
        int v = edge[i].v;
        if(u >= v) edge[i].u = v, edge[i].v = u;
    }
}

void cForwardReorder(int nodeNum, vector< Edge > &edge){
    ForwardNode *node = new ForwardNode[nodeNum];
    vector< vector< int > > degList(nodeNum);

    #pragma omp parallel for
    for(int i = 0; i < nodeNum; i++)
        node[i].realDeg = 0;

    vector< Edge >::iterator e = edge.begin();
    for(; e != edge.end(); ++e){
        node[e->u].realDeg++;
        node[e->v].realDeg++;
    }

    for(int i = 0; i < nodeNum; i++){
        degList[node[i].realDeg].push_back(i);
    }

    vector< vector< int > >::iterator deg = degList.begin();
    for(int idx = 0; deg != degList.end(); ++deg){
        vector< int >::iterator n = deg->begin();
        for(; n != deg->end(); ++n){
            node[*n].order = idx++;
        }
    }

    int edgeNum = (int)edge.size();
    #pragma omp parallel for
    for(int i = 0; i < edgeNum; i++){
        edge[i].u = node[edge[i].u].order;
        edge[i].v = node[edge[i].v].order;
    }

    delete [] node;
}

