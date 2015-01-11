#ifndef __MY_MAIN_H__
#define __MY_MAIN_H__

#include <set>
#include <queue>
#include <functional>

using namespace std;

#define UNDEF -1
#define MAX_LEN 100

typedef struct Node{
    int oriOrder, nextOrder;
    int laterNeiNum;
    set< int > nei;
    Node(int order = UNDEF){
        oriOrder = order;
        nextOrder = UNDEF;
        nei.clear();
    }
    void addNei(int v){
        nei.insert(v);
    }
    void removeNei(int v){
        nei.erase(v);
    }
} Node;

typedef struct Edge{
    int u, v;
    Edge(int _u = UNDEF, int _v = UNDEF){
        u = _u, v = _v;
    }
    bool operator < (const Edge &a) const{
        if(u != a.u) return u < a.u;
        return v < a.v;
    }
    bool outOfRange(int range){
        return (u >= range || v >= range);
    }
} Edge;

typedef struct DegInfo{
    int deg, nodeID;
    DegInfo(int d, int id){
        deg = d, nodeID = id;
    }
    bool operator > (const DegInfo &a) const{
        if(deg != a.deg) return deg > a.deg;
        return nodeID > a.nodeID;
    }
} DegInfo;

typedef priority_queue< DegInfo, vector< DegInfo >, greater< DegInfo > > DegInfoPQ;

#endif
