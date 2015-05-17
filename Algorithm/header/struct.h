#ifndef __STRUCT_H__
#define __STRUCT_H__

#include<vector>

using namespace std;

typedef struct Node{
    vector< int > nei;
    int degree(void) const{
        return (int)nei.size();
    }
} Node;

typedef struct Edge{
    int u, v;
    Edge(int _u, int _v){
        u = _u, v = _v;
    }
    bool operator < (const Edge &a) const{
        if(u != a.u) return u < a.u;
        return v < a.v;
    }
} Edge;

typedef struct vector< Edge > Row;
typedef struct vector< Row > Matrix;

#endif
