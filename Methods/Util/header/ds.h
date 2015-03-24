#ifndef __DS_H__
#define __DS_H__

#include<vector>

//#define swap(a,b) {int tmp=a;a=b,b=tmp;}

using namespace std;

struct Edge{
    int u, v;
    Edge(int _u, int _v){
        u = _u, v = _v;
    }
};
typedef struct Edge Edge;

struct Node{
    vector< int > nei;
    int newOrder;
    int realDeg; // forward
    int leftDeg, inListPos; // edge iterator
    Node(void){
        realDeg = 0;
        nei.clear();
    }
    void addNei(int v){
        nei.push_back(v);
    }
    int degree(void)const{
        return (int)nei.size();
    }
};
typedef struct Node Node;

struct DegList{
    vector< int > mem;
};
typedef struct DegList DegList;

/*struct Triangle{
    int a, b, c;
    Triangle(int _a, int _b, int _c){
        a = _a, b = _b, c = _c;
    }
    bool operator < (const Triangle &t) const{
        if(a != t.a) return a < t.a;
        if(b != t.b) return b < t.b;
        return c < t.c;
    }
    void sortNode(void){
        if(a > b) swap(a,b);
        if(a > c) swap(a,c);
        if(b > c) swap(b,c);
    }
};
typedef struct Triangle Triangle;*/

#endif
