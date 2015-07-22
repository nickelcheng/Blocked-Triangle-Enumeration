#ifndef __DEGENERACY_H__
#define __DEGENERACY_H__

#include "main.h"

struct DegeneracyNode{
    vector< int > nei;
    int newOrder;
    int realDeg; // forward
    int leftDeg, inListPos; // edge iterator
    DegeneracyNode(void){
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
typedef struct DegeneracyNode DegeneracyNode;

struct DegList{
    vector< int > mem;
};
typedef struct DegList DegList;

void reorderByDegeneracy(int nodeNum, vector< Edge > &edge);
void buildDegList(vector< DegeneracyNode > &node, vector< DegList > &degList);
void reordering(vector< DegeneracyNode > &node, vector< DegList > &degList);
int findMinDegNode(int &currPos, vector< DegList > &degList);
void removeNode(int v, vector< DegeneracyNode > &node, vector< DegList > &degList);
void decDegByOne(int v, vector< DegeneracyNode > &node, vector< DegList > &degList);
void removeNodeInList(int v, vector< DegeneracyNode > &node, vector< DegList > &degList);


#endif
