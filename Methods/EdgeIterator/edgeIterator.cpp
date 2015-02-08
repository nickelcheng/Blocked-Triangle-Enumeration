#include<cstdio>
#include<vector>
#include<algorithm>

#define swap(a,b) {int tmp=a;a=b,b=tmp;}

using namespace std;

typedef struct Node Node;
typedef struct Edge Edge;
typedef struct DegList DegList;
typedef struct Triangle Triangle;

struct Node{
    vector< int > nei;
    int leftDeg;
    int newOrder, inListPos;
    Node(void){
        nei.clear();
    }
    void addNei(int v){
        nei.push_back(v);
    }
    int degree(void)const{
        return (int)nei.size();
    }
};

struct Edge{
    int u, v;
};

struct DegList{
    vector< int > mem;
};

struct Triangle{
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

vector< Triangle > triList;
vector< int > oriOrder;

void input(const char *inFile, vector< Node > &node, vector< Edge > &edge);

void reorderByDegeneracy(vector< Node > &node, vector< Edge > &edge);
void buildDegList(vector< Node > &node, vector< DegList > &degList);
void reordering(vector< Node > &node, vector< DegList > &degList);
int findMinDegNode(int &currPos, vector< DegList > &degList);
void removeNode(int v, vector< Node > &node, vector< DegList > &degList);
void decDegByOne(int v, vector< Node > &node, vector< DegList > &degList);
void removeNodeInList(int v, vector< Node > &node, vector< DegList > &degList);

void updateGraph(vector< Edge > &edge, vector< Node > &node);
int intersectList(vector< int > &l1, vector< int > &l2, int a, int b);

int main(int argc, char *argv[]){
    if(argc != 2){
        fprintf(stderr, "usage: listIntersect <input_path>\n");
        return 0;
    }

    vector< Node > node;
    vector< Edge > edge;

    input(argv[1], node, edge);
    reorderByDegeneracy(node, edge);
    updateGraph(edge, node);
    
    int nodeNum = (int)node.size();
    int triNum = 0;
    triList.clear();
    for(int i = 0; i < nodeNum; i++){
        int deg = node[i].degree();
        for(int j = 0; j < deg; j++){
            int tar = node[i].nei[j];
            triNum += intersectList(node[i].nei, node[tar].nei, i, tar);
        }
    }
    fprintf(stderr, "total triangle: %d\n", triNum);

    for(int i = 0; i < triNum; i++){
        triList[i].a = oriOrder[triList[i].a];
        triList[i].b = oriOrder[triList[i].b];
        triList[i].c = oriOrder[triList[i].c];
        triList[i].sortNode();
    }
    sort(triList.begin(), triList.end());
    for(int i = 0; i < triNum; i++){
        printf("%d %d %d\n", triList[i].a, triList[i].b, triList[i].c);
    }

    return 0;
}

void input(const char *inFile, vector< Node > &node, vector< Edge > &edge){
    int nodeNum, edgeNum;
    FILE *fp = fopen(inFile, "r");

    fscanf(fp, "%d%d", &nodeNum, &edgeNum);
    node.resize(nodeNum);
    edge.resize(edgeNum);
    for(int i = 0; i < edgeNum; i++){
        int u, v;
        fscanf(fp, "%d%d", &u, &v);
        node[u].addNei(v);
        node[v].addNei(u);
        edge[i].u = u, edge[i].v = v;
    }

    for(int i = 0; i < nodeNum; i++)
        node[i].leftDeg = node[i].degree();

    fclose(fp);
}

void reorderByDegeneracy(vector< Node > &node, vector< Edge > &edge){
    int nodeNum = (int)node.size();
    vector< DegList > degList;

    degList.resize(nodeNum);
    buildDegList(node, degList);
    reordering(node, degList);
}

void buildDegList(vector< Node > &node, vector< DegList > &degList){
    int nodeNum = (int)node.size();
    for(int i = 0; i < nodeNum; i++){
        node[i].inListPos = (int)degList[node[i].degree()].mem.size();
        degList[node[i].degree()].mem.push_back(i);
    }
}

void reordering(vector< Node > &node, vector< DegList > &degList){
    int nodeNum = (int)node.size();
    int currPos = 0;
    oriOrder.resize(nodeNum);
    for(int i = 0; i < nodeNum; i++){
        int v = findMinDegNode(currPos, degList);
        removeNode(v, node, degList);
        node[v].newOrder = i;
        oriOrder[i] = v;
    }
}

int findMinDegNode(int &currPos, vector< DegList > &degList){
    if(currPos > 0 && !degList[currPos-1].mem.empty())
        currPos--;
    while(degList[currPos].mem.empty()) currPos++;
    return degList[currPos].mem.back();
}

void removeNode(int v, vector< Node > &node, vector< DegList > &degList){
    int deg = (int)node[v].nei.size();
    for(int i = 0; i < deg; i++){
        decDegByOne(node[v].nei[i], node, degList);
    }
    node[v].nei.clear();
    removeNodeInList(v, node, degList);
}

void decDegByOne(int v, vector< Node > &node, vector< DegList > &degList){
    if(node[v].inListPos == -1){
        return;
    }
    removeNodeInList(v, node, degList);
    node[v].leftDeg--;
    node[v].inListPos = (int)degList[node[v].leftDeg].mem.size();
    degList[node[v].leftDeg].mem.push_back(v);
}

void removeNodeInList(int v, vector< Node > &node, vector< DegList > &degList){
    int last = degList[node[v].leftDeg].mem.back();
    degList[node[v].leftDeg].mem[node[v].inListPos] = last;
    node[last].inListPos = node[v].inListPos;
    degList[node[v].leftDeg].mem.pop_back();
    node[v].inListPos = -1;
    
}

void updateGraph(vector< Edge > &edge, vector< Node > &node){
    int edgeNum = (int)edge.size();
    int nodeNum = (int)node.size();

    for(int i = 0; i < edgeNum; i++){
        edge[i].u = node[edge[i].u].newOrder;
        edge[i].v = node[edge[i].v].newOrder;
    }

    for(int i = 0; i < edgeNum; i++){
        int u = edge[i].u, v = edge[i].v;
        if(u < v) node[u].addNei(v);
        else node[v].addNei(u);
    }

    for(int i = 0; i < nodeNum; i++){
        sort(node[i].nei.begin(), node[i].nei.end());
    }
}

int intersectList(vector< int > &l1, vector< int > &l2, int a, int b){
    int sz1 = (int)l1.size();
    int sz2 = (int)l2.size();
    int triNum = 0;
    for(int i = 0, j = 0; i < sz1 && j < sz2;){
        if(l1[i] < l2[j]) i++;
        else if(l1[i] > l2[j]) j++;
        else{
            triList.push_back(Triangle(a,b,l1[i]));
            triNum++;
            i++, j++;
        }
    }
    return triNum;
}
