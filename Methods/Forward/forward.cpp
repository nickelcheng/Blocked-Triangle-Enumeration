#include<cstdio>
#include<cstdlib>
#include<vector>
#include<algorithm>

#define swap(a,b) {int tmp = a; a = b, b = tmp;}

using namespace std;

typedef struct Node Node;
typedef struct Edge Edge;
typedef struct Triangle Triangle;

struct Node{
    vector< int > largerDegNei;
    int realDeg;
    int newOrder;
    Node(void){
        realDeg = 0;
        largerDegNei.clear();
    }
    void addNei(int v){
        largerDegNei.push_back(v);
    }
    int degree(void) const{
        return (int)largerDegNei.size();
    }
};

struct Edge{
    int u, v;
    Edge(int _u, int _v){
        u = _u, v = _v;
    }
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

/*vector< Triangle > triList;
vector< int > oriOrder;*/

void input(const char *inFile, vector< Node > &node, vector< Edge > &edge);
void reorderByDegree(vector< Node > &node, vector< Edge > &edge);
void initDegList(vector< Node > &node, vector< Edge > &edge);
int intersectList(vector< int > &l1, vector< int > &l2, int a, int b);

int main(int argc, char *argv[]){
    if(argc != 3){
        fprintf(stderr, "usage: forward <input_path> <node_num>\n");
        return 0;
    }

    int nodeNum = atoi(argv[2]);
    vector< Node > node(nodeNum);
    vector< Edge > edge;

    input(argv[1], node, edge);
    reorderByDegree(node, edge);
    initDegList(node, edge);

    int triNum = 0;
    for(int i = 0; i < nodeNum; i++){
        int deg = node[i].degree();
        for(int j = 0; j < deg; j++){
            int tar = node[i].largerDegNei[j];
            triNum += intersectList(node[i].largerDegNei, node[tar].largerDegNei, i, tar);
        }
    }
    fprintf(stderr, "total triangle: %d\n", triNum);

/*    for(int i = 0; i < triNum; i++){
        triList[i].a = oriOrder[triList[i].a];
        triList[i].b = oriOrder[triList[i].b];
        triList[i].c = oriOrder[triList[i].c];
        triList[i].sortNode();
    }
    sort(triList.begin(), triList.end());
    for(int i = 0; i < triNum; i++){
        printf("%d %d %d\n", triList[i].a, triList[i].b, triList[i].c);
    }*/


    return 0;
}

void input(const char *inFile, vector< Node > &node, vector< Edge > &edge){
    FILE *fp = fopen(inFile, "r");

    int u, v;
    while(fscanf(fp, "%d%d", &u, &v) != EOF){
        node[u].realDeg++;
        node[v].realDeg++;
        edge.push_back(Edge(u,v));
    }

    fclose(fp);
}

void reorderByDegree(vector< Node > &node, vector< Edge > &edge){
    int nodeNum = (int)node.size();
    int edgeNum = (int)edge.size();
    vector< vector< int > > degList(nodeNum);
//    oriOrder.resize(nodeNum);

    for(int i = 0; i < nodeNum; i++){
        degList[node[i].realDeg].push_back(i);
    }
    for(int i = 0, deg = 0; deg < nodeNum; deg++){
        for(int j = 0; j < (int)degList[deg].size(); j++){
//            oriOrder[i] = degList[deg][j];
            node[degList[deg][j]].newOrder = i++;
        }
    }
    for(int i = 0; i < edgeNum; i++){
        edge[i].u = node[edge[i].u].newOrder;
        edge[i].v = node[edge[i].v].newOrder;
    }
}

void initDegList(vector< Node > &node, vector< Edge > &edge){
    int edgeNum = (int)edge.size();
    for(int i = 0; i < edgeNum; i++){
        int u = edge[i].u, v = edge[i].v;
        if(u < v) node[u].addNei(v);
        else node[v].addNei(u);
    }
    int nodeNum = (int)node.size();
    for(int i = 0; i < nodeNum; i++){
        sort(node[i].largerDegNei.begin(), node[i].largerDegNei.end());
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
//            triList.push_back(Triangle(a,b,l1[i]));
            triNum++;
            i++, j++;
        }
    }
    return triNum;
}
