#include<cstdio>
#include<vector>

using namespace std;

typedef struct Node Node;
typedef struct Edge Edge;

struct Node{
    vector< int > largerDegNei;
    int realDeg;
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
};


void input(const char *inFile, vector< Node > &node, vector< Edge > &edge);
void reorderByDegree(vector< Node > &node, vector< Edge > &edge);
void initDegList(vector< Node > &node, vector< Edge > &edge);
int intersectList(vector< int > &l1, vector< int > &l2);

int main(int argc, char *argv[]){
    if(argc != 2){
        fprintf(stderr, "usage: forward <input_path>\n");
        return 0;
    }

    vector< Node > node;
    vector< Edge > edge;

    input(argv[1], node, edge);
    reorderByDegree(node, edge);
    initDegList(node, edge);

    int nodeNum = (int)node.size();
    int triNum = 0;
    for(int i = 0; i < nodeNum; i++){
        int deg = node[i].degree();
        for(int j = 0; j < deg; j++){
            int tar = node[i].largerDegNei[j];
            triNum += intersectList(node[i].largerDegNei, node[tar].largerDegNei);
        }
    }
    fprintf(stderr, "total triangle: %d\n", triNum);

    return 0;
}

void input(const char *inFile, vector< Node > &node, vector< Edge > &edge){
    int nodeNum, edgeNum;
    FILE *fp = fopen(inFile, "r");

    fscanf(fp, "%d%d", &nodeNum, &edgeNum);
    node.resize(nodeNum);
    edge.resize(edgeNum);
    for(int i = 0; i < edgeNum; i++){
        fscanf(fp, "%d%d", &edge[i].u, &edge[i].v);
        node[edge[i].u].realDeg++;
        node[edge[i].v].realDeg++;
    }

    fclose(fp);
}

void reorderByDegree(vector< Node > &node, vector< Edge > &edge){
    int nodeNum = (int)node.size();
    int edgeNum = (int)edge.size();
    vector< vector< int > > degList(nodeNum);
    int newOrd[nodeNum];

    for(int i = 0; i < nodeNum; i++){
        degList[node[i].realDeg].push_back(i);
    }
    for(int i = 0, deg = 0; deg < nodeNum; deg++){
        for(int j = 0; j < (int)degList[deg].size(); j++){
            newOrd[degList[deg][j]] = i++;
        }
    }
    for(int i = 0; i < edgeNum; i++){
        edge[i].u = newOrd[edge[i].u];
        edge[i].v = newOrd[edge[i].v];
    }
}

void initDegList(vector< Node > &node, vector< Edge > &edge){
    int edgeNum = (int)edge.size();
    for(int i = 0; i < edgeNum; i++){
        int u = edge[i].u, v = edge[i].v;
        if(u < v) node[u].addNei(v);
        else node[v].addNei(u);
    }
}

int intersectList(vector< int > &l1, vector< int > &l2){
    int sz1 = (int)l1.size();
    int sz2 = (int)l2.size();
    int triNum = 0;
    for(int i = 0, j = 0; i < sz1 && j < sz2;){
        if(l1[i] < l2[j]) i++;
        else if(l1[i] > l2[j]) j++;
        else{
            triNum++;
            i++, j++;
        }
    }
    return triNum;
}
