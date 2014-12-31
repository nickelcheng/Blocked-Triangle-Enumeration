#include<cstdio>
#include<set>
#include<vector>
#include<queue>
#include<algorithm>
#include<functional>

#define UNDEF -1

using namespace std;

typedef struct Node{
    set< int > nei;
    int newOrder;
    Node(void){
        nei.clear();
        newOrder = UNDEF;
    }
    int removeNei(int v){
        return nei.erase(v);
    }
} Node;

typedef struct Edge{
    int u, v;
    Edge(void){
        u = v = -1;
    }
    Edge(int _u, int _v){
        u = _u, v = _v;
    }
    bool operator < (const Edge &a) const{
        if(u != a.u) return u < a.u;
        return v < a.v;
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

vector< Node > node;
vector< Edge > edge;
priority_queue< DegInfo, vector< DegInfo >, greater< DegInfo > > degInfoPQ;
int degeneracy;

void input(const char *srcFile);
void initDegInfo(void);
int findDegMinNode(void);
void removeNode(int v);
void renewDegInfoPQ(int v);
void sortNewEdge(void);
void output(const char *tarFile, const char *nodeMapFile);

int main(int argc, char *argv[]){
    if(argc != 4){
        fprintf(stderr, "usage: degeneracy <src_file> <tar_file> <node_map_file>\n");
        return 0;
    }

    char srcFile[100], tarFile[100], nodeMapFile[100];
    sprintf(srcFile, "%s", argv[1]);
    sprintf(tarFile, "%s", argv[2]);
    sprintf(nodeMapFile, "%s", argv[3]);

    input(srcFile);
    printf("input done\n");

    int vertices = (int)node.size();
    int next = 1;
    degeneracy = 0;
    initDegInfo();
    printf("init done\n");
    while(vertices--){
        int degMinNode = findDegMinNode();
        node[degMinNode].newOrder = next++;

        if(degeneracy < node[degMinNode].nei.size()){
            degeneracy = (int)node[degMinNode].nei.size();
        }

        removeNode(degMinNode);
    }
    printf("degeneracy = %d\n", degeneracy);

    sortNewEdge();
    output(tarFile, nodeMapFile);

    return 0;
}

void input(const char *srcFile){
    freopen(srcFile, "r", stdin);

    int nodeNum, edgeNum;
    scanf("%d%d", &nodeNum, &edgeNum);
    node.resize(nodeNum);
    edge.resize(edgeNum);

    for(int i = 0; i < edgeNum; i++){
        int u, v;
        scanf("%d%d", &u, &v);
        u--, v--;
        edge[i] = Edge(u, v);
        node[u].nei.insert(v);
        node[v].nei.insert(u);
    }
}

void initDegInfo(void){
    while(!degInfoPQ.empty()) degInfoPQ.pop();
    for(int i = 0; i < (int)node.size(); i++){
        int deg = (int)node[i].nei.size();
        degInfoPQ.push(DegInfo(deg, i));
    }
}

int findDegMinNode(void){
/*    int mmin = (int)node.size() + 1;
    int ch = -1;
    for(int i = 0; i < (int)node.size(); i++){
        if(node[i].newOrder != UNDEF) continue;
        int deg = (int)node[i].nei.size();
        if(deg < mmin){
            mmin = deg;
            ch = i;
        }
    }
    return ch;*/
    while(!degInfoPQ.empty() && node[degInfoPQ.top().nodeID].newOrder != UNDEF)
        degInfoPQ.pop();
    int mmin = degInfoPQ.top().nodeID;
    degInfoPQ.pop();
    return mmin;
}

void removeNode(int v){
    set< int >::iterator it;
    for(it = node[v].nei.begin(); it != node[v].nei.end(); ++it){
        node[*it].removeNei(v);
        renewDegInfoPQ(*it);
    }
}

void renewDegInfoPQ(int v){
    int deg = (int)node[v].nei.size();
    degInfoPQ.push(DegInfo(deg, v));
}

void sortNewEdge(void){
    for(int i = 0; i < (int)edge.size(); i++){
        edge[i].u = node[edge[i].u].newOrder;
        edge[i].v = node[edge[i].v].newOrder;
        if(edge[i].u > edge[i].v){
            int tmp = edge[i].u;
            edge[i].u = edge[i].v;
            edge[i].v = tmp;
        }
    }
    sort(edge.begin(), edge.end());
}

void output(const char *tarFile, const char *nodeMapFile){
    freopen(tarFile, "w", stdout);

    printf("%d %d\n", (int)node.size(), (int)edge.size());

    for(int i = 0; i < (int)edge.size(); i++){
        printf("%d %d\n", edge[i].u, edge[i].v);
    }

    freopen(nodeMapFile, "w", stdout);
    for(int i = 0; i < (int)node.size(); i++){
        printf("%d -> %d\n", i, node[i].newOrder);
    }
}

