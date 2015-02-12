#include<cstdio>
#include<cstdlib>
#include<vector>
#include<algorithm>
#include<sys/time.h>

#define swap(a,b) {int tmp=a;a=b,b=tmp;}

#define cntTime(st,ed)\
((double)ed.tv_sec*1000000+ed.tv_usec-(st.tv_sec*1000000+st.tv_usec))/1000

#define timerInit()\
struct timeval st, ed;

#define timerStart()\
gettimeofday(&st, NULL);

#define timerEnd(tar)\
gettimeofday(&ed, NULL);\
fprintf(stderr, " %.3lf", cntTime(st,ed));
//fprintf(stderr, "%s: %.3lf ms\n", tar, cntTime(st,ed));


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
    Edge(int _u, int _v){
        u = _u, v = _v;
    }
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

/*vector< Triangle > triList;
vector< int > oriOrder;*/

void input(const char *inFile, vector< Node > &node, vector< Edge > &edge);

void reorderByDegeneracy(vector< Node > &node, vector< Edge > &edge);
void buildDegList(vector< Node > &node, vector< DegList > &degList);
void reordering(vector< Node > &node, vector< DegList > &degList);
int findMinDegNode(int &currPos, vector< DegList > &degList);
void removeNode(int v, vector< Node > &node, vector< DegList > &degList);
void decDegByOne(int v, vector< Node > &node, vector< DegList > &degList);
void removeNodeInList(int v, vector< Node > &node, vector< DegList > &degList);

void updateGraph(vector< Edge > &edge, vector< Node > &node);
__global__ void countTriNum(int *offset, int *edgeU, int *edgeV, int *triNum, int edgeNum);
__device__ int intersectList(int sz1, int sz2, int *l1, int *l2);

int main(int argc, char *argv[]){
    if(argc != 3){
        fprintf(stderr, "usage: listIntersect <input_path> <node_num>\n");
        return 0;
    }

    timerInit()
    timerStart()

    int nodeNum = atoi(argv[2]);
    vector< Node > node(nodeNum);
    vector< Edge > edge;

    input(argv[1], node, edge);
    reorderByDegeneracy(node, edge);
    updateGraph(edge, node);

    timerEnd("preprocessing")
    timerStart()
    
    int edgeNum = (int)edge.size();
    int triNum = 0, *h_offset, *h_edgeU, *h_edgeV;
    int *d_triNum, *d_offset, *d_edgeU, *d_edgeV;
    
    h_offset = (int*)malloc(sizeof(int)*(nodeNum+1));
    h_edgeU = (int*)malloc(sizeof(int)*edgeNum);
    h_edgeV = (int*)malloc(sizeof(int)*edgeNum);

    h_offset[0] = 0;
    for(int i = 0; i < nodeNum; i++){
        int deg = node[i].degree();
        h_offset[i+1] = h_offset[i] + deg;
        for(int j = 0; j < deg; j++){
            int idx = h_offset[i] + j;
            h_edgeU[idx] = i;
            h_edgeV[idx] = node[i].nei[j];
        }
    }

    timerEnd("cpu copy")
    timerStart()

    cudaMalloc((void**)&d_triNum, sizeof(int));
    cudaMalloc((void**)&d_offset, sizeof(int)*(nodeNum+1));
    cudaMalloc((void**)&d_edgeU, sizeof(int)*edgeNum);
    cudaMalloc((void**)&d_edgeV, sizeof(int)*edgeNum);

    cudaMemcpy(d_triNum, &triNum, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offset, h_offset, sizeof(int)*(nodeNum+1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeU, h_edgeU, sizeof(int)*edgeNum, cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeV, h_edgeV, sizeof(int)*edgeNum, cudaMemcpyHostToDevice);

    timerEnd("cuda copy")
    timerStart()

    int nB = (int)ceil((edgeNum/1024.0)+0.001);
    countTriNum<<< nB, 1024 >>>(d_offset, d_edgeU, d_edgeV, d_triNum, edgeNum);
    cudaDeviceSynchronize();
    timerEnd("intersection")

    cudaMemcpy(&triNum, d_triNum, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_triNum);
    cudaFree(d_offset);
    cudaFree(d_edgeU);
    cudaFree(d_edgeV);

    free(h_offset);
    free(h_edgeU);
    free(h_edgeV);

    printf("total triangle: %d\n", triNum);

    return 0;
}

void input(const char *inFile, vector< Node > &node, vector< Edge > &edge){
    timerInit()
    timerStart()

    FILE *fp = fopen(inFile, "r");
    int u, v;
    while(fscanf(fp, "%d%d", &u, &v) != EOF){
        node[u].addNei(v);
        node[v].addNei(u);
        edge.push_back(Edge(u,v));
    }

    fclose(fp);
    timerEnd("input")
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
//    oriOrder.resize(nodeNum);

    for(int i = 0; i < nodeNum; i++){
        node[i].leftDeg = node[i].degree();
    }

    for(int i = 0; i < nodeNum; i++){
        int v = findMinDegNode(currPos, degList);
        removeNode(v, node, degList);
        node[v].newOrder = i;
//        oriOrder[i] = v;
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

__global__ void countTriNum(int *offset, int *edgeU, int *edgeV, int *triNum, int edgeNum){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < edgeNum){
        int u = edgeU[idx];
        int v = edgeV[idx];
        int szu = offset[u+1] - offset[u];
        int szv = offset[v+1] - offset[v];
        int tmp;
        tmp = intersectList(szu, szv, &edgeV[offset[u]], &edgeV[offset[v]]);
        atomicAdd(triNum, tmp);
    }
}

__device__ int intersectList(int sz1, int sz2, int *l1, int *l2){
    int i, j;
    int triNum = 0;
    for(i = sz1-1, j = sz2-1; i >= 0 && j >= 0;){
        if(l1[i] > l2[j]) i--;
        else if(l1[i] < l2[j]) j--;
        else{
            i--, j--;
            triNum++;
        }
    }
    return triNum;
}
