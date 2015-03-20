#include<cstdio>
#include<cstdlib>
#include<vector>
#include<algorithm>
#include<sys/time.h>

#define swap(a,b) {int tmp=a;a=b,b=tmp;}

#define cntTime(st,ed)\
((double)ed.tv_sec*1000000+ed.tv_usec-(st.tv_sec*1000000+st.tv_usec))/1000

#define timerInit(n)\
struct timeval st[n], ed[n];

#define timerStart(n)\
gettimeofday(&st[n], NULL);

#define timerEnd(tar, n)\
gettimeofday(&ed[n], NULL);\
fprintf(stderr, " %.3lf", cntTime(st[n],ed[n]));
//fprintf(stderr, "%s: %.3lf ms\n", tar, cntTime(st[n],ed[n]));


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
__global__ void countTriNum(int *offset, int *edgeV, int *triNum, int nodeNum);
__device__ int intersectList(int sz1, int sz2, int *l1, int *l2);

extern __shared__ int shared[]; // adj[maxDeg], threadTriNum[threadNum]

int main(int argc, char *argv[]){
    if(argc != 5){
        fprintf(stderr, "usage: listIntersect <input_path> <node_num> <thread_per_block> <block_num>\n");
        return 0;
    }

    timerInit(2)
    timerStart(0)

    int nodeNum = atoi(argv[2]);
    vector< Node > node(nodeNum);
    vector< Edge > edge;

    timerStart(1)
    input(argv[1], node, edge);
    timerEnd("input", 1)

    timerStart(1)
    reorderByDegeneracy(node, edge);
    updateGraph(edge, node);
    timerEnd("reordering", 1)

    int degeneracy = 0;
    for(int i = 0; i < nodeNum; i++){
        if(node[i].degree() > degeneracy)
            degeneracy = node[i].degree();
    }

    int edgeNum = (int)edge.size();
    int triNum = 0, *h_offset, *h_edgeV;
    int *d_triNum, *d_offset, *d_edgeV;
    
    h_offset = (int*)malloc(sizeof(int)*(nodeNum+1));
    h_edgeV = (int*)malloc(sizeof(int)*edgeNum);

    h_offset[0] = 0;
    for(int i = 0; i < nodeNum; i++){
        int deg = node[i].degree();
        h_offset[i+1] = h_offset[i] + deg;
        for(int j = 0; j < deg; j++){
            int idx = h_offset[i] + j;
            h_edgeV[idx] = node[i].nei[j];
        }
    }

    timerStart(1)
    cudaMalloc((void**)&d_triNum, sizeof(int));
    cudaMalloc((void**)&d_offset, sizeof(int)*(nodeNum+1));
    cudaMalloc((void**)&d_edgeV, sizeof(int)*edgeNum);

    cudaMemcpy(d_triNum, &triNum, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offset, h_offset, sizeof(int)*(nodeNum+1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeV, h_edgeV, sizeof(int)*edgeNum, cudaMemcpyHostToDevice);
    timerEnd("cuda copy", 1)

    timerStart(1)
    int threadNum = atoi(argv[3]);
    int blockNum = atoi(argv[4]);
    int smSize = (threadNum+degeneracy) * sizeof(int);
    countTriNum<<< blockNum, threadNum, smSize >>>(d_offset, d_edgeV, d_triNum, nodeNum);
    cudaDeviceSynchronize();
    timerEnd("intersection", 1)

    cudaMemcpy(&triNum, d_triNum, sizeof(int), cudaMemcpyDeviceToHost);
    printf("total triangle: %d\n", triNum);

    cudaFree(d_triNum);
    cudaFree(d_offset);
    cudaFree(d_edgeV);

    free(h_offset);
    free(h_edgeV);

    timerEnd("total", 0)

    return 0;
}

void input(const char *inFile, vector< Node > &node, vector< Edge > &edge){
    FILE *fp = fopen(inFile, "r");
    int u, v;
    while(fscanf(fp, "%d%d", &u, &v) != EOF){
        edge.push_back(Edge(u,v));
    }
    fclose(fp);
}

void reorderByDegeneracy(vector< Node > &node, vector< Edge > &edge){
    int nodeNum = (int)node.size();
    int edgeNum = (int)edge.size();
    vector< DegList > degList;

    // count degree for each node
    for(int i = 0; i < edgeNum; i++){
        node[edge[i].u].addNei(edge[i].v);
        node[edge[i].v].addNei(edge[i].u);
    }
    // reordering
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

__global__ void countTriNum(int *offset, int *edgeV, int *triNum, int nodeNum){
    int nodePerBlock = (int)ceil((double)nodeNum/gridDim.x-0.001);
    for(int r = 0; r < nodePerBlock; r++){
        int nodeID = blockIdx.x*nodePerBlock + r;
        if(nodeID >= nodeNum) continue;
        int myOffset = offset[nodeID];
        int nextOffset = offset[nodeID+1];
        int deg = nextOffset - myOffset;
        int jobPerThread = (int)ceil((double)deg/blockDim.x-0.001);

        // move node u's adj list to shared memory
        for(int i = 0; i < jobPerThread; i++){
            int idx = threadIdx.x*jobPerThread + i;
            if(idx < deg){
                shared[idx] = edgeV[myOffset+idx];
            }
        }
        __syncthreads();

        // counting triangle number
        shared[deg+threadIdx.x] = 0;
        for(int i = 0; i < jobPerThread; i++){
            int idx = threadIdx.x*jobPerThread + i;
            if(idx < deg){
                int v = shared[idx]; // adj[idx]
                int vNeiLen = offset[v+1] - offset[v];
                shared[deg+threadIdx.x] += intersectList(deg, vNeiLen, shared, &edgeV[offset[v]]); // threadTriNum[threadIdx.x]
            }
        }
        __syncthreads();

        // sum triangle number
        if(threadIdx.x == 0){
            int tmp = 0;
            for(int i = 0; i < blockDim.x; i++){
                tmp += shared[deg+i]; // threadTriNum[i]
            }
            atomicAdd(triNum, tmp);
        }
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
