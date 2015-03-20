#include<cstdio>
#include<cstdlib>
#include<vector>
#include<algorithm>
#include<sys/time.h>

#define swap(a,b) {int tmp = a; a = b, b = tmp;}

#define cntTime(st,ed)\
((double)ed.tv_sec*1000000+ed.tv_usec-(st.tv_sec*1000000+st.tv_usec))/1000

#define timerInit(n)\
struct timeval st[n], ed[n];

#define timerStart(n)\
gettimeofday(&st[n], NULL);

#define timerEnd(tar, n)\
gettimeofday(&ed[n], NULL);\
fprintf(stderr, "%s: %.3lf ms\n", tar, cntTime(st[n],ed[n]));
//fprintf(stderr, " %.3lf", cntTime(st[n],ed[n]));


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

void input(const char *inFile, vector< Node > &node, vector< Edge > &edge);
void reorderByDegree(vector< Node > &node, vector< Edge > &edge);
void updateGraph(vector< Node > &node, vector< Edge > &edge);
__global__ void countTriNum(int *offset, int *edgeV, int *triNum, int nodeNum);
__device__ int intersectList(int sz1, int sz2, int *l1, int *l2);

extern __shared__ int shared[]; // adj[maxDeg], threadTriNum[threadNum]

int main(int argc, char *argv[]){
    if(argc != 5){
        fprintf(stderr, "usage: forward <input_path> <node_num> <thread_per_block> <block_num>\n");
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
    reorderByDegree(node, edge);
    updateGraph(node, edge);
    timerEnd("reordering", 1)

    int maxDeg = 0;
    for(int i = 0; i < nodeNum; i++){
        if(node[i].degree() > maxDeg)
            maxDeg = node[i].degree();
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
            h_edgeV[idx] = node[i].largerDegNei[j];
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
    int smSize = (threadNum+maxDeg) * sizeof(int);
/*    printf("%d blocks, %d threads per block\n", nodeNum, threadNum);
    printf("smSize = %d bytes\n", smSize);*/
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

void reorderByDegree(vector< Node > &node, vector< Edge > &edge){
    int nodeNum = (int)node.size();
    int edgeNum = (int)edge.size();
    vector< vector< int > > degList(nodeNum);

    // count degree for each node
    for(int i = 0; i < edgeNum; i++){
        node[edge[i].u].realDeg++;
        node[edge[i].v].realDeg++;
    }
    // reorder by counting sort
    for(int i = 0; i < nodeNum; i++){
        degList[node[i].realDeg].push_back(i);
    }
    for(int i = 0, deg = 0; deg < nodeNum; deg++){
        for(int j = 0; j < (int)degList[deg].size(); j++){
            node[degList[deg][j]].newOrder = i++;
        }
    }
}

void updateGraph(vector< Node > &node, vector< Edge > &edge){
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
        sort(node[i].largerDegNei.begin(), node[i].largerDegNei.end());
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
                shared[idx] = edgeV[myOffset+idx]; // adj[idx]
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
