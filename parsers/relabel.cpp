#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<map>
#include<vector>
#include<algorithm>

#define mp(a,b) make_pair(a,b)
#define swap(a,b) {int tmp; tmp=a,a=b,b=tmp;}

using namespace std;

map< int, int > nodemap;
vector< pair< int, int > > edge;

int relabel(int v);
bool parsing(const char *srcFile, const char *tarFile, const int format);
void parser(const char *srcFile, const int format);
void ignoreComment(FILE *fp);
void output(const char *nodeMapFile, const char *tarFile);

int main(int argc, char *argv[]){
	if(argc != 5){
        fprintf(stderr, "usage: relabel <src_file> <tar_file> <node_map_file> <format>\n");
        return 0;
	}

    char srcFile[100], tarFile[100], nodeMapFile[100];
    int format;
    sprintf(srcFile, "%s", argv[1]);
    sprintf(tarFile, "%s", argv[2]);
    sprintf(nodeMapFile, "%s", argv[3]);
    format = atoi(argv[4]);

    edge.clear();
    nodemap.clear();

    if(parsing(srcFile, tarFile, format)){
        sort(edge.begin(), edge.end());
        edge.erase(unique(edge.begin(), edge.end()), edge.end());
        output(nodeMapFile, tarFile);
    }

	return 0;
}

bool parsing(const char *srcFile, const char *tarFile, const int format){
    if(format == 1 || format == 2)
        parser(srcFile, format);
    else{
        fprintf(stderr, "parser format error\n");
        return false;
    }
    return true;
}

/* 
 * parser1 format
 * several lines of comments (start with '#')
 * u <blank(s)> v <\n>
 *
 * parser2 format
 * several lines of comments (start with '#')
 * u <blank(s)> v <blank(s)> direction <\n>
 */ 
void parser(const char *srcFile, const int format){
    FILE *fp = fopen(srcFile, "r");

    ignoreComment(fp);

    int u, v, a, b;
    while(fscanf(fp, "%d%d", &u, &v) != EOF){
        if(format == 2) fscanf(fp, "%*d");

        if(u == v) continue;
        a = relabel(u);
        b = relabel(v);
        if(a > b) swap(a, b)
        edge.push_back(mp(a, b));
    }

    fclose(fp);
}

void ignoreComment(FILE *fp){
    char line[100010];
    int comment = 0;
    
    /* find number of comment lines */
    while(1){
        fgets(line, 100000, fp);
        if(line[0] == '#') comment++;
        else break;
    }

    /* ignore them */
    rewind(fp);
    for(int i = 0; i < comment; i++){
        fgets(line, 100000, fp);
    }
}

int relabel(int v){
    pair< map< int,int >::iterator, bool > result;
    int nextID = (int)nodemap.size()+1;
    result = nodemap.insert(mp(v, nextID));
    return result.first->second;
}

void output(const char *nodeMapFile, const char *tarFile){
    FILE *fp = fopen(tarFile, "w");
    fprintf(fp, "%d %d\n", (int)nodemap.size(), (int)edge.size());

    for(int i = 0; i < (int)edge.size(); i++){
        fprintf(fp, "%d %d\n", edge[i].first, edge[i].second);
    }
    fclose(fp);

    fp = fopen(nodeMapFile, "w");
    map< int, int >::iterator it;
    for(it = nodemap.begin(); it != nodemap.end(); ++it){
        fprintf(fp, "%d -> %d\n", it->first, it->second);
    }
    fclose(fp);
}
