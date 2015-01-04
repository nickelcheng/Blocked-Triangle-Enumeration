#include<cstdio>
#include<vector>

using namespace std;

int main(int argc, char *argv[]){
    if(argc != 3){
        printf("usage: analysisDValue <src_file> <result_file>\n");
        return 0;
    }
    
    vector< int > groupSize;
    char srcFile[100], resultFile[100];
    sprintf(srcFile, "%s", argv[1]);
    sprintf(resultFile, "%s", argv[2]);
    groupSize.resize(100, 0);

    FILE *fp = fopen(srcFile, "r");
    int dvalue;
    while(fscanf(fp, "%d", &dvalue) != EOF){
        if(dvalue >= (int)groupSize.size())
            groupSize.resize(dvalue+1, 0);
        groupSize[dvalue]++;
    }
    fclose(fp);

    fp = fopen(resultFile, "w");
    int maxD = (int)groupSize.size() - 1;
    fprintf(fp, "degeneracy value: %d\n", maxD);
    fprintf(fp, "dvalue\tcount\n");
    for(int i = 1; i <= maxD; i++){
        fprintf(fp, "%5d\t%5d\n", i, groupSize[i]);
    }
    fclose(fp);

    return 0;
}
