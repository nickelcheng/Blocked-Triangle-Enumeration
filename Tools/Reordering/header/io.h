#ifndef __MY_IO_H__
#define __MY_IO_H__

#include "main.h"

int input(
    const char *srcFile,
    vector< Edge > &edge
);

void output(
    const char *tarFile, const char *nodeMapFile, 
    vector< Node > &node, vector< Edge > &edge
);

void outputTar(const char *file, int nodeNum, vector< Edge > &edge);
void outputNodeMap(const char *file, vector< Node > &node);

#endif
