#ifndef __SOLVE_H__
#define __SOLVE_H__

#include "block.h"
#include "listArray.h"
#include <vector>

using std::vector;

const int GPU_BLOCK_NUM = 16000;
const int GPU_THREAD_NUM = 256;
enum{LIST = 0, G_LIST, MAT, G_MAT, UNDEF};
enum{CPU = 0, GPU};

long long findTriangle(const ListArrMatrix &block, const vector< int > &rowWidth, int blockDim);
void scheduler(const ListArray &edge, const ListArray &target, int entry, bool delTar);
void getStrategy(const ListArray &edge, const ListArray &target, int &device, int &proc);

#endif
