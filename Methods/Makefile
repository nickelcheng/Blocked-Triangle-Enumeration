GCC = g++
FLAG = -Wall -O2
NVCC = nvcc
GPU_FLAG = -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35
RDC = -rdc=true

VPATH = $(wildcard */src/)

src = $(wildcard */src/*)

OBJ_DIR = object/
cudaObj = $(patsubst %.cu,$(OBJ_DIR)%.o,$(notdir $(filter %.cu,$(src))))
cppObj = $(patsubst %.cpp,$(OBJ_DIR)%.o,$(notdir $(filter %.cpp,$(src))))

listObj = io.o reorder.o forward.o edgeIterator.o vertexCover.o triangle.o
matObj = matFunc.o

CUDA_INC = /opt/cuda/include/
LIST_HEADER = ListIntersect/header/
MAT_HEADER = MatrixMul/header/
UTIL_HEADER = Util/header/
INCLUDE = -I$(LIST_HEADER) -I$(MAT_HEADER) -I$(UTIL_HEADER) -I$(CUDA_INC)


all: cpu gpu
cpu: list mat
gpu: g_list g_mat

list: $(addprefix $(OBJ_DIR),list.o $(listObj) tools.o)
	$(NVCC) $^ -Xcompiler $(FLAG) -o $@

mat: $(addprefix $(OBJ_DIR),mat.o $(matObj) tools.o)
	$(NVCC) $^ -Xcompiler $(FLAG) -o $@

g_list:$(addprefix $(OBJ_DIR),g_list.o $(listObj) tools.o)
	$(NVCC) $^ $(GPU_FLAG) -Xcompiler $(FLAG) -o $@

g_mat: $(addprefix $(OBJ_DIR),g_mat.o $(matObj) tools.o)
	$(NVCC) $^ $(GPU_FLAG) -Xcompiler $(FLAG) -o $@

$(cudaObj): $(OBJ_DIR)%.o: %.cu
	$(NVCC) -c $< $(GPU_FLAG) $(RDC) -Xcompiler $(FLAG) $(INCLUDE) -o $@

$(cppObj): $(OBJ_DIR)%.o: %.cpp
	$(GCC) -c $< $(FLAG) $(INCLUDE) -o $@

clean:
	rm -rf $(OBJ_DIR)*
	rm -rf list mat g_list g_mat
