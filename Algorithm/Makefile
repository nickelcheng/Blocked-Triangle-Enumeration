GCC = g++
FLAG = -Wall -O2 -lpthread -fopenmp
NVCC = nvcc
GPU_FLAG = -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52
RDC = -rdc=true

VPATH = src

src = $(notdir $(wildcard src/*))

OBJ_DIR = object/
cudaObj = $(patsubst %.cu,$(OBJ_DIR)g_%.o, $(filter %.cu,$(src)))
cppObj = $(patsubst %.cpp,$(OBJ_DIR)%.o, $(filter %.cpp,$(src)))
obj = $(cudaObj) $(cppObj)
procObj = $(OBJ_DIR)procedure.o

HEADER_DIR = header/
THRUST_INC = /home/shadow9
INCLUDE = -I$(HEADER_DIR)


all: count proc

cardInfo: ../Tools/gpuCardInfo.cu
	$(NVCC) $^ -o $@

count: $(filter-out $(OBJ_DIR)procedure.o, $(obj))
	$(NVCC) $^ -Xcompiler "$(FLAG)" -o $@

proc: $(filter-out $(OBJ_DIR)main.o, $(obj))
	$(NVCC) $^ -Xcompiler "$(FLAG)" -o $@

$(cudaObj): $(OBJ_DIR)g_%.o: %.cu
	$(NVCC) -c $< $(GPU_FLAG) $(RDC) -Xcompiler "$(FLAG)" $(INCLUDE) -o $@

$(cppObj): $(OBJ_DIR)%.o: %.cpp
	$(GCC) -c $< $(FLAG) $(INCLUDE) -o $@

clean:
	rm -rf $(OBJ_DIR)*
	rm -rf count proc
	rm -rf cardInfo
