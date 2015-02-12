CC = g++
CFLAG = -Wall -O2
GCC = nvcc
GFLAG = -arch=sm_20 --ptxas-options="-dlcm=cg"

all: forward edgeIterator bruteForce g_forward g_edgeIterator

forward: Forward/forward.cpp
	$(CC) Forward/forward.cpp $(CFLAG) -o forward

g_forward: Forward/forward.cu
	$(GCC) Forward/forward.cu $(GFLAG) -Xcompiler $(CFLAG) -o g_forward

edgeIterator: EdgeIterator/edgeIterator.cpp
	$(CC) EdgeIterator/edgeIterator.cpp $(CFLAG) -o edgeIterator

g_edgeIterator: EdgeIterator/edgeIterator.cu
	$(GCC) EdgeIterator/edgeIterator.cu $(GFLAG) -Xcompiler $(CFLAG) -o g_edgeIterator

bruteForce: BruteForce/bruteForce.cpp
	$(CC) BruteForce/bruteForce.cpp $(CFLAG) -o bruteForce

clean:
	rm -rf forward edgeIterator bruteForce g_forward g_edgeIterator
