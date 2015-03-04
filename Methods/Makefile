CC = g++
CFLAG = -Wall -O2
GCC = nvcc
GFLAG = -arch=sm_20 --ptxas-options="-dlcm=cg"

all: forward edgeIterator bitwise g_forward g_edgeIterator g_bitwise

forward: Forward/forward.cpp
	$(CC) Forward/forward.cpp $(CFLAG) -o forward

g_forward: Forward/forward.cu
	$(GCC) Forward/forward.cu $(GFLAG) -Xcompiler $(CFLAG) -o g_forward

edgeIterator: EdgeIterator/edgeIterator.cpp
	$(CC) EdgeIterator/edgeIterator.cpp $(CFLAG) -o edgeIterator

g_edgeIterator: EdgeIterator/edgeIterator.cu
	$(GCC) EdgeIterator/edgeIterator.cu $(GFLAG) -Xcompiler $(CFLAG) -o g_edgeIterator

bitwise: Bitwise/bitwise.cpp
	$(CC) Bitwise/bitwise.cpp $(CFLAG) -o bitwise

g_bitwise: Bitwise/bitwise.cu
	$(GCC) Bitwise/bitwise.cpp $(GFLAG) -Xcompiler $(CFLAG) -o g_bitwise

clean:
	rm -rf forward edgeIterator bitwise g_forward g_edgeIterator g_bruteForce g_bitwise
