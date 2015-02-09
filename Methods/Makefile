CC = g++
FLAG = -Wall -O2

all: forward edgeIterator bruteForce

forward:
	$(CC) Forward/forward.cpp $(FLAG) -o forward

edgeIterator:
	$(CC) EdgeIterator/edgeIterator.cpp $(FLAG) -o edgeIterator

bruteForce:
	$(CC) BruteForce/bruteForce.cpp $(FLAG) -o bruteForce

clean:
	rm -rf forward edgeIterator bruteForce
