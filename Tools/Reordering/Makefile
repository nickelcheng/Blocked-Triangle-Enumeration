COMPILER = g++
SRC_DIR = src
HEADER_DIR = header
FLAGS = -Wall -O2 -std=c++11
EXEC = reorder

all:
	$(COMPILER) -I$(HEADER_DIR) $(FLAGS) $(SRC_DIR)/* -o $(EXEC)

clean:
	rm -rf $(EXEC)

