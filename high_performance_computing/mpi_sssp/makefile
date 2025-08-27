CXX := mpic++
CXXFLAGS := -std=c++20 -O3 -DREMOVE_REPETITIONS -DPRUNING -DHYBRIDIZATION -fopenmp

EXEC := sssp
SRCS := main.cpp
OBJS := $(SRCS:.cpp=.o)

HEADERS := common.h global.h edge_classification.h hybridization.h naive.h pruning.h solution.h

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(EXEC) $(OBJS)

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(EXEC)