CXX := g++
CXXFLAGS := -Wall -Wextra -std=c++23 -O2

SRCS := $(wildcard *.cpp)
EXES := $(SRCS:.cpp=)

.PHONY: all clean

all: $(EXES)

%.o: %.cpp funclist.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(EXES)