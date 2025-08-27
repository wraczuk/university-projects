NVCC := nvcc
NVCCFLAGS := -arch=sm_60 -O3 -g -G

TARGET = acostsp
SRC = main.cu
DEPS = common.cuh input.h queen.cuh worker.cuh

all: $(TARGET)

$(TARGET): $(SRC) $(DEPS)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
