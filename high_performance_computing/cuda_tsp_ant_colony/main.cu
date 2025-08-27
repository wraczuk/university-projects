#include <iostream>
#include <vector>
#include <string>
#include <stdio.h>

// #define CUDAGRAPH
// #define VERBOSE

std::vector<double>x, y;
#include "input.h"
#include "worker.cuh"
#include "queen.cuh"


int main(int argc, char* argv[]) {
    if (argc != 9) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file> <TYPE> <NUM_ITER>"
                                          << " <ALPHA> <BETA> <EVAPORATE> <SEED>\n";
        return 1;
    }
    char* input_file = argv[1];
    char* output_file = argv[2];
    std::string TYPE = argv[3];
    int num_iter = std::stoi(argv[4]);
    double alpha = std::stod(argv[5]);
    double beta = std::stod(argv[6]);
    double evaporate = std::stod(argv[7]);
    unsigned long seed = std::stoul(argv[8]);

    auto tr1 = freopen(input_file, "r", stdin);
    auto tr2 = freopen(output_file, "w", stdout);

    std::string name, comment, type, edge_weight_fromat;
    int n;

    auto [xx, yy] = read_input(name, comment, type, n, edge_weight_fromat);
    x = xx;
    y = yy;

    std::vector<float> dist(n * n);
    populate_dist(dist, x, y);

    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaDeviceSynchronize();
    // cudaEventRecord(start, 0);

    if (TYPE == "WORKER")
        worker(dist.data(), n, alpha, beta, evaporate, seed, num_iter);
    else
        queen(dist.data(), n, alpha, beta, evaporate, seed, num_iter);

    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // float elapsedTime;
    // cudaEventElapsedTime(&elapsedTime, start, stop);
    // std::cout << "Time to generate: " << elapsedTime << "\n";
    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);
}