#pragma once
#include <curand_kernel.h>
#include <cuda.h>
#include <cfloat>

#define cord(x, y) ((x) * n + (y))
#define MAX_N (1031)

__device__ float dist_dev[MAX_N * MAX_N];
__device__ float coef_dist_dev[MAX_N * MAX_N];
__device__ float prob_numerator[MAX_N * MAX_N];

__global__ void init_rng(curandState* states, unsigned long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    // Seed is the random seed, idx is like "subseed"
    // Each thread is thus initialized with different value and has its own currandState variable.
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void calculate_prob_numerator(float *pheromones, int n, float alpha) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)
        return;

    for (int i = idx; i < n * n; i += n) {
        prob_numerator[i] = fmaxf(__powf(pheromones[i], alpha) *
                                         coef_dist_dev[i], FLT_MIN);
    }
}
