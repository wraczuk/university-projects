#pragma once
#include <algorithm>
#include <cfloat>
#include <cub/cub.cuh>

#include "common.cuh"

#define QUEEN_THREAD_NUM        128
#define WARP_SIZE               32
#define q_tour_pos(i)           (n * block_id + (i))

__global__ void queen_kernel(int n, int *tour, curandState *states)
{
    int idx = threadIdx.x;
    int block_id = blockIdx.x;

    __shared__ int current_city; 
    __shared__ bool not_visited[MAX_N];
    __shared__ float prob[MAX_N];

    for (int i = idx; i < n; i += blockDim.x) {
        not_visited[i] = true;
    }
    __syncthreads();

    __shared__ float sh_sum_prob;
    if (idx == 0) {
        sh_sum_prob = 0.;
        current_city = (int)(curand_uniform(&states[block_id]) * n + 1) % n;
        tour[q_tour_pos(0)] = current_city;
        not_visited[current_city] = false;
    }
    __syncthreads();

    for (int step = 1; step < n; step++) {
        float sum_prob = 0.;
        for (int j = idx, cr = cord(current_city, idx); j < n; j += blockDim.x, cr += blockDim.x) {
            float tmp = prob_numerator[cr] * not_visited[j];
            prob[j] = tmp;
            sum_prob += tmp;
        }
        atomicAdd(&sh_sum_prob, sum_prob);
        __syncthreads();

        if (idx == 0) {
            sum_prob = sh_sum_prob;
            sh_sum_prob = 0.;
            float rand_val = curand_uniform(&states[block_id]) * sum_prob;
            float cumulative_prob = 0.;
            int j = -1;

            while (cumulative_prob < rand_val && j + 1 < n) {
                j++;
                cumulative_prob += prob[j];
            }
            
            if (cumulative_prob < rand_val) {
                while (!not_visited[j]) {
                    j--;
                }
            }

            current_city = j;
            tour[q_tour_pos(step)] = j;
            not_visited[j] = false;
        }
        __syncthreads();
    }
}

__global__ void queen_update_pheromones_and_lengths(float *pheromones, int n, int *tour, float *len, float evaporate) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int block_id = idx;
    if (idx >= n)
        return;

    len[idx] = 0.0;
    for (int i = idx; i < n * n; i += n) {
        pheromones[i] *= (1.0 - evaporate);
    }
    __syncthreads();
    
    // Update pheromones based on the tour
    for (int i = 0; i < n; i++) {
        int city_a = tour[q_tour_pos(i)];
        int city_b = tour[q_tour_pos((i + 1) % n)];
        len[idx] += dist_dev[cord(city_a, city_b)];
    }

    float inv_len = 1.0 / len[idx];
    for (int i = 0; i < n; i++) {
        int city_a = tour[q_tour_pos(i)];
        int city_b = tour[q_tour_pos((i + 1) % n)];
        atomicAdd(&pheromones[cord(city_a, city_b)], inv_len);
    }
    __syncthreads();
}

void queen(float *dist, int n, float alpha, float beta,
           float evaporate, unsigned long seed, int num_iter)
{
    curandState *states;
    cudaMalloc((void **)&states, sizeof(curandState) * n);
    init_rng<<<(n + WARP_SIZE - 1) / WARP_SIZE, WARP_SIZE>>>(states, seed, n);

    int *tour;
    float *len, *pheromones;
    cudaMalloc((void **)&tour, sizeof(int) * n * n);
    cudaMalloc((void **)&len, sizeof(float) * n);
    cudaMalloc((void **)&pheromones, sizeof(float) * n * n);
    
    cudaMemcpyToSymbol(dist_dev, dist, sizeof(float) * n * n);
    
    std::vector<float> coef_dist_filler(n * n);
    for (int i = 0; i < n * n; i++) {
        coef_dist_filler[i] = pow(1.0 / dist[i], beta);
    }
    cudaMemcpyToSymbol(coef_dist_dev, coef_dist_filler.data(), sizeof(float) * n * n);
 
    std::vector<float> pheromones_filler(n * n, 1.);
    cudaMemcpy(pheromones, pheromones_filler.data(), sizeof(float) * n * n, cudaMemcpyHostToDevice);

    calculate_prob_numerator<<<(n + WARP_SIZE - 1) / WARP_SIZE, WARP_SIZE>>>(pheromones, n, alpha);

    float min_len = 1e18;
    int *tour_h;
    float* len_h;
    cudaHostAlloc((void**)&len_h, sizeof(float) * n, cudaHostAllocDefault);
    cudaHostAlloc((void**)&tour_h, sizeof(int) * n * n, cudaHostAllocDefault);
    int id = -1;

    #ifdef CUDAGRAPH
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        queen_kernel<<<n, QUEEN_THREAD_NUM, 0, stream>>>(
            n, tour, states
        );

        queen_update_pheromones_and_lengths<<<(n + WARP_SIZE - 1) / WARP_SIZE, WARP_SIZE, 0, stream>>>(
                pheromones, n, tour, len, evaporate
        );

        calculate_prob_numerator<<<(n + WARP_SIZE - 1) / WARP_SIZE, WARP_SIZE, 0, stream>>>(
                pheromones, n, alpha
        );
        
        cudaGraph_t graph;
        cudaStreamEndCapture(stream, &graph);

        cudaGraphExec_t instance;
        cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
    #endif


    for (int i = 0; i < num_iter; i++) {
        #ifdef CUDAGRAPH
            cudaGraphLaunch(instance, stream);
            cudaStreamSynchronize(stream);
        #else
            queen_kernel<<<n, QUEEN_THREAD_NUM>>>(
                    n, tour, states
            );

            queen_update_pheromones_and_lengths<<<(n + WARP_SIZE - 1) / WARP_SIZE, WARP_SIZE>>>(
                    pheromones, n, tour, len, evaporate
            );

            calculate_prob_numerator<<<(n + WARP_SIZE - 1) / WARP_SIZE, WARP_SIZE>>>(
                    pheromones, n, alpha
            );
        #endif

        cudaMemcpy(len_h, len, sizeof(float) * n, cudaMemcpyDeviceToHost);

        if (min_len > *std::min_element(len_h, len_h + n)) {
            min_len = *std::min_element(len_h, len_h + n);
            id = std::min_element(len_h, len_h + n) - len_h;
            cudaMemcpy(tour_h, tour, sizeof(int) * n * n, cudaMemcpyDeviceToHost);
        }

        #ifdef VERBOSE
        std::cout << "Iteration " << i << ": " << min_len << "\n";
        #endif // VERBOSE
    }

    std::cout << min_len << "\n";

    std::vector<int> best_tour(tour_h + id * n, tour_h + (id + 1) * n);
    int zero_pos = std::find(best_tour.begin(), best_tour.end(), 0) - best_tour.begin();

    for (int i = 0; i < n; i++) {
        std::cout << best_tour[(zero_pos + i) % n] + 1 << " ";
    }

    cudaFree(states);
    cudaFree(tour);
    cudaFree(len);
    cudaFree(pheromones);
    cudaFreeHost(len_h);
    cudaFreeHost(tour_h);
    #ifdef CUDAGRAPH
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
        cudaGraphExecDestroy(instance);
    #endif
}