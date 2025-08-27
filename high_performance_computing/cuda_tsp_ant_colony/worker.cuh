#pragma once
#include <algorithm>
#include <cfloat>

#include "common.cuh"

#define WORKER_THREAD_NUM       32
#define warp_pos(i)             (n * threadIdx.x + (i))
#define tour_pos(i) (n * (i) + blockDim.x * blockIdx.x + threadIdx.x)


__global__ void worker_kernel(int n, int *tour, curandState *states)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)
        return;

    int current_city = (int)(curand_uniform(&states[idx]) * n + 1) % n;
    tour[tour_pos(0)] = current_city;

    __shared__ bool not_visited[WORKER_THREAD_NUM * MAX_N + 16];
    memset(not_visited + warp_pos(0), 1, sizeof(bool) * MAX_N);

    not_visited[warp_pos(current_city)] = false;

    float prob[MAX_N + 4];
    for (int step = 1; step < n; step++) {
        float sum_prob = 0.0;
        // Calculate the probabilities for each unvisited city
        for (int j = 0, cr = cord(current_city, 0); j < n; j++, cr++) {
            float tmp = prob_numerator[cr] * not_visited[warp_pos(j)];
            prob[j] = tmp;
            sum_prob += tmp;
        }
      
        float rand_val = curand_uniform(&states[idx]) * sum_prob;
        float cumulative_prob = 0.0;

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
        tour[tour_pos(step)] = j;
        not_visited[warp_pos(j)] = false;
    }
}

__global__ void worker_update_pheromones_and_lengths(float *pheromones, int n, int *tour, float *len, float evaporate) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)
        return;

    len[idx] = 0.0;
    for (int i = idx; i < n * n; i += n) {
        pheromones[i] *= (1.0 - evaporate);
    }
    __syncthreads();
    
    for (int i = 0; i < n; i++) {
        int city_a = tour[tour_pos(i)];
        int city_b = tour[tour_pos((i + 1) % n)];
        len[idx] += dist_dev[cord(city_a, city_b)];
    }

    float inv_len = 1.0 / len[idx];
    for (int i = 0; i < n; i++) {
        int city_a = tour[tour_pos(i)];
        int city_b = tour[tour_pos((i + 1) % n)];
        atomicAdd(&pheromones[cord(city_a, city_b)], inv_len);
    }
    __syncthreads();
}

void worker(float *dist, int n, float alpha, float beta,
            float evaporate, unsigned long seed, int num_iter)
{
    curandState *states;
    cudaMalloc((void **)&states, sizeof(curandState) * n);
    init_rng<<<(n + WORKER_THREAD_NUM - 1) / WORKER_THREAD_NUM, WORKER_THREAD_NUM>>>(states, seed, n);

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

    calculate_prob_numerator<<<(n + WORKER_THREAD_NUM - 1) / WORKER_THREAD_NUM, WORKER_THREAD_NUM>>>(pheromones, n, alpha);

    float min_len = 1e18;
    float* len_h;
    int* tour_h;
    cudaHostAlloc((void**)&len_h, sizeof(float) * n, cudaHostAllocDefault);
    cudaHostAlloc((void**)&tour_h, sizeof(int) * n * n, cudaHostAllocDefault);
    int id = -1;

    #ifdef CUDAGRAPH
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        worker_kernel<<<n, WORKER_THREAD_NUM, 0, stream>>>(
            n, tour, states
        );

        worker_update_pheromones_and_lengths<<<(n + WORKER_THREAD_NUM - 1) / WORKER_THREAD_NUM, WORKER_THREAD_NUM, 0, stream>>>(
                pheromones, n, tour, len, evaporate
        );

        calculate_prob_numerator<<<(n + WORKER_THREAD_NUM - 1) / WORKER_THREAD_NUM, WORKER_THREAD_NUM, 0, stream>>>(
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
            worker_kernel<<<(n + WORKER_THREAD_NUM - 1) / WORKER_THREAD_NUM, WORKER_THREAD_NUM>>>(
                    n, tour, states
            );

            worker_update_pheromones_and_lengths<<<(n + WORKER_THREAD_NUM - 1) / WORKER_THREAD_NUM, WORKER_THREAD_NUM>>>(
                    pheromones, n, tour, len, evaporate
            );

            calculate_prob_numerator<<<(n + WORKER_THREAD_NUM - 1) / WORKER_THREAD_NUM, WORKER_THREAD_NUM>>>(
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

    std::vector<int> best_tour(n);
    for (int i = 0; i < n; i++) {
        best_tour[i] = tour_h[i * n + id];
    }

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
}
