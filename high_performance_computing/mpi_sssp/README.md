# Distributed Single-Source Shortest Path using MPI and the Δ-stepping Algorithm

## Project Overview

This project presents a high-performance, distributed implementation of the Single-Source Shortest Path (SSSP) algorithm for large-scale graphs. The solution is developed using the Message Passing Interface (OpenMPI implementation) framework to enable parallel processing on distributed memory systems.

The core of the implementation is the **Δ-stepping algorithm**, which serves as an effective intermediate between the work-efficient but less parallelizable Dijkstra's algorithm and the highly parallel but computationally more expensive Bellman-Ford algorithm. This approach allows for a tunable balance between work and parallelism, making it suitable for modern cluster environments.

## Core Algorithm: The $Δ$-stepping Algorithm

The implementation is centered around the $Δ$-stepping algorithm. This algorithm partitions vertices into buckets based on tentative distance increments of a parameter $Δ$. In each phase, it concurrently relaxes light edges (weight $< Δ$) from the current bucket and moves vertices to new buckets, exposing a high degree of parallelism. Heavy edges (weight $\ge Δ$) are handled separately.

## Implemented Optimizations

To enhance performance and scalability, this implementation incorporates several advanced heuristic optimizations as described in the reference paper. These optimizations address challenges such as load balancing, redundant work, and communication overhead.

The key optimizations include:
* **Edge Classification**: Edges are categorized to handle them more efficiently, incorporating an **inner-outer short heuristic**.
* **Pruning and Push vs. Pull Hybridization**: A dynamic strategy that combines the benefits of both push-based and pull-based relaxation methods to reduce unnecessary computations and communication.
* **Load Balancing**: Both inter-node (between MPI processes) and intra-node (within a single process using OpenMP) load balancing strategies are implemented to ensure an even distribution of work and minimize idle time.

## Distributed Architecture and Data Handling

The solution is designed for distributed memory systems using MPI. The input graph is partitioned and distributed among the MPI processes before execution.

### Input Format
Each MPI process reads its own shard of the graph from a dedicated input file.
* The first line contains three integers: the total number of vertices in the graph ($N$), the starting vertex index for the process ($V_{start}$), and the ending vertex index for the process ($V_{end}$).
* Subsequent lines describe the edges incident to the vertices managed by the process. Each line contains three integers: a source vertex `u`, a destination vertex `v`, and the edge weight `w`.

The implementation operates under the following specific assumptions for the input graph:
* The graph is connected.
* The graph contains no self-loops or repeated edges.
* The data shard assigned to a single process fits into its main memory.
* The total number of vertices is less than $10^9$.
* Edge weights are non-negative integers, and the sum of edge weights along any shortest path is less than $10^{18}$.
* Vertices are distributed among processes according to a strict partitioning scheme:
    * The distribution is as even as possible (with at most a one-vertex difference between any two processes).
    * Vertex indices are assigned to processes contiguously and in increasing order of MPI rank.
    * A process with rank `i` is guaranteed to be assigned at least as many vertices as a process with rank `i+1`.

### Output Format
Upon completion, each process writes its portion of the results to a specified output file. For each vertex the process is responsible for, it writes a single line containing the calculated shortest path distance from the source vertex (vertex 0).

## Performance Analysis

A significant component of this project was the empirical evaluation of the implementation's performance and scalability. The analysis focused on:
* **Weak Scaling (Gustafson's Law)**: Measuring the ability of the solution to handle a growing problem size with a proportionally increasing number of processors.
* **Optimization Impact**: Quantifying the performance improvements gained from each of the implemented heuristics.

Experiments were conducted on a high-performance computing cluster to provide realistic performance metrics.

## Build and Execution

The project is compiled using a standard `Makefile`, which produces an executable named `sssp`.

## Execution:
The binary is intended to be launched in an MPI environment. Each rank requires paths to its specific input file and a designated output file.
```bash
mpiexec -n <number_of_processes> ./sssp <input_file_path> <output_file_path>
```

## References

Meyer, U., Sanders, P. (2014). Δ-Stepping: A Parallelizable Shortest Path Algorithm. This project's core algorithm and optimizations are based on techniques described in subsequent research building on this foundation, particularly the paper:

Madduri, K., et al. (2014). "Parallel Shortest Path Algorithms for Massive Graphs on a GPGPU/CPU Hybrid System". 2014 IEEE 28th International Parallel and Distributed Processing Symposium. The specific heuristics are detailed here: https://www.odbms.org/wp-content/uploads/2014/05/sssp-ipdps2014.pdf

