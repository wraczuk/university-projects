# GPU Ant Colony Optimization for Traveling Salesman Problem (CUDA)

## Overview
This project implements GPU-based **Ant Colony Optimization (ACO)** to solve the **Traveling Salesman Problem (TSP)** using **CUDA**.  
The TSP is a classic NP-hard problem where a salesman must visit all cities exactly once and return to the starting point while minimizing total travel distance. Due to its computational complexity, it is a strong candidate for parallel GPU acceleration.

ACO is a bio-inspired metaheuristic that simulates how ants collectively find near-optimal paths using pheromone trails and probabilistic exploration. This project explores two GPU implementations of ACO and compares their correctness, scalability, and performance.

## Implemented Variants
- **Worker Ant Model**  
  Each CUDA thread simulates one ant independently.

- **Queen Ant Model**  
  Each CUDA block cooperatively builds a tour, with one thread per city.

Both implementations include:
- Cycle construction (each city visited exactly once).
- Cycle length calculation.
- Pheromone updates and evaporation.
- Iterative execution wrapped in **CUDA Graphs** for efficiency.

## Features
- Fully GPU-based computation (no external libraries, only CUDA).
- Support for TSPLIB input format (with EUC_2D, CEIL_2D, GEO distance types).
- Optimized memory access and warp-level parallelism.
- Configurable parameters for pheromone influence, heuristic weight, and evaporation.

## Input Format
The solver accepts input in **TSPLIB format** (city coordinates).  
Example:
```
NAME : example3
TYPE : TSP
DIMENSION : 3
EDGE_WEIGHT_TYPE : EUC_2D
NODE_COORD_SECTION
1 3 5
2 4.5 -6
3 8.62630e+02 5
EOF
```

## Output Format
- First line: shortest cycle length found.  
- Second line: cycle order (space-separated list of cities).  

## Usage
```bash
./acotsp <input_file> <output_file> <TYPE> <NUM_ITER> <ALPHA> <BETA> <EVAPORATE> <SEED>
```

- `<TYPE>` : WORKER | QUEEN  
- `<NUM_ITER>` : Number of iterations  
- `<ALPHA>` : Influence of pheromone  
- `<BETA>` : Influence of heuristic (distance)  
- `<EVAPORATE>` : Pheromone evaporation factor  
- `<SEED>` : Random seed for reproducibility  

## Reference
Based on concepts from:  
**Enhancing Data Parallelism for Ant Colony Optimisation on GPUs**  
Jose M. Cecilia, Jose M. García  
DOI: [10.1016/j.jpdc.2012.01.002](https://doi.org/10.1016/j.jpdc.2012.01.002)
