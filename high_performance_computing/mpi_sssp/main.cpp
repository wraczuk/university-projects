// #define LOCAL
// #define REMOVE_REPETITIONS
// #define EDGE_CLASSIFICATION
// #define PRUNING
// #define NAIVE
// #define HYBRIDIZATION
// #define MEASURE_TIME

#include "solution.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    // cerr << "RUNNING" << endl;
    auto T0 = get_time();
    read_input(argc, argv);
    auto T1 = get_time();
    // cerr << "READ " << argv[1] << " " << argv[2] << endl;

    #ifdef MEASURE_TIME
    if (mpi_rank == 0)
        cerr << "Reading time: " << (T1 - T0) / 1e6 << "\n";
    #endif // MEASURE_TIME

    sssp();
    for (int i = low; i <= high; i++) {
        cout << dist[i - low] << "\n";
    }
    // cerr << "PRINT" << endl;
    auto T2 = get_time();
    
    #ifdef MEASURE_TIME
    if (mpi_rank == 0)
        cerr << "Total time: " << (T1 - T0) / 1e6 << "\n";
    #endif // MEASURE_TIME

    #ifdef PRUNING
    if (win != MPI_WIN_NULL)
        MPI_Win_free(&win);
    #endif // PRUNING

    MPI_Finalize();
    return 0;
}

// mpiexec -n 4 ./sssp checker/tests/path_20_4/ checker/outputs/path_20_4/
// mpiexec -n 4 xterm -e gdb --args ./sssp checker/tests/path_20_4/ checker/outputs/path_20_4/
