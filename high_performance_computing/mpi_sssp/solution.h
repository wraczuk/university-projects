#ifndef SOLUTION_H
#define SOLUTION_H

#include "common.h"
#include "naive.h"
#include "edge_classification.h"
#include "pruning.h"
#include "hybridization.h"

void relax_bucket(long long k) {
    #ifdef PRUNING
    pruning_delta_stepping(k);
    #elif defined EDGE_CLASSIFICATION
    classification_delta_stepping(k);
    #else
    naive_delta_stepping(k);
    #endif
}

void sssp(long long override_delta=-1) {
    if (override_delta > 0) {
        vector<int> last(std::move(buckets[bucket_id(INF)]));
        buckets.erase(bucket_id(INF));
        DELTA = override_delta;
        buckets[bucket_id(INF)].insert(buckets[bucket_id(INF)].end(), last.begin(), last.end());
    }

    auto T0 = get_time();
    #ifdef PRUNING
    prepare_pruning();
    #elif defined EDGE_CLASSIFICATION
    prepare_edge_classification();    
    #endif
    auto T1 = get_time();

    #ifdef MEASURE_TIME
    cerr << "Edge classification preparation time: " << (T1 - T0) / 1e6 << " s\n";
    #endif // MEASURE_TIME

    int settled = 0;
    for (long long k = 0, min_k = 0; min_k < bucket_id(INF);
        k = buckets.upper_bound(k)->first, MPI_Allreduce(&k, &min_k, 1, MPI_LONG_LONG, MPI_MIN, MPI_COMM_WORLD)) {

        relax_bucket(k = min_k);

        auto it = buckets.find(k);
        settled += it != buckets.end() ? it->second.size() : 0;
        
        #ifdef HYBRIDIZATION
        int sum_settled = 0;
        MPI_Allreduce(&settled, &sum_settled, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (n * CUTOFF_COEF <= sum_settled) {
            vector<int> active;
            for (it = buckets.lower_bound(k); it != buckets.end(); it++) {
                for_each(it->second.begin(), it->second.end(),
                    [&](int u) {
                        if (bucket_id(dist[u - low]) == it->first) {
                            active.push_back(u);
                            edges[u - low].insert(edges[u - low].end(),
                                    long_edges[u - low].begin(), long_edges[u - low].end());
                        }
                    });
            }
            // cerr << "BELLMAN FORD" << endl;
            bellman_ford(active);
            break;
        }
        #endif // HYBRIDIZATION
    }
}

#endif // SOLUTION_H