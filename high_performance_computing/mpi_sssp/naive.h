#ifndef NAIVE_H
#define NAIVE_H

#include "common.h"

using namespace std;

void naive_delta_stepping(int k) {
    auto it = buckets.find(k);

    if (it != buckets.end())
        it->second.resize(remove_if(it->second.begin(), it->second.end(),
                          [&](int u) { return bucket_id(dist[u - low]) != k; })
            - it->second.begin()
        );
    vector<int> active(it != buckets.end() ? it->second : vector<int>());
    vector<int> next_iteration_active;
    next_iteration_active.reserve(active.size());

    vector<vector<plli>> send_buffers(mpi_size);
    vector<vector<plli>> recv_buffers(mpi_size);
    vector<int> send_count(mpi_size);
    vector<int> recv_count(mpi_size);
    vector<MPI_Request> send_req(mpi_size, MPI_REQUEST_NULL);
    vector<MPI_Request> recv_req(mpi_size, MPI_REQUEST_NULL);
    vector<MPI_Status> send_stat(mpi_size);
    vector<MPI_Status> recv_stat(mpi_size);

    bool is_active = !active.empty();
    bool is_any_active;
    MPI_Allreduce(&is_active, &is_any_active, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);

    for (; is_any_active; swap(active, next_iteration_active), next_iteration_active.clear(),
                          is_active = !active.empty(),
                          MPI_Allreduce(&is_active, &is_any_active, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD)) {
        
        it_cnt++;

        for (int i = 0; i < mpi_size; i++) {
            send_buffers[i].clear();
            recv_buffers[i].clear();
            send_count[i] = recv_count[i] = 0;
        }

        for (int u : active) {

            #ifdef REMOVE_REPETITIONS
            if (when_relaxed[u - low] == it_cnt)
                continue;
            when_relaxed[u - low] = it_cnt;
            #endif // REMOVE REPETITIONS
            
            for (const auto &edge : edges[u - low]) {
                int v = edge.first;
                long long w = edge.second;
                // local edge
                if (is_mine(v)) {
                    if (dist[u - low] + w < dist[v - low]) {
                        long long cur_bucket = bucket_id(exchange(dist[v - low], dist[u - low] + w));
                        long long new_bucket = bucket_id(dist[v - low]);
                        if (new_bucket == k)
                            next_iteration_active.push_back(v);
                        if (new_bucket != cur_bucket)
                            buckets[new_bucket].push_back(v);
                    }
                }
                // remote edge
                else {
                    int peer = whose(v);
                    send_buffers[peer].emplace_back(dist[u - low] + w, v);                
                }
            }
        }

        for (int i = 0; i < mpi_size; i++)
            send_count[i] = send_buffers[i].size();

        MPI_Alltoall(send_count.data(), 1, MPI_INT,
                     recv_count.data(), 1, MPI_INT, MPI_COMM_WORLD);
 
        for (int i = 0; i < mpi_size; i++) 
            recv_buffers[i].resize(recv_count[i]);

        MPI_Barrier(MPI_COMM_WORLD);
        for (int i = 1; i < mpi_size; i++) {
            int to = (mpi_rank + i) % mpi_size;
            int from = (mpi_rank - i + mpi_size) % mpi_size;
            MPI_Isend(send_buffers[to].data(), 3 * send_count[to], MPI_INT, to, 0, MPI_COMM_WORLD, &send_req[i - 1]);
            MPI_Irecv(recv_buffers[from].data(), 3 * recv_count[from], MPI_INT, from, 0, MPI_COMM_WORLD, &recv_req[i - 1]);
        }

        MPI_Waitall(mpi_size - 1, recv_req.data(), recv_stat.data());

        for (int i = 0; i < mpi_size; i++) {
            for (auto [d, v] : recv_buffers[i]) {
                if (d < dist[v - low]) {
                    long long cur_bucket = bucket_id(exchange(dist[v - low], (long long)d));
                    long long new_bucket = bucket_id(dist[v - low]);
                    if (new_bucket == k)
                        next_iteration_active.push_back(v);
                    if (new_bucket != cur_bucket)
                        buckets[new_bucket].push_back(v);
                }
            }
        }

        MPI_Waitall(mpi_size - 1, send_req.data(), send_stat.data());
    }
}

#endif // NAIVE_H