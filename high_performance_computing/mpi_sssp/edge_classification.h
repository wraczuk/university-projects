#ifndef CLASSIFICATION_H
#define CLASSIFICATION_H

#include "common.h"

using namespace std;

void prepare_edge_classification() {
    long_edges.resize(high - low + 1);
    #pragma omp parallel for
    for (int i = 0; i < high - low + 1; i++) {
        auto it = partition(edges[i].begin(), edges[i].end(), [](pair<int, long long>& a){ return a.second < DELTA; });
        long_edges[i].reserve(edges[i].end() - it);
        for_each(it, edges[i].end(), [&](pair<int, long long> a){ long_edges[i].push_back(a); });
        edges[i].resize(it - edges[i].begin());
    }
}

template <typename T>
void push(int k, vector<vector<T>>& send_buffers, vector<vector<T>>& recv_buffers,
            vector<int>& send_count, vector<int>& recv_count,
            vector<MPI_Request>& send_req, vector<MPI_Request>& recv_req,
            vector<MPI_Status>& send_stat, vector<MPI_Status>& recv_stat) {
    for (int i = 0; i < mpi_size; i++) {
        send_buffers[i].clear();
        recv_buffers[i].clear();
        send_count[i] = recv_count[i] = 0;
    }

    for (auto i : buckets[k]) {
        for (const auto& e : long_edges[i - low])
            send_buffers[whose(e.first)].emplace_back(dist[i - low] + e.second, e.first);
    }
    
    for (auto i : buckets[k])
        for (const auto& e : edges[i - low]) {
            int peer = whose(e.first);
            if (peer != mpi_rank && bucket_id(dist[i - low] + e.second) != k)
                send_buffers[peer].emplace_back(dist[i - low] + e.second, e.first);
        }
    
    for (int i = 0; i < mpi_size; i++)
        send_count[i] = send_buffers[i].size();

    MPI_Alltoall(send_count.data(), 1, MPI_INT,
                 recv_count.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    for (int i = 0; i < mpi_size; i++) 
        recv_buffers[i].resize(recv_count[i]);

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < mpi_size; i++) {
        int to = (mpi_rank + i) % mpi_size;
        int from = (mpi_rank - i + mpi_size) % mpi_size;
        MPI_Isend(send_buffers[to].data(), 3 * send_count[to], MPI_INT, to, 0, MPI_COMM_WORLD, &send_req[i]);
        MPI_Irecv(recv_buffers[from].data(), 3 * recv_count[from], MPI_INT, from, 0, MPI_COMM_WORLD, &recv_req[i]);
    }

    MPI_Waitall(mpi_size, recv_req.data(), recv_stat.data());

    for (int i = 0; i < mpi_size; i++) {
        for (auto [d, v] : recv_buffers[i]) {
            if (d < dist[v - low]) {
                long long cur_bucket = bucket_id(exchange(dist[v - low], (long long)d));
                long long new_bucket = bucket_id(dist[v - low]);
                if (new_bucket != cur_bucket)
                    buckets[new_bucket].push_back(v);
            }
        }
    }

    MPI_Waitall(mpi_size, send_req.data(), send_stat.data());
}

void classification_delta_stepping(int k) {
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
                    if (bucket_id(dist[u - low] + w) == k) 
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

    push(k, send_buffers, recv_buffers, send_count, recv_count,
            send_req, recv_req, send_stat, recv_stat);
}

#endif // CLASSIFICATION_H