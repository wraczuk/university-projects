#ifndef PRUNING_H
#define PRUNING_H

#include "global.h"
#include "common.h"

using namespace std;

MPI_Win win = MPI_WIN_NULL;
void prepare_pruning() {
    long_edges.resize(high - low + 1);
    #pragma omp parallel for
    for (int i = 0; i < high - low + 1; i++) {
        auto it = partition(edges[i].begin(), edges[i].end(), [](pair<int, long long>& a){ return a.second < DELTA; });
        long_edges[i].reserve(edges[i].end() - it);
        for_each(it, edges[i].end(), [&](pair<int, long long> a){ long_edges[i].push_back(a); });
        edges[i].resize(it - edges[i].begin());
    }

    MPI_Win_create(dist.data(), sizeof(long long) * dist.size(),
                   sizeof(long long), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
}

template <typename T>
void push(int k, vector<vector<T>>& send_buffers, vector<vector<T>>& recv_buffers,
          vector<int>& send_count, vector<int>& recv_count,
          vector<MPI_Request>& send_req, vector<MPI_Request>& recv_req,
          vector<MPI_Status>& send_stat, vector<MPI_Status>& recv_stat, bool skip_long) {

    for (int i = 0; i < mpi_size; i++) {
        send_buffers[i].clear();
        recv_buffers[i].clear();
        send_count[i] = recv_count[i] = 0;
    }

    if (!skip_long) {
        for (auto i : buckets[k]) {
            for (const auto& e : long_edges[i - low]) {
                int v = e.first;
                long long w = e.second;
                if (is_mine(v) && dist[i - low] + w < dist[v - low]) {
                    long long cur_bucket = bucket_id(exchange(dist[v - low], dist[i - low] + w));
                    long long new_bucket = bucket_id(dist[v - low]);
                    if (new_bucket != cur_bucket)
                        buckets[new_bucket].push_back(v);
                }
                else {
                    send_buffers[whose(v)].emplace_back(dist[i - low] + w, v);
                }
            }
        }
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

    for (int i = 1; i < mpi_size; i++) {
        int to = (mpi_rank + i) % mpi_size;
        int from = (mpi_rank - i + mpi_size) % mpi_size;
        MPI_Isend(send_buffers[to].data(), 3 * send_count[to], MPI_INT, to, 0, MPI_COMM_WORLD, &send_req[i]);
        MPI_Irecv(recv_buffers[from].data(), 3 * recv_count[from], MPI_INT, from, 0, MPI_COMM_WORLD, &recv_req[i]);
    }

    for (int i = 1; i < mpi_size; i++) {
        int from = (mpi_rank - i + mpi_size) % mpi_size;
        MPI_Wait(&recv_req[i], &recv_stat[i]);
        for (auto [d, v] : recv_buffers[from]) {
            if (d < dist[v - low]) {
                long long cur_bucket = bucket_id(exchange(dist[v - low], (long long)d));
                long long new_bucket = bucket_id(dist[v - low]);
                if (new_bucket != cur_bucket)
                    buckets[new_bucket].push_back(v);
            }
        }
    }

    MPI_Waitall(mpi_size - 1, send_req.data(), send_stat.data());
}

template <typename T>
void pull(int k, vector<vector<T>>& send_buffers, vector<vector<T>>& recv_buffers,
            vector<int>& send_count, vector<int>& recv_count,
            vector<MPI_Request>& send_req, vector<MPI_Request>& recv_req,
            vector<MPI_Status>& send_stat, vector<MPI_Status>& recv_stat) {
    
    // push short outer edges
    push(k, send_buffers, recv_buffers, send_count, recv_count,
            send_req, recv_req, send_stat, recv_stat, true);
    
    vector<decltype(buckets)::iterator> buckets_to_check;
    for (auto it = buckets.upper_bound(k); it != buckets.end();) {
        it->second.resize(remove_if(it->second.begin(), it->second.end(),
                          [&](int u) { return bucket_id(dist[u - low]) != it->first; })
            - it->second.begin()
        );
        
        if (it->second.empty()) {
            auto cur = it;
            it++;
            if (it != buckets.end())
                buckets.erase(cur);
        }
        else {
            buckets_to_check.push_back(it);
            it++;
        }
    }

    MPI_Win_fence(MPI_MODE_NOPRECEDE | MPI_MODE_NOPUT, win);

    struct batch_t {
        long long dist;
        long long edge_len;
        int to;
    };
    constexpr int BATCH_SIZE = 1 << 17;
    vector<batch_t> batch(BATCH_SIZE);
    int pos = 0, finished = 0, cnt_finished = 0;

    auto free_batch = [&]() {
        MPI_Win_fence(MPI_MODE_NOPUT | MPI_MODE_NOSTORE, win);
        for (int j = 0; j < pos; j++) {
            int i = batch[j].to - low;
            if (bucket_id(batch[j].dist) == k && batch[j].dist + batch[j].edge_len < dist[i]) {
                long long cur_bucket = bucket_id(exchange(dist[i], batch[j].dist + batch[j].edge_len));
                long long new_bucket = bucket_id(dist[i]);
                if (new_bucket != cur_bucket)
                    buckets[new_bucket].push_back(i + low);
            }
        }
        pos = 0;
        MPI_Allreduce(&finished, &cnt_finished, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    };

    for (const auto& it : buckets_to_check) {
        for (auto i : it->second) {
            for (const auto& e : long_edges[i - low]) {
                if (k * DELTA + e.second < dist[i - low]) {
                    batch[pos] = {.edge_len = e.second, .to = i};
                    MPI_Get(&batch[pos].dist, 1, MPI_LONG_LONG, whose(e.first), trans(e.first), 1, MPI_LONG_LONG, win);
                    if (++pos == BATCH_SIZE)
                        free_batch();
                }
            }
        }
    }
    finished = 1;
    while (cnt_finished != mpi_size)
        free_batch();

    MPI_Win_fence(MPI_MODE_NOSUCCEED, win);
}

void pruning_delta_stepping(int k) {
    auto it = buckets.find(k);

    // cerr << k << " " << (it == buckets.end()) << endl;

    if (it != buckets.end())
        it->second.resize(remove_if(it->second.begin(), it->second.end(),
                          [&](int u) { return bucket_id(dist[u - low]) != k; })
            - it->second.begin()
        );
    vector<int> active(it != buckets.end() ? it->second : vector<int>());
    vector<int> next_iteration_active;
    next_iteration_active.reserve(active.size());

    // vector<vector<pair<long long, int>>> send_buffers(mpi_size);
    // vector<vector<pair<long long, int>>> recv_buffers(mpi_size);
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

            for (const auto& e : edges[u - low]) {
                int v = e.first;
                long long w = e.second;
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

        for (int i = 1; i < mpi_size; i++) {
            int to = (mpi_rank + i) % mpi_size;
            int from = (mpi_rank - i + mpi_size) % mpi_size;
            MPI_Isend(send_buffers[to].data(), 3 * send_count[to], MPI_INT, to, 0, MPI_COMM_WORLD, &send_req[i - 1]);
            MPI_Irecv(recv_buffers[from].data(), 3 * recv_count[from], MPI_INT, from, 0, MPI_COMM_WORLD, &recv_req[i - 1]);
        }

        for (int i = 0; i < mpi_size; i++) {
            int from = (mpi_rank - i + mpi_size) % mpi_size;
            MPI_Wait(&recv_req[i], &recv_stat[i]);
            for (auto [d, v] : recv_buffers[from]) {
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

    long long push_vol = 0;
    double pull_vol = 0;
    if (it != buckets.end()) {
        for (auto i : it->second)
            if (bucket_id(dist[i - low]) == k)
                push_vol += long_edges[i - low].size();
    }
    for (auto i = buckets.upper_bound(k); i != buckets.end();) {
        i->second.resize(remove_if(i->second.begin(), i->second.end(),
                          [&](int u) { return bucket_id(dist[u - low]) != i->first; })
            - i->second.begin()
        );
        if (i->second.empty()) {
            auto cur = i;
            i++;
            if (i != buckets.end())
                buckets.erase(cur);
        }
        else {
            for (auto j : i->second)
                pull_vol += long_edges[j - low].size() * min(1., (double)(dist[j - low] - (k + 1) * DELTA) / (double)(max_w - DELTA));
            i++;
        }
    }

    long long sum_push_vol = 0;
    double sum_pull_vol = 0;
    MPI_Allreduce(&push_vol, &sum_push_vol, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&pull_vol, &sum_pull_vol, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (sum_push_vol < sum_pull_vol)
        // cerr << "push " << sum_push_vol << " " << sum_pull_vol << "\n",
        push(k, send_buffers, recv_buffers, send_count, recv_count,
                send_req, recv_req, send_stat, recv_stat, false);
    else
        // cerr << "pull " << sum_pull_vol << " " << sum_push_vol << "\n",
        pull(k, send_buffers, recv_buffers, send_count, recv_count,
                send_req, recv_req, send_stat, recv_stat);
}

#endif // PRUNING_H
