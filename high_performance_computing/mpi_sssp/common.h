#ifndef COMMON_H
#define COMMON_H

#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdio>
#include <unordered_map>
#include <map>
#include <climits>
#include <utility>
#include <algorithm>
#include <memory>
#include <chrono>
#include <cstdio>
#include <omp.h>
#include <cassert>

#include "global.h"

using namespace std;

long long bucket_id(long long d) {
    return d / DELTA;
}

bool is_mine(int u) {
    return low <= u && u <= high;
}

static int bigger_chunk;
static int first_smaller_chunk_id;
int whose(int u) {
    // u < x * bigger_chunk - max(0, x - first_smaller_chunk_id )
    // u < x * (bigger_chunk - 1) + first_smaller_chunk_id
    // x > (u - first_smaller_chunk_id) / (bigger_chunk - 1)
    if (u < first_smaller_chunk_id * bigger_chunk) {
        return u / bigger_chunk;
    }
    else {
        return (u - first_smaller_chunk_id) / (bigger_chunk - 1);
    }
}

int lowest(int u) {
    if (u < first_smaller_chunk_id * bigger_chunk) {
        return u / bigger_chunk * bigger_chunk;
    }
    else {
        int off = first_smaller_chunk_id * bigger_chunk;
        return off + (u - off) / (bigger_chunk - 1) * (bigger_chunk - 1);
    }
}

int trans(int u) {
    return u - lowest(u);
}

long long get_time() {
    return chrono::duration_cast<chrono::microseconds>(
        chrono::system_clock::now().time_since_epoch()).count();
}

unsigned read_uint() {
    unsigned int num = 0;
    int c = getchar_unlocked();
    if (c == EOF) {
        return (unsigned)-1;
    }
    
    while (c < '0' || c > '9') {
        c = getchar_unlocked();
    }

    while (c >= '0' && c <= '9') {
        num = num * 10 + (c - '0');
        c = getchar_unlocked();
    }

    return num;
}

unsigned long long read_ulong() {
    unsigned long long num = 0;
    int c = getchar_unlocked();
    if (c == EOF) {
        return (unsigned)-1;
    }

    while (c < '0' || c > '9') {
        c = getchar_unlocked();
    }

    while (c >= '0' && c <= '9') {
        num = num * 10 + (c - '0');
        c = getchar_unlocked();
    }

    return num;
}


void read_input(int argc, char **argv) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <input file> <output file>\n";;
        exit(EXIT_FAILURE);
    }

    #ifdef LOCAL
    string input = argv[1] + to_string(mpi_rank) + ".in";
    string output = argv[2] + to_string(mpi_rank) + ".out";
    cout << "FROM " << mpi_rank << " " << input << " " << output << std::endl;
    auto trash = freopen(input.data(), "r", stdin);
    trash = freopen(output.data(), "w", stdout);
    #else
    auto trash = freopen(argv[1], "r", stdin);
    trash = freopen(argv[2], "w", stdout);
    #endif // LOCAL

    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    // cin >> n >> low >> high;
    n = read_uint();
    low = read_uint();
    high = read_uint();
    edges.resize(high - low + 1);

    #ifdef REMOVE_REPETITIONS
    when_relaxed.resize(high - low + 1);
    #endif // REMOVE_REPETITIONS

    int u, v;
    long long w = -1;
    vector<int> cnt(high - low + 1, 0);
    // while (cin >> u >> v >> w) {
    while ((u = read_uint()) != (unsigned)-1) {
        v = read_uint();
        w = read_ulong();
        if (low <= u && u <= high)
            edges[u - low].emplace_back(v, w);
        if (low <= v && v <= high)
            edges[v - low].emplace_back(u, w);
        max_w = max(max_w, w);
    }

    MPI_Allreduce(&max_w, &DELTA, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    DELTA = max(DELTA / 5, 1ll);

    // Process 0 always has bigger chunk
    bigger_chunk = high - low + 1;
    MPI_Bcast(&bigger_chunk, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int size_and_id[2] = {high - low + 1, mpi_rank};
    int reduced_size_and_id[2];
    MPI_Allreduce(size_and_id, reduced_size_and_id, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);
    if (reduced_size_and_id[1] == 0)
        first_smaller_chunk_id = mpi_size;
    else
        first_smaller_chunk_id = reduced_size_and_id[1];

    dist.resize(high - low + 1, INF);
    buckets[bucket_id(INF)] = [&](){ 
        vector<int> ret(high - low + 1);
        for (int i = low; i <= high; i++)
            ret[i - low] = i;
        return ret;
    }();

    if (low == 0) {
        dist[0] = 0;
        buckets[0] = {0};
    } 
}

struct __attribute__((packed)) plli {
    long long first;
    int second;

    plli(long long f, int s) : first(f), second(s) {}
    plli() : first(0), second(0) {}
};

#endif // COMMON_H
