#ifndef GLOBAL_H
#define GLOBAL_H

#include <vector>
#include <list>
#include <map>
#include <utility>

using namespace std;

constexpr long long INF = 1e18;
constexpr size_t LONG_EDGES_OMP_THRESHOLD = 1e5;

static long long DELTA;
static long long max_w;
static int mpi_rank, mpi_size;
static int n, low, high;
static vector<long long> dist;
static vector<vector<pair<int, long long>>> edges, long_edges;
static map<long long, vector<int>> buckets;

static long long it_cnt;
static vector<long long> when_relaxed;

#endif // GLOBAL_H