// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Thiago Egon Lange
// CUDA kernels for AI-Lake GPU-accelerated vector search.
//
// Build with:   cmake -DAILAKE_CUDA=ON ..
// Requires:     CUDA Toolkit 11.8+ (sm_70+)
//
// Usage pattern:
//   ailake::cuda::GpuSearchContext ctx(dim, n_vectors, metric);
//   ctx.upload(flat_vectors, row_ids);
//   ctx.search(query, top_k, results);
#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

// ── Error check macro ─────────────────────────────────────────────────────────
#define AILAKE_CUDA_CHECK(call)                                             \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess)                                               \
            throw std::runtime_error(                                        \
                std::string("CUDA error: ") + cudaGetErrorString(_e) +       \
                " at " __FILE__ ":" + std::to_string(__LINE__));             \
    } while(0)

namespace ailake {
namespace cuda {

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------

// Batch cosine distance: query[dim] vs vectors[n × dim] → distances[n]
// Each thread handles one vector.
__global__ void cosine_distance_kernel(
    const float* __restrict__ query,    // [dim]
    const float* __restrict__ vectors,  // [n × dim], row-major
    float* __restrict__       distances,// [n]
    int n, int dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float* v = vectors + (long long)i * dim;
    double dot = 0, na = 0, nb = 0;
    for (int j = 0; j < dim; ++j) {
        double qj = query[j], vj = v[j];
        dot += qj * vj;
        na  += qj * qj;
        nb  += vj * vj;
    }
    float sim = (na > 0 && nb > 0)
        ? (float)(dot / (sqrt(na) * sqrt(nb)))
        : 0.f;
    if (sim >  1.f) sim =  1.f;
    if (sim < -1.f) sim = -1.f;
    distances[i] = 1.f - sim;
}

// Batch euclidean distance.
__global__ void euclidean_distance_kernel(
    const float* __restrict__ query,
    const float* __restrict__ vectors,
    float* __restrict__       distances,
    int n, int dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* v = vectors + (long long)i * dim;
    double sum = 0;
    for (int j = 0; j < dim; ++j) {
        double d = query[j] - v[j];
        sum += d * d;
    }
    distances[i] = (float)sqrt(sum);
}

// Batch dot-product distance (negated so lower = more similar).
__global__ void dot_distance_kernel(
    const float* __restrict__ query,
    const float* __restrict__ vectors,
    float* __restrict__       distances,
    int n, int dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* v = vectors + (long long)i * dim;
    double sum = 0;
    for (int j = 0; j < dim; ++j) sum += (double)query[j] * v[j];
    distances[i] = -(float)sum;
}

// ---------------------------------------------------------------------------
// IVF-PQ ADC kernel — compute approximate distances over one inverted list.
//
// lut[j*k + c]  = precomputed dist(query_sub_j, codebook[j][c])
// codes[r*m + j] = PQ code for vector r, sub-space j
// distances[r]   = sum of lut lookups
// ---------------------------------------------------------------------------
__global__ void ivfpq_adc_kernel(
    const float*   __restrict__ lut,        // [m × k]  precomputed on host
    const uint8_t* __restrict__ codes,      // [n_codes × m]
    float*         __restrict__ distances,  // [n_codes]
    int n_codes, int m, int k)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_codes) return;
    float d = 0.f;
    const uint8_t* row = codes + (long long)r * m;
    for (int j = 0; j < m; ++j) {
        uint8_t c = row[j];
        d += lut[j * k + c];
    }
    distances[r] = d;
}

// ---------------------------------------------------------------------------
// Shared-memory LUT version of ADC (faster when m×k fits in smem).
// Launch with: blockDim.x = 256, blockDim.y = 1
// Shared mem:  m × k × sizeof(float) bytes per block
// ---------------------------------------------------------------------------
__global__ void ivfpq_adc_smem_kernel(
    const float*   __restrict__ lut,
    const uint8_t* __restrict__ codes,
    float*         __restrict__ distances,
    int n_codes, int m, int k)
{
    extern __shared__ float smem_lut[];
    // Load LUT into shared memory cooperatively
    int tid = threadIdx.x;
    int lut_sz = m * k;
    for (int i = tid; i < lut_sz; i += blockDim.x)
        smem_lut[i] = lut[i];
    __syncthreads();

    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_codes) return;
    float d = 0.f;
    const uint8_t* row = codes + (long long)r * m;
    for (int j = 0; j < m; ++j)
        d += smem_lut[j * k + row[j]];
    distances[r] = d;
}

// ---------------------------------------------------------------------------
// Top-K selection on GPU (partial sort via min-heap in shared memory).
// For simplicity, done on host after kernel — see GpuSearchContext::search().
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// GpuSearchContext — owns GPU memory for a set of flat vectors.
//
// Usage:
//   GpuSearchContext ctx(1536, 50000, Metric::Cosine);
//   ctx.upload(host_flat_vecs.data(), host_row_ids.data());
//   auto results = ctx.search(query.data(), 10);
// ---------------------------------------------------------------------------
class GpuSearchContext {
public:
    GpuSearchContext(int dim, int n, int metric_int = 0)
        : dim_(dim), n_(n), metric_(metric_int) {
        AILAKE_CUDA_CHECK(cudaMalloc(&d_vecs_,     (size_t)n * dim * sizeof(float)));
        AILAKE_CUDA_CHECK(cudaMalloc(&d_query_,    dim * sizeof(float)));
        AILAKE_CUDA_CHECK(cudaMalloc(&d_distances_, (size_t)n * sizeof(float)));
        row_ids_.resize(n);
    }

    ~GpuSearchContext() {
        cudaFree(d_vecs_);
        cudaFree(d_query_);
        cudaFree(d_distances_);
    }

    // Upload host flat vectors [n × dim] and row IDs [n].
    void upload(const float* host_vecs, const uint64_t* host_row_ids) {
        AILAKE_CUDA_CHECK(cudaMemcpy(d_vecs_, host_vecs,
            (size_t)n_ * dim_ * sizeof(float), cudaMemcpyHostToDevice));
        std::copy(host_row_ids, host_row_ids + n_, row_ids_.begin());
    }

    struct Result { uint64_t row_id; float distance; };

    // Run GPU batch distance + host-side top-K selection.
    std::vector<Result> search(const float* query, int top_k, cudaStream_t stream = nullptr) {
        // Copy query to GPU
        AILAKE_CUDA_CHECK(cudaMemcpyAsync(d_query_, query,
            dim_ * sizeof(float), cudaMemcpyHostToDevice, stream));

        // Launch distance kernel
        int block = 256;
        int grid  = (n_ + block - 1) / block;
        switch (metric_) {
        case 1: // Euclidean
            euclidean_distance_kernel<<<grid, block, 0, stream>>>(
                d_query_, d_vecs_, d_distances_, n_, dim_);
            break;
        case 2: // DotProduct
            dot_distance_kernel<<<grid, block, 0, stream>>>(
                d_query_, d_vecs_, d_distances_, n_, dim_);
            break;
        default: // Cosine
            cosine_distance_kernel<<<grid, block, 0, stream>>>(
                d_query_, d_vecs_, d_distances_, n_, dim_);
        }
        AILAKE_CUDA_CHECK(cudaGetLastError());

        // Copy distances back to host
        std::vector<float> h_dists(n_);
        AILAKE_CUDA_CHECK(cudaMemcpyAsync(h_dists.data(), d_distances_,
            n_ * sizeof(float), cudaMemcpyDeviceToHost, stream));
        AILAKE_CUDA_CHECK(cudaStreamSynchronize(stream));

        // Host top-K selection
        std::vector<int> idx(n_);
        std::iota(idx.begin(), idx.end(), 0);
        int k = std::min(top_k, n_);
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
            [&](int a, int b){ return h_dists[a] < h_dists[b]; });

        std::vector<Result> out(k);
        for (int i = 0; i < k; ++i)
            out[i] = {row_ids_[idx[i]], h_dists[idx[i]]};
        return out;
    }

private:
    int dim_, n_, metric_;
    float*    d_vecs_      = nullptr;
    float*    d_query_     = nullptr;
    float*    d_distances_ = nullptr;
    std::vector<uint64_t> row_ids_;
};

// ---------------------------------------------------------------------------
// GpuIvfPqContext — GPU-accelerated IVF-PQ search.
// Uploads codebook LUT + all inverted lists to GPU.
// ---------------------------------------------------------------------------
class GpuIvfPqContext {
public:
    GpuIvfPqContext() = default;
    ~GpuIvfPqContext() {
        for (auto p : d_codes_) cudaFree(p);
        for (auto p : d_dists_) cudaFree(p);
        cudaFree(d_lut_);
    }

    // Call once after deserialize_ivfpq().
    void upload(const std::vector<std::vector<uint8_t>>& inv_codes,
                const std::vector<std::vector<uint64_t>>& inv_row_ids,
                int m, int k) {
        m_ = m; k_ = k;
        n_cells_ = (int)inv_codes.size();
        inv_row_ids_ = inv_row_ids;

        d_codes_.resize(n_cells_, nullptr);
        d_dists_.resize(n_cells_, nullptr);

        for (int i = 0; i < n_cells_; ++i) {
            size_t sz = inv_codes[i].size();
            cell_sizes_.push_back((int)(sz / m));
            if (sz == 0) continue;
            AILAKE_CUDA_CHECK(cudaMalloc(&d_codes_[i], sz));
            AILAKE_CUDA_CHECK(cudaMemcpy(d_codes_[i], inv_codes[i].data(), sz,
                                         cudaMemcpyHostToDevice));
            AILAKE_CUDA_CHECK(cudaMalloc(&d_dists_[i], cell_sizes_.back() * sizeof(float)));
        }
        AILAKE_CUDA_CHECK(cudaMalloc(&d_lut_, (size_t)m * k * sizeof(float)));
    }

    struct Result { uint64_t row_id; float distance; };

    // Search: precompute LUT on host (cheap), run ADC on GPU per probed cell.
    std::vector<Result> search(const float* lut_host, int top_k, int nprobe,
                               cudaStream_t stream = nullptr) {
        // Upload LUT
        AILAKE_CUDA_CHECK(cudaMemcpyAsync(d_lut_, lut_host,
            (size_t)m_ * k_ * sizeof(float), cudaMemcpyHostToDevice, stream));

        // Probe cells
        std::vector<std::pair<float,uint64_t>> hits;
        hits.reserve((size_t)top_k);

        for (int p = 0; p < nprobe && p < n_cells_; ++p) {
            int n = cell_sizes_[p];
            if (n == 0 || !d_codes_[p]) continue;

            int block = 256;
            int grid  = (n + block - 1) / block;

            // Use shared-memory ADC when LUT fits (m*k ≤ 4096 floats = 16 KB)
            size_t smem = (size_t)m_ * k_ * sizeof(float);
            if (smem <= 16 * 1024) {
                ivfpq_adc_smem_kernel<<<grid, block, smem, stream>>>(
                    d_lut_, d_codes_[p], d_dists_[p], n, m_, k_);
            } else {
                ivfpq_adc_kernel<<<grid, block, 0, stream>>>(
                    d_lut_, d_codes_[p], d_dists_[p], n, m_, k_);
            }
            AILAKE_CUDA_CHECK(cudaGetLastError());

            // Copy distances back
            std::vector<float> h_d(n);
            AILAKE_CUDA_CHECK(cudaMemcpyAsync(h_d.data(), d_dists_[p],
                n * sizeof(float), cudaMemcpyDeviceToHost, stream));
            AILAKE_CUDA_CHECK(cudaStreamSynchronize(stream));

            const auto& rids = inv_row_ids_[p];
            for (int r = 0; r < n; ++r)
                hits.push_back({h_d[r], rids[r]});
        }

        // Top-K
        std::partial_sort(hits.begin(),
            hits.begin() + std::min((int)hits.size(), top_k),
            hits.end());
        if ((int)hits.size() > top_k) hits.resize(top_k);

        std::vector<Result> out;
        out.reserve(hits.size());
        for (auto [d, rid] : hits)
            out.push_back({rid, std::sqrt(d)});
        return out;
    }

private:
    int m_ = 0, k_ = 0, n_cells_ = 0;
    std::vector<int>      cell_sizes_;
    std::vector<uint8_t*> d_codes_;
    std::vector<float*>   d_dists_;
    std::vector<std::vector<uint64_t>> inv_row_ids_;
    float* d_lut_ = nullptr;
};

} // namespace cuda
} // namespace ailake
