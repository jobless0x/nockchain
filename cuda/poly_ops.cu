// cuda/poly_ops.cu
// cuda/poly_ops.cu
#include <stdint.h>

extern "C" __global__ void hadamard_batch(
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* out,
    uint64_t poly_len,
    uint64_t num_polys
) {
    uint64_t total_len = poly_len * num_polys;
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_len) {
        out[idx] = a[idx] * b[idx]; // Use modular arithmetic if needed.
    }
}

extern "C" __global__ void scalar_batch(
    const uint64_t* a,
    const uint64_t* scalars,
    uint64_t* out,
    uint64_t poly_len,
    uint64_t num_polys
) {
    uint64_t total_len = poly_len * num_polys;
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_len) {
        uint64_t poly_idx = idx / poly_len;
        uint64_t scalar = scalars[poly_idx];
        out[idx] = a[idx] * scalar; // Use modular arithmetic if needed.
    }
}