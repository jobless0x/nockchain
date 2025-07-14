// cuda/poly_ops.cu

extern "C" __global__ void hadamard_batch(
    const unsigned long long* a,
    const unsigned long long* b,
    unsigned long long* out,
    unsigned long long poly_len,
    unsigned long long num_polys
) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_polys) {
        const unsigned long long* a_poly = a + idx * poly_len;
        const unsigned long long* b_poly = b + idx * poly_len;
        unsigned long long* out_poly = out + idx * poly_len;
        for (unsigned long long i = 0; i < poly_len; ++i) {
            out_poly[i] = a_poly[i] * b_poly[i]; // For field arithmetic, use mod if needed.
        }
    }
}