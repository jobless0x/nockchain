use std::path::Path;
use cust::prelude::*;

/// Performs elementwise Hadamard (multiply) of batches of polynomials, on GPU if available, else on CPU.
/// Each "polynomial" is a chunk of poly_len in the input slices.
/// Returns the output vector, or CPU fallback if GPU/ptx not found.
pub fn hadamard_batch(
    a: &[u64],
    b: &[u64],
    poly_len: usize,
    num_polys: usize,
) -> Vec<u64> {
    let mut out = vec![0u64; poly_len * num_polys];

    // Check if we're on a platform with CUDA and the .ptx file is present.
    if Path::new("cuda/poly_ops.ptx").exists() {
        // Try GPU, fallback to CPU on error
        match gpu_hadamard_batch(a, b, poly_len, num_polys) {
            Ok(result) => return result,
            Err(e) => {
                eprintln!("GPU batch failed: {:?}, falling back to CPU.", e);
            }
        }
    } else {
        eprintln!("CUDA .ptx file not found, using CPU fallback.");
    }

    // CPU fallback: normal elementwise multiply
    for poly in 0..num_polys {
        let ab = poly * poly_len;
        for i in 0..poly_len {
            out[ab + i] = a[ab + i] * b[ab + i];
        }
    }
    out
}

fn gpu_hadamard_batch(
    a: &[u64],
    b: &[u64],
    poly_len: usize,
    num_polys: usize,
) -> cust::error::CudaResult<Vec<u64>> {
    // 1. Setup CUDA context
    let _ctx = cust::quick_init()?;

    // 2. Load PTX module and kernel
    let ptx = std::fs::read_to_string("cuda/poly_ops.ptx")?;
    let module = Module::from_ptx(ptx, &[])?;
    let stream = Stream::new(StreamFlags::DEFAULT, None)?;

    let func = module.get_function("hadamard_batch")?;

    // 3. Allocate device buffers
    let a_buf = DeviceBuffer::from_slice(a)?;
    let b_buf = DeviceBuffer::from_slice(b)?;
    let mut out_buf = DeviceBuffer::from_slice(&vec![0u64; poly_len * num_polys])?;

    // 4. Launch kernel
    let block_size = 128u32;
    let grid_size = ((num_polys as u32) + block_size - 1) / block_size;

    unsafe {
        launch!(
            func<<<grid_size, block_size, 0, stream>>>(
                a_buf.as_device_ptr(),
                b_buf.as_device_ptr(),
                out_buf.as_device_ptr(),
                poly_len as u64,
                num_polys as u64
            )
        )?;
    }

    stream.synchronize()?;
    let mut out = vec![0u64; poly_len * num_polys];
    out_buf.copy_to(&mut out)?;
    Ok(out)
}