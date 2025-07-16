use std::path::Path;
use cust::prelude::*;
use std::time::Instant;

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
        let iterations = std::env::var("GPU_LOOP").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        let mut final_result = vec![0u64; poly_len * num_polys];

        for _ in 0..iterations {
            match gpu_hadamard_batch(a, b, poly_len, num_polys) {
                Ok(result) => final_result = result,
                Err(e) => {
                    eprintln!("GPU batch failed: {:?}, falling back to CPU.", e);
                    break;
                }
            }
        }
        return final_result;
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
    let start = std::time::Instant::now();

    // 1. Setup CUDA context
    let _ctx = cust::quick_init()?;

    // 2. Load PTX module and kernel
    let ptx = match std::fs::read_to_string("cuda/poly_ops.ptx") {
        Ok(ptx) => ptx,
        Err(_e) => return Err(cust::error::CudaError::InvalidValue),
    };
    let module = Module::from_ptx(ptx, &[])?;
    let stream = Stream::new(StreamFlags::DEFAULT, None)?;

    let func = module.get_function("hadamard_batch")?;

    // 3. Allocate device buffers
    let a_buf = DeviceBuffer::from_slice(a)?;
    let b_buf = DeviceBuffer::from_slice(b)?;
    let mut out_buf = DeviceBuffer::from_slice(&vec![0u64; poly_len * num_polys])?;

    // 4. Launch kernel (elementwise parallelism)
    let total_elems = (poly_len * num_polys) as u32;
    println!("📊 hadamard_batch: poly_len = {}, num_polys = {}, total_elems = {}", poly_len, num_polys, total_elems);
    let block_size = 1024u32;
    let grid_size = (total_elems + block_size - 1) / block_size;

    unsafe {
        launch!(
            func<<<grid_size, block_size, 0, stream>>>(
                a_buf.as_device_ptr(),
                b_buf.as_device_ptr(),
                out_buf.as_device_ptr(),
                (poly_len * num_polys) as u64
            )
        )?;
    }

    stream.synchronize()?;
    let mut out = vec![0u64; poly_len * num_polys];
    out_buf.copy_to(&mut out)?;
    println!("⏱️ GPU hadamard_batch completed in {:?}", start.elapsed());
    Ok(out)
}
/// Performs elementwise scalar multiplication of batches of polynomials, on GPU if available, else on CPU.
/// Each "polynomial" is a chunk of poly_len in the input slices, and each has a scalar (one per poly).
pub fn scalar_batch(
    a: &[u64],
    scalars: &[u64],
    poly_len: usize,
    num_polys: usize,
) -> Vec<u64> {
    let mut out = vec![0u64; poly_len * num_polys];

    // Check if we're on a platform with CUDA and the .ptx file is present.
    if Path::new("cuda/poly_ops.ptx").exists() {
        let iterations = std::env::var("GPU_LOOP").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        let mut final_result = vec![0u64; poly_len * num_polys];

        for _ in 0..iterations {
            match gpu_scalar_batch(a, scalars, poly_len, num_polys) {
                Ok(result) => final_result = result,
                Err(e) => {
                    eprintln!("GPU scalar batch failed: {:?}, falling back to CPU.", e);
                    break;
                }
            }
        }
        return final_result;
    } else {
        eprintln!("CUDA .ptx file not found, using CPU fallback.");
    }

    // CPU fallback: normal elementwise scalar multiply
    for poly in 0..num_polys {
        let ab = poly * poly_len;
        let scalar = scalars[poly];
        for i in 0..poly_len {
            out[ab + i] = a[ab + i] * scalar;
        }
    }
    out
}

fn gpu_scalar_batch(
    a: &[u64],
    scalars: &[u64],
    poly_len: usize,
    num_polys: usize,
) -> cust::error::CudaResult<Vec<u64>> {
    let start = std::time::Instant::now();

    let _ctx = cust::quick_init()?;

    let ptx = match std::fs::read_to_string("cuda/poly_ops.ptx") {
        Ok(ptx) => ptx,
        Err(_e) => return Err(cust::error::CudaError::InvalidValue),
    };
    let module = Module::from_ptx(ptx, &[])?;
    let stream = Stream::new(StreamFlags::DEFAULT, None)?;

    let func = module.get_function("scalar_batch")?;

    let a_buf = DeviceBuffer::from_slice(a)?;
    let scalars_buf = DeviceBuffer::from_slice(scalars)?;
    let mut out_buf = DeviceBuffer::from_slice(&vec![0u64; poly_len * num_polys])?;

    let total_elems = (poly_len * num_polys) as u32;
    println!("📊 scalar_batch: poly_len = {}, num_polys = {}, total_elems = {}", poly_len, num_polys, total_elems);
    let block_size = 1024u32;
    let grid_size = (total_elems + block_size - 1) / block_size;

    unsafe {
        launch!(
            func<<<grid_size, block_size, 0, stream>>>(
                a_buf.as_device_ptr(),
                scalars_buf.as_device_ptr(),
                out_buf.as_device_ptr(),
                (poly_len * num_polys) as u64,
                poly_len as u64
            )
        )?;
    }

    stream.synchronize()?;
    let mut out = vec![0u64; poly_len * num_polys];
    out_buf.copy_to(&mut out)?;
    println!("⏱️ GPU scalar_batch completed in {:?}", start.elapsed());
    Ok(out)
}