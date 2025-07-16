use nockvm::interpreter::Context;
use nockvm::jets::util::slot;
use nockvm::jets::Result;
use nockvm::noun::{IndirectAtom, Noun, D};

use crate::form::math::base::bpow;
use crate::form::math::gpu_poly::hadamard_batch;
use crate::form::math::gpu_poly::scalar_batch;
use crate::form::math::bpoly::*;
use crate::form::mega::{brek, MegaTyp};
use crate::form::poly::*;
use crate::hand::handle::*;
use crate::hand::structs::{HoonMap, HoonMapIter};
use crate::jets::utils::jet_err;
use crate::noun::noun_ext::NounExt;

const BATCH_SIZE: usize = 65536;

fn zero_bpoly() -> BPolyVec {
    BPolyVec::from(vec![0u64])
}

fn lagrange_one_bpoly(len: usize) -> BPolyVec {
    BPolyVec::from(vec![1u64; len])
}

pub fn mp_substitute_mega_jet(context: &mut Context, subject: Noun) -> Result {
    macro_rules! fail_with_log {
        ($msg:expr) => {{
            println!("❌ ERROR: {}", $msg);
            return jet_err::<()>();
        }}
    }
    println!("🚀 mega_jet STARTED");

    let sam = slot(subject, 6)?;
    let stack = &mut context.stack;

    let [p_noun, trace_evals_noun, height_noun, chal_map_noun, dyns_noun, com_map_noun] =
        sam.uncell()?;

    let Ok(trace_evals) = BPolySlice::try_from(trace_evals_noun) else {
        return jet_err::<Noun>();
    };
    let Ok(height_atom) = height_noun.as_atom() else {
        return jet_err::<Noun>();
    };
    let Ok(height) = height_atom.as_u64() else {
        return jet_err::<Noun>();
    };
    let height_usize = height as usize;

    let num_polys = 32_768; // adjust

    let Ok(dyns) = BPolySlice::try_from(dyns_noun) else {
        return jet_err::<Noun>();
    };

    let chal_map_opt: Option<HoonMap> = unsafe {
        if chal_map_noun.raw_equals(&D(0)) {
            None
        } else {
            HoonMap::try_from(chal_map_noun).ok()
        }
    };
    let com_map_opt: Option<HoonMap> = unsafe {
        if com_map_noun.raw_equals(&D(0)) {
            None
        } else {
            HoonMap::try_from(com_map_noun).ok()
        }
    };

    println!("🛠 Initializing acc_vec for p_iter loop");
    let mut acc_vec = zero_bpoly();

    let mut p_iter = HoonMapIter::from(p_noun);
    p_iter.try_fold((), |_, n| {
        println!("🔁 Processing key-value in p_iter...");
        let [k_noun, v_noun] = n.uncell()?;
        let Ok(k) = BPolySlice::try_from(k_noun) else {
            fail_with_log!("Failed to convert key to BPolySlice");
        };
        let Ok(v) = v_noun.as_belt() else {
            fail_with_log!("Failed to convert value to Belt");
        };

        if v.0 == 0 {
            return Ok(());
        }

        let poly_len_for_var_com = 16 * height_usize;
        let mut inner_acc_vec = lagrange_one_bpoly(poly_len_for_var_com);

        for i in 0..k.len() {
            let ter = k.0[i];

            let (typ, idx, exp) = brek(ter);

            match typ {
                MegaTyp::Var => {
                    println!("🧬 Entered MegaTyp::Var idx = {}, exp = {}", idx, exp);
                    let var_start_idx = idx * poly_len_for_var_com;
                    let var_end_idx = var_start_idx + poly_len_for_var_com;

                    if var_end_idx > trace_evals.len() {
                        fail_with_log!("Var slice out of bounds");
                    }
                    println!("📏 trace_evals.len = {}, requested slice = {}..{}", trace_evals.len(), var_start_idx, var_end_idx);
                    let var_slice = &trace_evals.0[var_start_idx..var_end_idx];
                    let var_slice_u64: Vec<u64> = var_slice.iter().map(|b| b.0).collect();
                    let hadamard_res_len = inner_acc_vec.len().min(var_slice.len());

                    let mut temp_res_vec_belt: Vec<Belt> = vec![Belt::from(0u64); hadamard_res_len];
                    let res_poly_slice = temp_res_vec_belt.as_mut_slice();

                    for _ in 0..exp {
                        let current_inner_acc_vec = &inner_acc_vec.0;
                        let current_inner_acc_slice_u64: Vec<u64> = current_inner_acc_vec.iter().map(|x| x.0).collect();
                        let batched_input: Vec<u64> = current_inner_acc_slice_u64
                            .iter()
                            .flat_map(|&x| std::iter::repeat(x).take(num_polys))
                            .collect();
                        let batched_vars: Vec<u64> = var_slice_u64
                            .iter()
                            .flat_map(|&x| std::iter::repeat(x).take(num_polys))
                            .collect();
                        println!("🚀 hadamard_batch: {} × {}", batched_input.len(), batched_vars.len());
                        let hadamard_result = hadamard_batch(
                            &batched_input,
                            &batched_vars,
                            current_inner_acc_slice_u64.len(),
                            num_polys,
                        );
                        inner_acc_vec = BPolyVec::from(hadamard_result);
                    }
                }
                MegaTyp::Rnd => {
                    println!("🎲 Entered MegaTyp::Rnd idx = {}, exp = {}", idx, exp);
                    let rnd_noun = chal_map_opt
                        .as_ref()
                        .and_then(|m| m.get(stack, D(idx as u64)))
                        .ok_or_else(|| jet_err::<()>().unwrap_err())?;
                    let Ok(rnd) = rnd_noun.as_belt() else {
                        fail_with_log!("Failed to convert rnd_noun to Belt");
                    };

                    let pow_rnd = bpow(rnd.0, exp);
                    let scalar = pow_rnd;
                    let poly_len = inner_acc_vec.0.len();
                    let scalars_vec_belt = vec![scalar; 1];
                    let scalars_vec_u64: Vec<u64> = scalars_vec_belt.clone();
                    let inner_acc_vec_u64: Vec<u64> = inner_acc_vec.0.iter().map(|x| x.0).collect();
                    let batched_acc: Vec<u64> = inner_acc_vec_u64
                        .iter()
                        .flat_map(|&x| std::iter::repeat(x).take(num_polys))
                        .take(poly_len * num_polys)
                        .collect();
                    let batched_scalars: Vec<u64> = scalars_vec_u64
                        .iter()
                        .flat_map(|&x| std::iter::repeat(x).take(num_polys))
                        .take(num_polys)
                        .collect();
                    if poly_len == 0 || batched_acc.is_empty() || batched_scalars.is_empty() {
                        println!("🚨 Skipping scalar_batch due to invalid inputs. poly_len: {}, batched_acc: {}, batched_scalars: {}", poly_len, batched_acc.len(), batched_scalars.len());
                        fail_with_log!("Invalid inputs for scalar_batch in MegaTyp::Rnd");
                    }
                    println!("📏 scalar_batch poly_len = {}, num_polys (BATCH_SIZE) = {}", poly_len, num_polys);
                    println!("🚀 scalar_batch: {} × {}", batched_acc.len(), batched_scalars.len());
                    println!("🔥 Starting GPU scalar_batch...");
                    let scalar_result = scalar_batch(&batched_acc, &batched_scalars, poly_len, num_polys);
                    println!("✅ scalar_batch returned with {} elements", scalar_result.len());
                    println!("✅ GPU scalar_batch completed, consuming result...");
                    let mut temp_res_vec_belt: Vec<Belt> = scalar_result.iter().map(|&x| Belt::from(x)).collect();
                    let _res_poly_slice = temp_res_vec_belt.as_mut_slice();
                    println!("🧠 Output slice length = {}", _res_poly_slice.len());
                    if !_res_poly_slice.is_empty() {
                        println!("✅ Jet result used: first u64 = {:?}", _res_poly_slice[0]);
                    } else {
                        println!("❌ GPU output slice was empty — skipping jet result usage!");
                    }
                    inner_acc_vec = BPolyVec::from(scalar_result);
                }
                MegaTyp::Dyn => {
                    println!("⚡ Entered MegaTyp::Dyn idx = {}, exp = {}", idx, exp);
                    if idx >= dyns.len() {
                        fail_with_log!("Dyn index out of bounds");
                    }
                    let dyn_val = dyns.0[idx];

                    let pow_dyn = bpow(dyn_val.0, exp);
                    let scalar = pow_dyn;
                    let poly_len = inner_acc_vec.0.len();
                    let scalars_vec_belt = vec![scalar; 1];
                    let scalars_vec_u64: Vec<u64> = scalars_vec_belt.clone();
                    let inner_acc_vec_u64: Vec<u64> = inner_acc_vec.0.iter().map(|x| x.0).collect();
                    let batched_acc: Vec<u64> = inner_acc_vec_u64
                        .iter()
                        .flat_map(|&x| std::iter::repeat(x).take(num_polys))
                        .take(poly_len * num_polys)
                        .collect();
                    let batched_scalars: Vec<u64> = scalars_vec_u64
                        .iter()
                        .flat_map(|&x| std::iter::repeat(x).take(num_polys))
                        .take(num_polys)
                        .collect();
                    if poly_len == 0 || batched_acc.is_empty() || batched_scalars.is_empty() {
                        println!("🚨 Skipping scalar_batch due to invalid inputs. poly_len: {}, batched_acc: {}, batched_scalars: {}", poly_len, batched_acc.len(), batched_scalars.len());
                        fail_with_log!("Invalid inputs for scalar_batch in MegaTyp::Dyn");
                    }
                    println!("📏 scalar_batch poly_len = {}, num_polys (BATCH_SIZE) = {}", poly_len, num_polys);
                    println!("🚀 scalar_batch: {} × {}", batched_acc.len(), batched_scalars.len());
                    println!("🔥 Starting GPU scalar_batch...");
                    let scalar_result = scalar_batch(&batched_acc, &batched_scalars, poly_len, num_polys);
                    println!("✅ scalar_batch returned with {} elements", scalar_result.len());
                    println!("✅ GPU scalar_batch completed, consuming result...");
                    let mut temp_res_vec_belt: Vec<Belt> = scalar_result.iter().map(|&x| Belt::from(x)).collect();
                    let _res_poly_slice = temp_res_vec_belt.as_mut_slice();
                    println!("🧠 Output slice length = {}", _res_poly_slice.len());
                    if !_res_poly_slice.is_empty() {
                        println!("✅ Jet result used: first u64 = {:?}", _res_poly_slice[0]);
                    } else {
                        println!("❌ GPU output slice was empty — skipping jet result usage!");
                    }
                    inner_acc_vec = BPolyVec::from(scalar_result);
                }
                MegaTyp::Con => {
                    println!("📦 MegaTyp::Con (no-op)");
                }
                MegaTyp::Com => {
                    println!("🔗 Entered MegaTyp::Com idx = {}, exp = {}", idx, exp);
                    let com_noun = com_map_opt
                        .as_ref()
                        .and_then(|m| m.get(stack, D(idx as u64)))
                        .ok_or_else(|| jet_err::<()>().unwrap_err())?;
                    let Ok(com_slice) = BPolySlice::try_from(com_noun) else {
                        fail_with_log!("Failed to convert com_noun to BPolySlice");
                    };

                    let com_slice_u64: Vec<u64> = com_slice.0.iter().map(|b| b.0).collect();
                    let hadamard_res_len = inner_acc_vec.len().min(com_slice.len());
                    let mut temp_res_vec_belt: Vec<Belt> = vec![Belt::from(0u64); hadamard_res_len];
                    let res_poly_slice = temp_res_vec_belt.as_mut_slice();

                    for _ in 0..exp {
                        let current_inner_acc_vec = &inner_acc_vec.0;
                        let current_inner_acc_slice_u64: Vec<u64> = current_inner_acc_vec.iter().map(|x| x.0).collect();
                        let batched_input: Vec<u64> = current_inner_acc_slice_u64
                            .iter()
                            .flat_map(|&x| std::iter::repeat(x).take(num_polys))
                            .collect();
                        let batched_vars: Vec<u64> = com_slice_u64
                            .iter()
                            .flat_map(|&x| std::iter::repeat(x).take(num_polys))
                            .collect();
                        println!("🚀 hadamard_batch: {} × {}", batched_input.len(), batched_vars.len());
                        let hadamard_result = hadamard_batch(
                            &batched_input,
                            &batched_vars,
                            current_inner_acc_slice_u64.len(),
                            num_polys,
                        );
                        inner_acc_vec = BPolyVec::from(hadamard_result);
                    }
                }
            }
        }

        let scalar = v.0;
        let poly_len = inner_acc_vec.0.len();
        let scalars_vec_belt = vec![scalar; 1];
        let scalars_vec_u64: Vec<u64> = scalars_vec_belt.clone();
        let inner_acc_vec_u64: Vec<u64> = inner_acc_vec.0.iter().map(|x| x.0).collect();
        let batched_acc: Vec<u64> = inner_acc_vec_u64
            .iter()
            .flat_map(|&x| std::iter::repeat(x).take(num_polys))
            .take(poly_len * num_polys)
            .collect();
        let batched_scalars: Vec<u64> = scalars_vec_u64
            .iter()
            .flat_map(|&x| std::iter::repeat(x).take(num_polys))
            .take(num_polys)
            .collect();
        if poly_len == 0 || batched_acc.is_empty() || batched_scalars.is_empty() {
            println!("🚨 Skipping scalar_batch due to invalid inputs. poly_len: {}, batched_acc: {}, batched_scalars: {}", poly_len, batched_acc.len(), batched_scalars.len());
            fail_with_log!("Invalid inputs for scalar_batch at end of p_iter loop");
        }
        println!("📏 scalar_batch poly_len = {}, num_polys (BATCH_SIZE) = {}", poly_len, num_polys);
        println!("🚀 scalar_batch: {} × {}", batched_acc.len(), batched_scalars.len());
        println!("✅ Launching scalar_batch with poly_len: {}, BATCH_SIZE: {}", poly_len, num_polys);
        let scalar_result = scalar_batch(&batched_acc, &batched_scalars, poly_len, num_polys);
        println!("✅ scalar_batch returned with {} elements", scalar_result.len());
        println!("✅ scalar_batch completed, updating acc_vec...");
        let scaled_inner_bpolyvec = BPolyVec::from(scalar_result);
        println!("✅ acc_vec updated, continuing main loop...");

        let new_acc_len = acc_vec.len().max(scaled_inner_bpolyvec.len());
        let mut new_acc_vec_belt: Vec<Belt> = vec![Belt::from(0u64); new_acc_len];
        let new_acc_poly_slice = new_acc_vec_belt.as_mut_slice();
        println!("➕ Accumulating new result into acc_vec");
        bpadd(&acc_vec.0, &scaled_inner_bpolyvec.0, new_acc_poly_slice);
        acc_vec = BPolyVec::from(
            new_acc_vec_belt
                .into_iter()
                .map(|b| b.0)
                .collect::<Vec<u64>>(),
        );

        Ok(())
    })?;
    println!("🎉 End of p_iter loop, acc_vec.len = {}", acc_vec.len());
    println!("✅ Finished MegaJet GPU proof batch");
    println!("🧠 allocating handle for acc_vec len = {}", acc_vec.len());
    let (_final_res_atom, final_res_poly_slice): (IndirectAtom, &mut [Belt]) =
        new_handle_mut_slice(stack, Some(acc_vec.len()));
    println!("✅ handle allocated, copying...");
    final_res_poly_slice.copy_from_slice(&acc_vec.0);

    let res_cell = finalize_poly(stack, Some(final_res_poly_slice.len()), _final_res_atom);
    println!("🏁 mega_jet FINISHED successfully. Returning result.");

    Ok(res_cell)
}
