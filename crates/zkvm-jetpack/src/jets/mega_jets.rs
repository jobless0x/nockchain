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

fn zero_bpoly() -> BPolyVec {
    BPolyVec::from(vec![0u64])
}

fn lagrange_one_bpoly(len: usize) -> BPolyVec {
    BPolyVec::from(vec![1u64; len])
}

pub fn mp_substitute_mega_jet(context: &mut Context, subject: Noun) -> Result {
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

    let mut acc_vec = zero_bpoly();

    let mut p_iter = HoonMapIter::from(p_noun);
    p_iter.try_fold((), |_, n| {
        let [k_noun, v_noun] = n.uncell()?;
        let Ok(k) = BPolySlice::try_from(k_noun) else {
            return jet_err::<()>();
        };
        let Ok(v) = v_noun.as_belt() else {
            return jet_err::<()>();
        };

        if v.0 == 0 {
            return Ok(());
        }

        let poly_len_for_var_com = 4 * height_usize;
        let mut inner_acc_vec = lagrange_one_bpoly(poly_len_for_var_com);

        for i in 0..k.len() {
            let ter = k.0[i];

            let (typ, idx, exp) = brek(ter);

            match typ {
                MegaTyp::Var => {
                    let var_start_idx = idx * poly_len_for_var_com;
                    let var_end_idx = var_start_idx + poly_len_for_var_com;

                    if var_end_idx > trace_evals.len() {
                        return jet_err::<()>();
                    }
                    let var_slice = &trace_evals.0[var_start_idx..var_end_idx];
                    let hadamard_res_len = inner_acc_vec.len().min(var_slice.len());

                    let mut temp_res_vec_belt: Vec<Belt> = vec![Belt::from(0u64); hadamard_res_len];
                    let res_poly_slice = temp_res_vec_belt.as_mut_slice();

                    for _ in 0..exp {
                        let current_inner_acc_slice = &inner_acc_vec.0;
                        let hadamard_result = hadamard_batch(
                            current_inner_acc_slice,
                            var_slice,
                            current_inner_acc_slice.len(),
                            1,
                        );
                        inner_acc_vec = BPolyVec::from(hadamard_result);
                    }
                }
                MegaTyp::Rnd => {
                    let rnd_noun = chal_map_opt
                        .as_ref()
                        .and_then(|m| m.get(stack, D(idx as u64)))
                        .ok_or_else(|| jet_err::<()>().unwrap_err())?;
                    let Ok(rnd) = rnd_noun.as_belt() else {
                        return jet_err::<()>();
                    };

                    let pow_rnd = bpow(rnd.0, exp);
                    let scalar = pow_rnd;
                    let num_polys = 1;
                    let poly_len = inner_acc_vec.0.len();
                    let scalars_vec = vec![scalar; num_polys];
                    let scalar_result = scalar_batch(&inner_acc_vec.0, &scalars_vec, poly_len, num_polys);
                    inner_acc_vec = BPolyVec::from(scalar_result);
                }
                MegaTyp::Dyn => {
                    if idx >= dyns.len() {
                        return jet_err::<()>();
                    }
                    let dyn_val = dyns.0[idx];

                    let pow_dyn = bpow(dyn_val.0, exp);
                    let scalar = pow_dyn;
                    let num_polys = 1;
                    let poly_len = inner_acc_vec.0.len();
                    let scalars_vec = vec![scalar; num_polys];
                    let scalar_result = scalar_batch(&inner_acc_vec.0, &scalars_vec, poly_len, num_polys);
                    inner_acc_vec = BPolyVec::from(scalar_result);
                }
                MegaTyp::Con => {}
                MegaTyp::Com => {
                    let com_noun = com_map_opt
                        .as_ref()
                        .and_then(|m| m.get(stack, D(idx as u64)))
                        .ok_or_else(|| jet_err::<()>().unwrap_err())?;
                    let Ok(com_slice) = BPolySlice::try_from(com_noun) else {
                        return jet_err::<()>();
                    };

                    let hadamard_res_len = inner_acc_vec.len().min(com_slice.len());
                    let mut temp_res_vec_belt: Vec<Belt> = vec![Belt::from(0u64); hadamard_res_len];
                    let res_poly_slice = temp_res_vec_belt.as_mut_slice();

                    for _ in 0..exp {
                        let current_inner_acc_slice = &inner_acc_vec.0;
                        let hadamard_result = hadamard_batch(
                            current_inner_acc_slice,
                            com_slice.0,
                            current_inner_acc_slice.len(),
                            1,
                        );
                        inner_acc_vec = BPolyVec::from(hadamard_result);
                    }
                }
            }
        }

        let scalar = v.0;
        let num_polys = 1;
        let poly_len = inner_acc_vec.0.len();
        let scalars_vec = vec![scalar; num_polys];
        let scalar_result = scalar_batch(&inner_acc_vec.0, &scalars_vec, poly_len, num_polys);
        let scaled_inner_bpolyvec = BPolyVec::from(scalar_result);

        let new_acc_len = acc_vec.len().max(scaled_inner_bpolyvec.len());
        let mut new_acc_vec_belt: Vec<Belt> = vec![Belt::from(0u64); new_acc_len];
        let new_acc_poly_slice = new_acc_vec_belt.as_mut_slice();
        bpadd(&acc_vec.0, &scaled_inner_bpolyvec.0, new_acc_poly_slice);
        acc_vec = BPolyVec::from(
            new_acc_vec_belt
                .into_iter()
                .map(|b| b.0)
                .collect::<Vec<u64>>(),
        );

        Ok(())
    })?;
    let (_final_res_atom, final_res_poly_slice): (IndirectAtom, &mut [Belt]) =
        new_handle_mut_slice(stack, Some(acc_vec.len()));
    final_res_poly_slice.copy_from_slice(&acc_vec.0);

    let res_cell = finalize_poly(stack, Some(final_res_poly_slice.len()), _final_res_atom);

    Ok(res_cell)
}
