use nockvm::jets::util::slot;
use nockvm::interpreter::Context;
use nockvm::jets::JetErr;
use nockvm::noun::{Noun, D, T};

use crate::form::mary::MarySlice;
use crate::form::math::bpoly::*;
use crate::form::poly::*;
use crate::hand::handle::*;
use crate::hand::structs::HoonList;

pub fn compute_codeword_commitments_jet(ctx: &mut Context, subject: Noun) -> Result<Noun, JetErr> {
    let sam = slot(subject, 6)?;
    let table_marys_noun = slot(sam, 2)?;
    let fri_domain_len = slot(sam, 6)?.as_atom()?.as_u64()? as usize;
    let _total_cols = slot(sam, 7)?.as_atom()?.as_u64()? as usize;

    let table_list = HoonList::try_from(table_marys_noun)?.into_iter();
    let mut table_polys: Vec<Vec<Belt>> = vec![];

    for item in table_list {
        let mary = MarySlice::try_from(item).map_err(|_| JetErr::from(nockvm::noun::Error::NotCell))?;
        let height = mary.len as usize;

        for col_idx in 0..(mary.step as usize) {
            let col: Vec<Belt> = (0..height)
                .map(|i| {
                    let idx = i * (mary.step as usize) + col_idx;
                    Belt(mary.dat[idx])
                })
                .collect();

            // Replace interpolate_table(&col)? with bp_fft if interpolate_table isn't defined
            let poly = bp_fft(&col)?;
            table_polys.push(poly);
        }
    }

    let mut lde_codewords: Vec<Vec<Belt>> = vec![];
    for poly in table_polys.iter() {
        let mut padded = vec![Belt(0); fri_domain_len];
        for (i, &val) in poly.iter().enumerate() {
            padded[i] = val;
        }
        let root = Belt(fri_domain_len as u64).ordered_root()?;
        let lde = bp_ntt(&padded, &root);
        lde_codewords.push(lde);
    }

    let transposed = transpose_bpolys(&lde_codewords);
    let merk_heap = bp_build_merk_heap(&transposed, ctx)?;

    let poly_list = hoon_list_from_bpolys(&mut ctx.stack, &table_polys)?;
    let codeword_list = hoon_list_from_bpolys(&mut ctx.stack, &lde_codewords)?;
    let result = T(&mut ctx.stack, &[poly_list, codeword_list, merk_heap]);

    Ok(result)
}

fn hoon_list_from_bpolys(ctx: &mut nockvm::mem::NockStack, polys: &[Vec<Belt>]) -> Result<Noun, JetErr> {
    let mut list = D(0);
    for poly in polys.iter().rev() {
        let (atom, buf) = new_handle_mut_slice(ctx, Some(poly.len()));
        buf.copy_from_slice(poly);
        let mary = finalize_poly(ctx, Some(poly.len()), atom);
        list = T(ctx, &[mary, list]);
    }
    Ok(list)
}