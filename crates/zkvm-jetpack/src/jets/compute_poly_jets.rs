use nockvm::interpreter::Context;
use nockvm::jets::util::slot;
use nockvm::jets::JetErr;
use nockvm::noun::{Noun, IndirectAtom, D};
use nockvm_macros::tas;
use nockvm::jets::cold::FromNounError;

use crate::form::math::bpoly::*;
use crate::form::poly::*;
use crate::hand::handle::*;
use crate::hand::structs::{HoonList, HoonMap};
use crate::form::math::base::{bneg, binv};
use crate::form::PRIME;

struct ConstraintCounts {
    boundary: usize,
    row: usize,
    transition: usize,
    terminal: usize,
    extra: usize,
}

impl ConstraintCounts {
    fn from_noun(noun: &Noun) -> Result<Self, FromNounError> {
        let cell = noun.as_cell()?;
        let boundary = cell.head().as_atom()?.as_u64()? as usize;
        let cell = cell.tail().as_cell()?;
        let row = cell.head().as_atom()?.as_u64()? as usize;
        let cell = cell.tail().as_cell()?;
        let transition = cell.head().as_atom()?.as_u64()? as usize;
        let cell = cell.tail().as_cell()?;
        let terminal = cell.head().as_atom()?.as_u64()? as usize;
        let extra = cell.tail().as_atom()?.as_u64()? as usize;
        Ok(Self {
            boundary,
            row,
            transition,
            terminal,
            extra,
        })
    }
}

pub fn compute_composition_poly_jet(ctx: &mut Context, subject: Noun) -> Result<Noun, JetErr> {
    let sam = slot(subject, 6)?;

    let omicrons = slot(sam, 2)?;
    let heights = slot(sam, 3)?;
    let trace_polys = slot(sam, 4)?;
    let constraints = slot(sam, 5)?;
    let counts = slot(sam, 6)?;
    let _chals = slot(sam, 7)?;
    let chal_map_noun = slot(sam, 8)?;
    let dyn_map_noun = slot(sam, 9)?;
    let is_extra = slot(sam, 10)?;

    let omicrons = BPolySlice::try_from(omicrons)?;
    let heights: Vec<u64> = HoonList::try_from(heights)?
        .into_iter()
        .map(|n| n.as_atom()?.as_u64())
        .collect::<Result<_, _>>()?;

    let trace_polys: Vec<BPolySlice> = HoonList::try_from(trace_polys)?
        .into_iter()
        .map(BPolySlice::try_from)
        .collect::<Result<_, _>>()?;

    let constraints = HoonMap::try_from(constraints)?;
    let counts = HoonMap::try_from(counts)?;
    let is_extra = is_extra.is_cell();

    let mut acc: Vec<Belt> = vec![Belt::zero()];

    for (i, trace_poly) in trace_polys.iter().enumerate() {
        let cons = HoonMap::try_from(constraints.get(&mut ctx.stack, D(i as u64)).unwrap())?;
        let raw_count = counts.get(&mut ctx.stack, D(i as u64)).unwrap();
        let parsed_counts = ConstraintCounts::from_noun(&raw_count)?;

        let height = heights[i] as usize;
        let omicron = omicrons.0[i];

        let last_row: Vec<Belt> = vec![Belt(bneg(binv(omicron.into()))), Belt(1)];
        let row_zerofier = bpsub_(&bppow(&[Belt(0), Belt(1)], height), &[Belt(1)]);
        let transition_zerofier = bpmul_(&row_zerofier, &last_row);

        let mut apply = |tag: &str, count: usize, zerofier: &[Belt]| {
            if count == 0 {
                return;
            }

            let tag_atom = match tag {
                "boundary" => tas!(b"boundary"),
                "row" => tas!(b"row"),
                "transition" => tas!(b"transitn"), // 8 bytes max
                "terminal" => tas!(b"terminal"),
                "extra" => tas!(b"extra"),
                _ => return,
            };

            let cons_list = HoonList::try_from(cons.get(&mut ctx.stack, D(tag_atom)).unwrap()).unwrap();
            let mut sum = vec![Belt::zero()];
            for item in cons_list.into_iter() {
                let cell = item.as_cell().unwrap();
                let degs_noun = cell.head();
                let mp = cell.tail();
                let degs = HoonList::try_from(degs_noun).unwrap();

                let trace_poly_noun = {
                    let (atom, buf) = new_handle_mut_slice(&mut ctx.stack, Some(trace_poly.0.len()));
                    buf.copy_from_slice(trace_poly.0);
                    atom.as_noun()
                };

                let val = crate::jets::verifier_jets::mpeval::<Belt>(
                    &mut ctx.stack,
                    mp,
                    trace_poly_noun,
                    chal_map_noun,
                    *trace_poly,
                    dyn_map_noun,
                )
                .unwrap();

                for deg in degs {
                    let d = deg.as_atom().unwrap().as_u64().unwrap() as usize;
                    let mut shifted = vec![Belt::zero(); d];
                    shifted.push(Belt::one());
                    let term = bpmul_(&vec![val], &shifted);
                    sum = bpadd_(&sum, &term);
                }
            }

            let mut div_result = vec![Belt::zero(); sum.len()];
            let mut _rem = vec![Belt::zero(); sum.len()];
            bpdvr(&sum, zerofier, &mut div_result, &mut _rem);
            acc = bpadd_(&acc, &div_result);
        };

        apply("boundary", parsed_counts.boundary, &[Belt(PRIME - 1), Belt(1)]);
        apply("row", parsed_counts.row, &row_zerofier);
        apply("transition", parsed_counts.transition, &transition_zerofier);
        apply("terminal", parsed_counts.terminal, &last_row);
        if is_extra && parsed_counts.extra > 0 {
            apply("extra", parsed_counts.extra, &row_zerofier);
        }
    }

    let (res_atom, res_poly): (IndirectAtom, &mut [Belt]) =
        new_handle_mut_slice(&mut ctx.stack, Some(acc.len()));
    res_poly.copy_from_slice(&acc);
    Ok(finalize_poly(&mut ctx.stack, Some(acc.len()), res_atom))
}