use ibig::UBig;
use nockvm::interpreter::Context;
use nockvm::jets::JetErr;
use nockvm::noun::{Atom, Noun, T};
use tracing::info;

pub fn build_tree_data_jet(ctx: &mut Context, subject: Noun) -> Result<Noun, JetErr> {
    let args = subject
        .cell()
        .ok_or(JetErr::from(nockvm::noun::Error::NotCell))?;
    let t = args.head();
    let alf_atom = args
        .tail()
        .atom()
        .ok_or(JetErr::from(nockvm::noun::Error::NotAtom))?;
    let alf = alf_atom.as_u64()? as u128;

    let leaf = flatten_leaf(&t);
    let dyck = flatten_dyck(&t);
    let len = leaf.len() as u32;

    let size = alf
        .checked_pow(len)
        .ok_or(JetErr::from(nockvm::noun::Error::NotRepresentable))?;

    let compress = |xs: &[u128]| -> u128 {
        xs.iter().rev().fold(0u128, |acc, &x| acc * alf + x)
    };

    let leaf_pelt = compress(&leaf);
    let dyck_pelt = compress(&dyck);

    let size_noun = Atom::from_ubig(&mut ctx.stack, &UBig::from(size)).as_noun();
    let dyck_noun = Atom::from_ubig(&mut ctx.stack, &UBig::from(dyck_pelt)).as_noun();
    let leaf_noun = Atom::from_ubig(&mut ctx.stack, &UBig::from(leaf_pelt)).as_noun();

    let inner = T(&mut ctx.stack, &[leaf_noun, t]);
    let mid = T(&mut ctx.stack, &[dyck_noun, inner]);
    let tree_data = T(&mut ctx.stack, &[size_noun, mid]);

    info!("✅ Jet ++build-tree-data (u128) triggered");
    Ok(tree_data)
}

fn flatten_leaf(t: &Noun) -> Vec<u128> {
    if let Some(n) = t.atom() {
        vec![n.as_u64().unwrap_or(0) as u128]
    } else if let Ok(cell) = t.as_cell() {
        let mut out = flatten_leaf(&cell.head());
        out.extend(flatten_leaf(&cell.tail()));
        out
    } else {
        vec![]
    }
}

fn flatten_dyck(t: &Noun) -> Vec<u128> {
    if t.atom().is_some() {
        vec![]
    } else if let Ok(cell) = t.as_cell() {
        let mut out = vec![1u128];
        out.extend(flatten_dyck(&cell.head()));
        out.push(2u128);
        out.extend(flatten_dyck(&cell.tail()));
        out
    } else {
        vec![]
    }
}