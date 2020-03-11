use crate::hash::Algorithm;

use anyhow::Result;
use std::marker::PhantomData;
use typenum::marker_traits::Unsigned;

#[cfg(test)]
use crate::ccompound_merkle::CCompoundMerkleTree;
#[cfg(test)]
use crate::compound_merkle::CompoundMerkleTree;
use crate::compound_merkle_proof::CompoundMerkleProof;
#[cfg(test)]
use crate::hash::Hashable;
#[cfg(test)]
use crate::store::VecStore;
#[cfg(test)]
use crate::test_common::{get_vec_tree_from_slice, Item, XOR128};
#[cfg(test)]
use typenum::{U3, U4, U5, U8};

/// CCompound Merkle Proof.
///
/// A compound merkle proof is a type of compound merkle tree proof.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CCompoundMerkleProof<T: Eq + Clone + AsRef<[u8]>, U: Unsigned, N: Unsigned, R: Unsigned>
{
    sub_tree_proof: CompoundMerkleProof<T, U, N>,
    lemma: Vec<T>,    // top layer proof hashes
    path: Vec<usize>, // top layer tree index
    _r: PhantomData<R>,
}

impl<T: Eq + Clone + AsRef<[u8]>, U: Unsigned, N: Unsigned, R: Unsigned>
    CCompoundMerkleProof<T, U, N, R>
{
    /// Creates new compound MT inclusion proof
    pub fn new(
        sub_tree_proof: CompoundMerkleProof<T, U, N>,
        lemma: Vec<T>,
        path: Vec<usize>,
    ) -> Result<CCompoundMerkleProof<T, U, N, R>> {
        ensure!(lemma.len() == R::to_usize(), "Invalid lemma length");
        Ok(CCompoundMerkleProof {
            sub_tree_proof,
            lemma,
            path,
            _r: PhantomData,
        })
    }

    /// Return tree root
    pub fn sub_tree_root(&self) -> T {
        self.sub_tree_proof.root()
    }

    /// Return tree root
    pub fn root(&self) -> T {
        self.lemma.last().unwrap().clone()
    }

    /// Verifies MT inclusion proof
    pub fn validate<A: Algorithm<T>>(&self) -> bool {
        // Ensure that the sub_tree validates to the root of that
        // sub_tree.
        if !self.sub_tree_proof.validate::<A>() {
            return false;
        }

        // Check that size is root + top_layer_nodes - 1.
        let top_layer_nodes = R::to_usize();
        if self.lemma.len() != top_layer_nodes {
            return false;
        }

        // Check that the remaining proof matches the tree root (note
        // that Proof::validate cannot handle a proof this small, so
        // this is a version specific for what we know we have in this
        // case).
        let mut a = A::default();
        a.reset();
        let h = {
            let mut nodes: Vec<T> = Vec::with_capacity(top_layer_nodes);
            let mut cur_index = 0;
            for j in 0..top_layer_nodes {
                if j == self.path[0] {
                    nodes.push(self.sub_tree_root().clone());
                } else {
                    nodes.push(self.lemma[cur_index].clone());
                    cur_index += 1;
                }
            }

            if cur_index != top_layer_nodes - 1 {
                return false;
            }

            a.multi_node(&nodes, 0)
        };

        h == self.root()
    }

    /// Returns the path of this proof.
    pub fn path(&self) -> &Vec<usize> {
        &self.path
    }

    /// Returns the lemma of this proof.
    pub fn lemma(&self) -> &Vec<T> {
        &self.lemma
    }
}

#[cfg(test)]
// Break one element inside the proof's top layer.
fn modify_proof<U: Unsigned, N: Unsigned, R: Unsigned>(
    proof: &mut CCompoundMerkleProof<Item, U, N, R>,
) {
    use rand::prelude::*;

    let i = random::<usize>() % proof.lemma.len();
    let j = random::<usize>();

    let mut a = XOR128::new();
    j.hash(&mut a);

    // Break random element
    proof.lemma[i].hash(&mut a);
    proof.lemma[i] = a.hash();
}

#[test]
fn test_ccompound_quad_broken_proofs() {
    let leafs = 16384;

    let mt1 = get_vec_tree_from_slice::<U4>(leafs);
    let mt2 = get_vec_tree_from_slice::<U4>(leafs);
    let mt3 = get_vec_tree_from_slice::<U4>(leafs);
    let cmt1: CompoundMerkleTree<Item, XOR128, VecStore<_>, U4, U3> =
        CompoundMerkleTree::from_trees(vec![mt1, mt2, mt3])
            .expect("failed to build compound merkle tree");

    let mt4 = get_vec_tree_from_slice::<U4>(leafs);
    let mt5 = get_vec_tree_from_slice::<U4>(leafs);
    let mt6 = get_vec_tree_from_slice::<U4>(leafs);
    let cmt2: CompoundMerkleTree<Item, XOR128, VecStore<_>, U4, U3> =
        CompoundMerkleTree::from_trees(vec![mt4, mt5, mt6])
            .expect("failed to build compound merkle tree");

    let mt7 = get_vec_tree_from_slice::<U4>(leafs);
    let mt8 = get_vec_tree_from_slice::<U4>(leafs);
    let mt9 = get_vec_tree_from_slice::<U4>(leafs);
    let cmt3: CompoundMerkleTree<Item, XOR128, VecStore<_>, U4, U3> =
        CompoundMerkleTree::from_trees(vec![mt7, mt8, mt9])
            .expect("failed to build compound merkle tree");

    let tree: CCompoundMerkleTree<Item, XOR128, VecStore<_>, U4, U3, U3> =
        CCompoundMerkleTree::from_trees(vec![cmt1, cmt2, cmt3])
            .expect("Failed to build ccompound tree");

    for i in 0..tree.leafs() {
        let mut p: CCompoundMerkleProof<Item, U4, U3, U3> = tree.gen_proof(i).unwrap();
        assert!(p.validate::<XOR128>());

        modify_proof(&mut p);
        assert!(!p.validate::<XOR128>());
    }
}

#[test]
fn test_ccompound_octree_broken_proofs() {
    let leafs = 32768;

    let mt1 = get_vec_tree_from_slice::<U8>(leafs);
    let mt2 = get_vec_tree_from_slice::<U8>(leafs);
    let mt3 = get_vec_tree_from_slice::<U8>(leafs);
    let mt4 = get_vec_tree_from_slice::<U8>(leafs);
    let cmt1: CompoundMerkleTree<Item, XOR128, VecStore<_>, U8, U4> =
        CompoundMerkleTree::from_trees(vec![mt1, mt2, mt3, mt4])
            .expect("Failed to build compound tree");

    let mt5 = get_vec_tree_from_slice::<U8>(leafs);
    let mt6 = get_vec_tree_from_slice::<U8>(leafs);
    let mt7 = get_vec_tree_from_slice::<U8>(leafs);
    let mt8 = get_vec_tree_from_slice::<U8>(leafs);
    let cmt2: CompoundMerkleTree<Item, XOR128, VecStore<_>, U8, U4> =
        CompoundMerkleTree::from_trees(vec![mt5, mt6, mt7, mt8])
            .expect("Failed to build compound tree");

    let mt9 = get_vec_tree_from_slice::<U8>(leafs);
    let mt10 = get_vec_tree_from_slice::<U8>(leafs);
    let mt11 = get_vec_tree_from_slice::<U8>(leafs);
    let mt12 = get_vec_tree_from_slice::<U8>(leafs);
    let cmt3: CompoundMerkleTree<Item, XOR128, VecStore<_>, U8, U4> =
        CompoundMerkleTree::from_trees(vec![mt9, mt10, mt11, mt12])
            .expect("Failed to build compound tree");

    let mt13 = get_vec_tree_from_slice::<U8>(leafs);
    let mt14 = get_vec_tree_from_slice::<U8>(leafs);
    let mt15 = get_vec_tree_from_slice::<U8>(leafs);
    let mt16 = get_vec_tree_from_slice::<U8>(leafs);
    let cmt4: CompoundMerkleTree<Item, XOR128, VecStore<_>, U8, U4> =
        CompoundMerkleTree::from_trees(vec![mt13, mt14, mt15, mt16])
            .expect("Failed to build compound tree");

    let mt17 = get_vec_tree_from_slice::<U8>(leafs);
    let mt18 = get_vec_tree_from_slice::<U8>(leafs);
    let mt19 = get_vec_tree_from_slice::<U8>(leafs);
    let mt20 = get_vec_tree_from_slice::<U8>(leafs);
    let cmt5: CompoundMerkleTree<Item, XOR128, VecStore<_>, U8, U4> =
        CompoundMerkleTree::from_trees(vec![mt17, mt18, mt19, mt20])
            .expect("Failed to build compound tree");

    let tree: CCompoundMerkleTree<Item, XOR128, VecStore<_>, U8, U4, U5> =
        CCompoundMerkleTree::from_trees(vec![cmt1, cmt2, cmt3, cmt4, cmt5])
            .expect("Failed to build compound tree");

    for i in 0..tree.leafs() {
        let mut p: CCompoundMerkleProof<Item, U8, U4, U5> = tree.gen_proof(i).unwrap();
        assert!(p.validate::<XOR128>());

        modify_proof(&mut p);
        assert!(!p.validate::<XOR128>());
    }
}
