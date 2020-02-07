use crate::hash::Algorithm;
use crate::proof::Proof;

use std::marker::PhantomData;
use typenum::marker_traits::Unsigned;

#[cfg(test)]
use crate::compound_merkle::CompoundMerkleTree;
#[cfg(test)]
use crate::hash::Hashable;
#[cfg(test)]
use crate::store::VecStore;
#[cfg(test)]
use crate::test_common::{get_vec_tree_from_slice, Item, XOR128};
#[cfg(test)]
use typenum::{U3, U4, U8};

/// Compound Merkle Proof.
///
/// A compound merkle proof is a type of merkle tree proof.
///
/// Compound merkle tree inclusion proof for data element, for which
/// item = Leaf(Hash(Data Item)).
///
/// Lemma layout:
///
/// ```text
/// [ item h1x h2y h3z ... root ]
/// ```
///
/// Proof validation is positioned hash against lemma path to match root hash.
///
/// Unlike the existing Proof type, this type of proof requires 2
/// proofs, potentially each of different arity.
///
/// Essentially this type of proof consists of an inner proof within a
/// specific sub-tree as well as a proof for the top-layer to the
/// root (a type of small proof also not supported by Proof).
///
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CompoundProof<T: Eq + Clone + AsRef<[u8]>, U: Unsigned, N: Unsigned> {
    sub_tree_proof: Proof<T, U>,
    lemma: Vec<T>,
    path: Vec<usize>, // top layer tree index
    _n: PhantomData<N>,
}

impl<T: Eq + Clone + AsRef<[u8]>, U: Unsigned, N: Unsigned> CompoundProof<T, U, N> {
    /// Creates new compound MT inclusion proof
    pub fn new(
        sub_tree_proof: Proof<T, U>,
        lemma: Vec<T>,
        path: Vec<usize>,
    ) -> CompoundProof<T, U, N> {
        CompoundProof {
            sub_tree_proof,
            lemma,
            path,
            _n: PhantomData,
        }
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
        let top_layer_nodes = N::to_usize();
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
fn modify_proof<U: Unsigned, N: Unsigned>(proof: &mut CompoundProof<Item, U, N>) {
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
fn test_compound_quad_broken_proofs() {
    let leafs = 1024;
    let mt1 = get_vec_tree_from_slice::<U4>(leafs);
    let mt2 = get_vec_tree_from_slice::<U4>(leafs);
    let mt3 = get_vec_tree_from_slice::<U4>(leafs);

    let tree: CompoundMerkleTree<Item, XOR128, VecStore<_>, U4, U3> =
        CompoundMerkleTree::from_trees(vec![mt1, mt2, mt3]).expect("Failed to build compound tree");

    for i in 0..tree.leafs() {
        let mut p = tree.gen_proof(i).unwrap();
        assert!(p.validate::<XOR128>());

        modify_proof(&mut p);
        assert!(!p.validate::<XOR128>());
    }
}

#[test]
fn test_compound_octree_broken_proofs() {
    let leafs = 4096;
    let mt1 = get_vec_tree_from_slice::<U8>(leafs);
    let mt2 = get_vec_tree_from_slice::<U8>(leafs);
    let mt3 = get_vec_tree_from_slice::<U8>(leafs);
    let mt4 = get_vec_tree_from_slice::<U8>(leafs);

    let tree: CompoundMerkleTree<Item, XOR128, VecStore<_>, U8, U4> =
        CompoundMerkleTree::from_trees(vec![mt1, mt2, mt3, mt4])
            .expect("Failed to build compound tree");

    for i in 0..tree.leafs() {
        let mut p = tree.gen_proof(i).unwrap();
        assert!(p.validate::<XOR128>());

        modify_proof(&mut p);
        assert!(!p.validate::<XOR128>());
    }
}
