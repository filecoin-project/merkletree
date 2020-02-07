use crate::hash::{Algorithm, Hashable};
use crate::merkle::get_merkle_proof_lemma_len;

use anyhow::Result;
use std::marker::PhantomData;
use typenum::marker_traits::Unsigned;
use typenum::U2;

#[cfg(test)]
use crate::test_common::{get_vec_tree_from_slice, Item, XOR128};

/// Merkle tree inclusion proof for data element, for which item = Leaf(Hash(Data Item)).
///
/// Lemma layout:
///
/// ```text
/// [ item h1x h2y h3z ... root ]
/// ```
///
/// Proof validation is positioned hash against lemma path to match root hash.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Proof<T: Eq + Clone + AsRef<[u8]>, U: Unsigned = U2> {
    lemma: Vec<T>,
    path: Vec<usize>,   // branch index
    _u: PhantomData<U>, // number of branches per node
}

impl<T: Eq + Clone + AsRef<[u8]>, U: Unsigned> Proof<T, U> {
    /// Creates new MT inclusion proof
    pub fn new(lemma: Vec<T>, path: Vec<usize>) -> Result<Proof<T, U>> {
        ensure!(lemma.len() > 2, "Invalid lemma length (short)");
        ensure!(
            lemma.len() == get_merkle_proof_lemma_len(path.len() + 1, U::to_usize()),
            "Invalid lemma length"
        );
        Ok(Proof {
            lemma,
            path,
            _u: PhantomData,
        })
    }

    /// Return proof target leaf
    pub fn item(&self) -> T {
        self.lemma.first().unwrap().clone()
    }

    /// Return tree root
    pub fn root(&self) -> T {
        self.lemma.last().unwrap().clone()
    }

    /// Verifies MT inclusion proof
    pub fn validate<A: Algorithm<T>>(&self) -> bool {
        let size = self.lemma.len();
        if size < 2 {
            return false;
        }

        let branches = U::to_usize();
        let mut a = A::default();
        let mut h = self.item();
        let mut path_index = 1;

        for i in (1..size - 1).step_by(branches - 1) {
            a.reset();
            h = {
                let mut nodes: Vec<T> = Vec::with_capacity(branches);
                let mut cur_index = 0;
                for j in 0..branches {
                    if j == self.path[path_index - 1] {
                        nodes.push(h.clone());
                    } else {
                        nodes.push(self.lemma[i + cur_index].clone());
                        cur_index += 1;
                    }
                }

                if cur_index != branches - 1 {
                    return false;
                }

                path_index += 1;
                a.multi_node(&nodes, i - 1)
            };
        }

        h == self.root()
    }

    /// Verifies MT inclusion proof and that leaf_data is the original leaf data for which proof was generated.
    pub fn validate_with_data<A: Algorithm<T>>(&self, leaf_data: &dyn Hashable<A>) -> bool {
        let mut a = A::default();
        leaf_data.hash(&mut a);
        let item = a.hash();
        a.reset();
        let leaf_hash = a.leaf(item);

        (leaf_hash == self.item()) && self.validate::<A>()
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
// Break one element inside the proof.
fn modify_proof<U: Unsigned>(proof: &mut Proof<Item, U>) {
    use rand::prelude::*;

    let i = random::<usize>() % proof.lemma.len();
    let k = random::<usize>();

    let mut a = XOR128::new();
    k.hash(&mut a);

    // Break random element
    proof.lemma[i].hash(&mut a);
    proof.lemma[i] = a.hash();
}

#[test]
fn test_proofs() {
    let leafs = 32768;
    let tree = get_vec_tree_from_slice::<U2>(leafs);

    for i in 0..tree.leafs() {
        let mut p = tree.gen_proof(i).unwrap();
        assert!(p.validate::<XOR128>());

        // Break the proof here and assert negative validation.
        modify_proof(&mut p);
        assert!(!p.validate::<XOR128>());
    }
}
