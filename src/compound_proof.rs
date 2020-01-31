use crate::hash::Algorithm;
use crate::proof::Proof;

use std::marker::PhantomData;
use typenum::marker_traits::Unsigned;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CompoundProof<T: Eq + Clone + AsRef<[u8]>, U: Unsigned, N: Unsigned> {
    sub_tree_proof: Proof<T, U>,
    lemma: Vec<Vec<T>>,
    path: Vec<usize>, // top layer tree index
    _n: PhantomData<N>,
}

impl<T: Eq + Clone + AsRef<[u8]>, U: Unsigned, N: Unsigned> CompoundProof<T, U, N> {
    /// Creates new compound MT inclusion proof
    pub fn new(
        sub_tree_proof: Proof<T, U>,
        hash: Vec<Vec<T>>,
        path: Vec<usize>,
    ) -> CompoundProof<T, U, N> {
        CompoundProof {
            sub_tree_proof,
            lemma: hash,
            path,
            _n: PhantomData,
        }
    }

    /// Return tree root
    pub fn sub_tree_root(&self) -> T {
        self.sub_tree_proof.root().clone()
    }

    /// Return tree root
    pub fn root(&self) -> T {
        self.lemma.last().unwrap()[0].clone()
    }

    /// Verifies MT inclusion proof
    pub fn validate<A: Algorithm<T>>(&self) -> bool {
        // Ensure that the sub_tree validates to the root of that
        // sub_tree.
        if !self.sub_tree_proof.validate::<A>() {
            return false;
        }

        let size = self.lemma.len();
        let top_layer_nodes = <N as Unsigned>::to_usize();

        assert_eq!(size, 2); // root + top_layer_hashes
        let mut a = A::default();

        // Assert that the remaining proof matches the tree root (note
        // that Proof::validate cannot handle a proof this small, so
        // this is a version specific for what we know we have in this
        // case).
        a.reset();
        let h = {
            let mut nodes: Vec<T> = Vec::with_capacity(top_layer_nodes);
            let mut cur_index = 0;
            for j in 0..top_layer_nodes {
                if j == self.path[0] {
                    nodes.push(self.sub_tree_root().clone());
                } else {
                    nodes.push(self.lemma[0][cur_index].clone());
                    cur_index += 1;
                }
            }
            assert_eq!(cur_index, top_layer_nodes - 1);
            a.multi_node(nodes, 0)
        };

        h == self.root()
    }
}
