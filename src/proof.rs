use anyhow::Result;

use crate::hash::{Algorithm, Hashable};
use crate::merkle::get_merkle_proof_lemma_len;

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
/// U is the default tree arity (2 = binary)
pub struct Proof<T: Eq + Clone + AsRef<[u8]>, const BASE_TREE_ARITY: usize = 2> {
    // Optional proofs at immediate lower level from current.  Should
    // be None at the base layer.
    pub sub_tree_proof: Option<Box<Proof<T, BASE_TREE_ARITY>>>,

    top_layer_nodes: usize,      // arity of top layer
    sub_tree_layer_nodes: usize, // arity of sub-tree layer

    lemma: Vec<T>,
    path: Vec<usize>, // branch index
}

impl<T: Eq + Clone + AsRef<[u8]>, const BASE_TREE_ARITY: usize> Proof<T, BASE_TREE_ARITY> {
    /// Creates new MT inclusion proof
    pub fn new<const TOP_LAYER_ARITY: usize, const SUB_TREE_ARITY: usize>(
        sub_tree_proof: Option<Box<Proof<T, BASE_TREE_ARITY>>>,
        lemma: Vec<T>,
        path: Vec<usize>,
    ) -> Result<Proof<T, BASE_TREE_ARITY>> {
        if TOP_LAYER_ARITY == 0 && SUB_TREE_ARITY == 0 {
            ensure!(lemma.len() > 2, "Invalid lemma length (short)");
            ensure!(
                lemma.len() == get_merkle_proof_lemma_len(path.len() + 1, BASE_TREE_ARITY),
                "Invalid lemma length"
            );
        }

        Ok(Proof {
            sub_tree_proof,
            top_layer_nodes: TOP_LAYER_ARITY,
            sub_tree_layer_nodes: SUB_TREE_ARITY,
            lemma,
            path,
        })
    }

    /// Return proof target leaf
    pub fn item(&self) -> T {
        self.lemma.first().unwrap().clone()
    }

    /// Return sub tree root
    pub fn sub_tree_root(&self) -> T {
        assert!(self.sub_tree_layer_nodes > 0 && self.sub_tree_proof.is_some());
        self.sub_tree_proof.as_ref().unwrap().root()
    }

    /// Return tree root
    pub fn root(&self) -> T {
        self.lemma.last().unwrap().clone()
    }

    /// Validates sub-tree proofs with the specified arity.
    fn validate_sub_tree_proof<A: Algorithm<T>>(&self, arity: usize) -> Result<bool> {
        // Ensure that the sub_tree validates to the root of that
        // sub_tree.
        let valid = self.sub_tree_proof.as_ref().unwrap().validate::<A>()?;
        if !valid {
            return Ok(valid);
        }

        // Validate top-most/current layer
        //
        // Check that the remaining proof matches the tree root (note
        // that Proof::validate at the base layer cannot handle a
        // proof this small, so this is a version specific for what we
        // know we have in this case).
        let mut a = A::default();
        a.reset();
        let node_count = arity;
        let h = {
            let mut nodes: Vec<T> = Vec::with_capacity(node_count);
            let mut cur_index = 0;
            for j in 0..node_count {
                if j == self.path()[0] {
                    nodes.push(self.sub_tree_root().clone());
                } else {
                    nodes.push(self.lemma()[cur_index].clone());
                    cur_index += 1;
                }
            }

            if cur_index != node_count - 1 {
                return Ok(false);
            }

            a.multi_node(&nodes, 0)
        };

        Ok(h == self.root())
    }

    /// Verifies MT inclusion proof
    pub fn validate<A: Algorithm<T>>(&self) -> Result<bool> {
        if self.top_layer_nodes > 0 {
            // Special Top layer handling here.
            ensure!(
                self.sub_tree_proof.is_some(),
                "Sub tree proof must be present for validation"
            );

            return self.validate_sub_tree_proof::<A>(self.top_layer_nodes);
        }

        if self.sub_tree_layer_nodes > 0 {
            // Sub-tree layer handling here.
            ensure!(
                self.sub_tree_proof.is_some(),
                "Sub tree proof must be present for validation"
            );

            return self.validate_sub_tree_proof::<A>(self.sub_tree_layer_nodes);
        }

        // Base layer handling here.
        ensure!(
            self.sub_tree_layer_nodes == 0,
            "Base layer proof must have 0 as sub-tree layer node count"
        );
        ensure!(
            self.top_layer_nodes == 0,
            "Base layer proof must have 0 as top layer node count"
        );
        ensure!(self.sub_tree_proof.is_none(), "Sub tree proof must be None");

        let size = self.lemma.len();
        if size < 2 {
            return Ok(false);
        }

        let branches = BASE_TREE_ARITY;
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
                    return Ok(false);
                }

                path_index += 1;
                a.multi_node(&nodes, i - 1)
            };
        }

        Ok(h == self.root())
    }

    /// Verifies MT inclusion proof and that leaf_data is the original leaf data for which proof was generated.
    pub fn validate_with_data<A: Algorithm<T>>(&self, leaf_data: &dyn Hashable<A>) -> Result<bool> {
        let mut a = A::default();
        leaf_data.hash(&mut a);
        let item = a.hash();
        a.reset();
        let leaf_hash = a.leaf(item);

        if leaf_hash == self.item() {
            self.validate::<A>()
        } else {
            Ok(false)
        }
    }

    /// Returns the path of this proof.
    pub fn path(&self) -> &Vec<usize> {
        &self.path
    }

    /// Returns the lemma of this proof.
    pub fn lemma(&self) -> &Vec<T> {
        &self.lemma
    }

    /// Returns the lemma of this proof as mutable.
    pub fn lemma_mut(&mut self) -> &mut Vec<T> {
        &mut self.lemma
    }

    pub fn top_layer_nodes(&self) -> usize {
        self.top_layer_nodes
    }

    pub fn sub_layer_nodes(&self) -> usize {
        self.sub_tree_layer_nodes
    }
}

#[cfg(test)]
mod tests {
    use crate::hash::{Algorithm, Hashable};
    use crate::merkle::Element;
    use crate::merkle::MerkleTree;
    use crate::proof::Proof;
    use crate::store::VecStore;
    use crate::test_common::{get_vec_tree_from_slice, Item, Sha256Hasher, XOR128};

    // Break one element inside the proof's top layer (if available).
    // Otherwise, break the sub-proof.
    fn modify_proof<
        E: Element,
        A: Algorithm<E>,
        const BASE_TREE_ARITY: usize,
        const SUB_TREE_ARITY: usize,
        const TOP_TREE_ARITY: usize,
    >(
        proof: &mut Proof<E, BASE_TREE_ARITY>,
    ) {
        use rand::prelude::*;

        if TOP_TREE_ARITY > 0 {
            assert!(proof.sub_tree_proof.is_some());
            assert!(proof
                .sub_tree_proof
                .as_ref()
                .unwrap()
                .sub_tree_proof
                .is_some());
        } else if SUB_TREE_ARITY > 0 {
            assert!(proof.sub_tree_proof.is_some());
        }

        let mut hasher_alg = A::default();
        let mut tmp = vec![0u8; E::byte_len()];

        if TOP_TREE_ARITY > 0 || SUB_TREE_ARITY > 0 {
            let i = random::<usize>() % proof.sub_tree_proof.as_ref().unwrap().lemma().len();
            let j = random::<usize>();

            j.hash(&mut hasher_alg);

            // Break random sub-tree proof element
            proof.sub_tree_proof.as_ref().unwrap().lemma()[i].copy_to_slice(&mut tmp);
            tmp.hash(&mut hasher_alg);
            proof.sub_tree_proof.as_mut().unwrap().lemma_mut()[i] = hasher_alg.hash();
        } else {
            let i = random::<usize>() % proof.lemma.len();
            let k = random::<usize>();

            k.hash(&mut hasher_alg);

            // Break random element
            proof.lemma[i].copy_to_slice(&mut tmp);
            tmp.hash(&mut hasher_alg);
            proof.lemma[i] = hasher_alg.hash();
        }
    }

    #[test]
    fn test_proofs() {
        fn run_test<
            E: Element,
            A: Algorithm<E>,
            const BASE_TREE_ARITY: usize,
            const SUB_TREE_ARITY: usize,
            const TOP_TREE_ARITY: usize,
        >() {
            let leafs = 32768;
            let tree = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);

            for i in 0..tree.leafs() {
                let mut p = tree.gen_proof(i).unwrap();
                assert!(p.validate::<A>().expect("failed to validate"));

                // Break the proof here and assert negative validation.
                modify_proof::<E, A, BASE_TREE_ARITY, SUB_TREE_ARITY, TOP_TREE_ARITY>(&mut p);
                assert!(!p.validate::<A>().expect("failed to validate"));
            }
        }

        run_test::<Item, XOR128, 2, 0, 0>();
        run_test::<Item, Sha256Hasher, 2, 0, 0>();
    }

    #[test]
    fn test_compound_quad_broken_proofs() {
        fn run_test<
            E: Element,
            A: Algorithm<E>,
            const BASE_TREE_ARITY: usize,
            const SUB_TREE_ARITY: usize,
            const TOP_TREE_ARITY: usize,
        >() {
            let leafs = 16384;
            let mt1 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let mt2 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let mt3 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);

            let tree: MerkleTree<E, A, VecStore<E>, BASE_TREE_ARITY, SUB_TREE_ARITY> =
                MerkleTree::from_trees(vec![mt1, mt2, mt3]).expect("Failed to build compound tree");

            for i in 0..tree.leafs() {
                let mut p = tree.gen_proof(i).unwrap();
                assert!(p.validate::<A>().expect("failed to validate"));

                modify_proof::<E, A, BASE_TREE_ARITY, SUB_TREE_ARITY, TOP_TREE_ARITY>(&mut p);
                assert!(!p.validate::<A>().expect("failed to validate"));
            }
        }
        run_test::<Item, XOR128, 4, 3, 0>();
        run_test::<Item, Sha256Hasher, 4, 3, 0>();
    }

    #[test]
    fn test_compound_single_octree_broken_proofs() {
        fn run_test<
            E: Element,
            A: Algorithm<E>,
            const BASE_TREE_ARITY: usize,
            const SUB_TREE_ARITY: usize,
            const TOP_TREE_ARITY: usize,
        >() {
            let leafs = 32768;
            let mt1 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);

            let tree: MerkleTree<E, A, VecStore<E>, BASE_TREE_ARITY, SUB_TREE_ARITY> =
                MerkleTree::from_trees(vec![mt1]).expect("Failed to build compound tree");

            for i in 0..tree.leafs() {
                let mut p = tree.gen_proof(i).unwrap();
                assert!(p.validate::<A>().expect("failed to validate"));

                modify_proof::<E, A, BASE_TREE_ARITY, SUB_TREE_ARITY, TOP_TREE_ARITY>(&mut p);
                assert!(!p.validate::<A>().expect("failed to validate"));
            }
        }
        run_test::<Item, XOR128, 8, 1, 0>();
        run_test::<Item, Sha256Hasher, 8, 1, 0>();
    }

    #[test]
    #[ignore]
    fn test_compound_octree_broken_proofs() {
        fn run_test<
            E: Element,
            A: Algorithm<E>,
            const BASE_TREE_ARITY: usize,
            const SUB_TREE_ARITY: usize,
            const TOP_TREE_ARITY: usize,
        >() {
            let leafs = 32768;
            let mt1 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let mt2 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let mt3 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let mt4 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);

            let tree: MerkleTree<E, A, VecStore<E>, BASE_TREE_ARITY, SUB_TREE_ARITY> =
                MerkleTree::from_trees(vec![mt1, mt2, mt3, mt4])
                    .expect("Failed to build compound tree");

            for i in 0..tree.leafs() {
                let mut p = tree.gen_proof(i).unwrap();
                assert!(p.validate::<A>().expect("failed to validate"));

                modify_proof::<E, A, BASE_TREE_ARITY, SUB_TREE_ARITY, TOP_TREE_ARITY>(&mut p);
                assert!(!p.validate::<A>().expect("failed to validate"));
            }
        }
        run_test::<Item, XOR128, 8, 4, 0>();
        run_test::<Item, Sha256Hasher, 8, 4, 0>();
    }

    #[test]
    fn test_compound_compound_quad_broken_proofs() {
        fn run_test<
            E: Element,
            A: Algorithm<E>,
            const BASE_TREE_ARITY: usize,
            const SUB_TREE_ARITY: usize,
            const TOP_TREE_ARITY: usize,
        >() {
            let leafs = 16384;

            let mt1 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let mt2 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let mt3 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let cmt1: MerkleTree<E, A, VecStore<E>, BASE_TREE_ARITY, SUB_TREE_ARITY> =
                MerkleTree::from_trees(vec![mt1, mt2, mt3])
                    .expect("failed to build compound merkle tree");

            let mt4 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let mt5 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let mt6 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let cmt2: MerkleTree<E, A, VecStore<E>, BASE_TREE_ARITY, SUB_TREE_ARITY> =
                MerkleTree::from_trees(vec![mt4, mt5, mt6])
                    .expect("failed to build compound merkle tree");

            let mt7 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let mt8 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let mt9 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let cmt3: MerkleTree<E, A, VecStore<E>, BASE_TREE_ARITY, SUB_TREE_ARITY> =
                MerkleTree::from_trees(vec![mt7, mt8, mt9])
                    .expect("failed to build compound merkle tree");

            let tree: MerkleTree<
                E,
                A,
                VecStore<E>,
                BASE_TREE_ARITY,
                SUB_TREE_ARITY,
                TOP_TREE_ARITY,
            > = MerkleTree::from_sub_trees(vec![cmt1, cmt2, cmt3])
                .expect("Failed to build compound-compound tree");

            for i in 0..tree.leafs() {
                let mut p = tree.gen_proof(i).unwrap();
                assert!(p.validate::<A>().expect("failed to validate"));

                modify_proof::<E, A, BASE_TREE_ARITY, SUB_TREE_ARITY, TOP_TREE_ARITY>(&mut p);
                assert!(!p.validate::<A>().expect("failed to validate"));
            }
        }

        run_test::<Item, XOR128, 4, 3, 3>();
        run_test::<Item, Sha256Hasher, 4, 3, 3>();
    }

    #[test]
    #[ignore]
    fn test_compound_compound_single_quad_broken_proofs() {
        fn run_test<
            E: Element,
            A: Algorithm<E>,
            const BASE_TREE_ARITY: usize,
            const SUB_TREE_ARITY: usize,
            const TOP_TREE_ARITY: usize,
        >() {
            let leafs = 16384;

            let mt1 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let mt2 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let mt3 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let cmt1: MerkleTree<E, A, VecStore<E>, BASE_TREE_ARITY, SUB_TREE_ARITY> =
                MerkleTree::from_trees(vec![mt1, mt2, mt3])
                    .expect("failed to build compound merkle tree");

            let tree: MerkleTree<
                E,
                A,
                VecStore<E>,
                BASE_TREE_ARITY,
                SUB_TREE_ARITY,
                TOP_TREE_ARITY,
            > = MerkleTree::from_sub_trees(vec![cmt1])
                .expect("Failed to build compound-compound tree");

            for i in 0..tree.leafs() {
                let mut p = tree.gen_proof(i).unwrap();
                assert!(p.validate::<A>().expect("failed to validate"));

                // TODO investigate why SubTree and TopTree are substituted (in origin test)
                modify_proof::<E, A, BASE_TREE_ARITY, TOP_TREE_ARITY, SUB_TREE_ARITY>(&mut p);
                assert!(!p.validate::<A>().expect("failed to validate"));
            }
        }
        run_test::<Item, XOR128, 4, 3, 1>();
        run_test::<Item, Sha256Hasher, 4, 3, 1>();
    }

    #[test]
    #[ignore]
    fn test_compound_compound_octree_broken_proofs() {
        fn run_test<
            E: Element,
            A: Algorithm<E>,
            const BASE_TREE_ARITY: usize,
            const SUB_TREE_ARITY: usize,
            const TOP_TREE_ARITY: usize,
        >() {
            let leafs = 32768;

            let mt1 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let mt2 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let mt3 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let mt4 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let cmt1: MerkleTree<E, A, VecStore<E>, BASE_TREE_ARITY, SUB_TREE_ARITY> =
                MerkleTree::from_trees(vec![mt1, mt2, mt3, mt4])
                    .expect("Failed to build compound tree");

            let mt5 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let mt6 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let mt7 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let mt8 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let cmt2: MerkleTree<E, A, VecStore<E>, BASE_TREE_ARITY, SUB_TREE_ARITY> =
                MerkleTree::from_trees(vec![mt5, mt6, mt7, mt8])
                    .expect("Failed to build compound tree");

            let mt9 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let mt10 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let mt11 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let mt12 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let cmt3: MerkleTree<E, A, VecStore<E>, BASE_TREE_ARITY, SUB_TREE_ARITY> =
                MerkleTree::from_trees(vec![mt9, mt10, mt11, mt12])
                    .expect("Failed to build compound tree");

            let mt13 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let mt14 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let mt15 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let mt16 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let cmt4: MerkleTree<E, A, VecStore<E>, BASE_TREE_ARITY, SUB_TREE_ARITY> =
                MerkleTree::from_trees(vec![mt13, mt14, mt15, mt16])
                    .expect("Failed to build compound tree");

            let mt17 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let mt18 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let mt19 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let mt20 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let cmt5: MerkleTree<E, A, VecStore<E>, BASE_TREE_ARITY, SUB_TREE_ARITY> =
                MerkleTree::from_trees(vec![mt17, mt18, mt19, mt20])
                    .expect("Failed to build compound tree");

            let tree: MerkleTree<
                E,
                A,
                VecStore<E>,
                BASE_TREE_ARITY,
                SUB_TREE_ARITY,
                TOP_TREE_ARITY,
            > = MerkleTree::from_sub_trees(vec![cmt1, cmt2, cmt3, cmt4, cmt5])
                .expect("Failed to build compound-compound tree");

            for i in 0..tree.leafs() {
                let mut p = tree.gen_proof(i).unwrap();
                assert!(p.validate::<A>().expect("failed to validate"));

                // TODO investigate why SubTree and TopTree are substituted (in origin test)
                modify_proof::<E, A, BASE_TREE_ARITY, TOP_TREE_ARITY, SUB_TREE_ARITY>(&mut p);
                assert!(!p.validate::<A>().expect("failed to validate"));
            }
        }
        run_test::<Item, XOR128, 8, 4, 5>();
        run_test::<Item, Sha256Hasher, 8, 4, 5>();
    }

    #[test]
    #[ignore]
    fn test_compound_compound_single_octree_broken_proofs() {
        fn run_test<
            E: Element,
            A: Algorithm<E>,
            const BASE_TREE_ARITY: usize,
            const SUB_TREE_ARITY: usize,
            const TOP_TREE_ARITY: usize,
        >() {
            let leafs = 32768;

            let mt1 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let mt2 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let mt3 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let mt4 = get_vec_tree_from_slice::<E, A, BASE_TREE_ARITY>(leafs);
            let cmt1: MerkleTree<E, A, VecStore<E>, BASE_TREE_ARITY, SUB_TREE_ARITY> =
                MerkleTree::from_trees(vec![mt1, mt2, mt3, mt4])
                    .expect("Failed to build compound tree");

            let tree: MerkleTree<
                E,
                A,
                VecStore<E>,
                BASE_TREE_ARITY,
                SUB_TREE_ARITY,
                TOP_TREE_ARITY,
            > = MerkleTree::from_sub_trees(vec![cmt1]).expect("Failed to build ccompound tree");

            for i in 0..tree.leafs() {
                let mut p = tree.gen_proof(i).unwrap();
                assert!(p.validate::<A>().expect("failed to validate"));

                // TODO investigate why SubTree and TopTree are substituted (in origin test)
                modify_proof::<E, A, BASE_TREE_ARITY, TOP_TREE_ARITY, SUB_TREE_ARITY>(&mut p);
                assert!(!p.validate::<A>().expect("failed to validate"));
            }
        }

        run_test::<Item, XOR128, 8, 4, 1>();
        run_test::<Item, Sha256Hasher, 8, 4, 1>();
    }
}
