use anyhow::Result;
use std::marker::PhantomData;
use typenum::marker_traits::Unsigned;
use typenum::U2;

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
/// U is the default tree arity (U2 = binary)
pub struct Proof<T: Eq + Clone + AsRef<[u8]>, BaseTreeArity: Unsigned = U2> {
    // Optional proofs at immediate lower level from current.  Should
    // be None at the base layer.
    pub sub_tree_proof: Option<Box<Proof<T, BaseTreeArity>>>,

    top_layer_nodes: usize,      // arity of top layer
    sub_tree_layer_nodes: usize, // arity of sub-tree layer

    lemma: Vec<T>,
    path: Vec<usize>, // branch index

    _u: PhantomData<BaseTreeArity>, // number of branches per node
}

impl<T: Eq + Clone + AsRef<[u8]>, BaseTreeArity: Unsigned> Proof<T, BaseTreeArity> {
    /// Creates new MT inclusion proof
    pub fn new<TopLayerArity: Unsigned, SubTreeArity: Unsigned>(
        sub_tree_proof: Option<Box<Proof<T, BaseTreeArity>>>,
        lemma: Vec<T>,
        path: Vec<usize>,
    ) -> Result<Proof<T, BaseTreeArity>> {
        if TopLayerArity::to_usize() == 0 && SubTreeArity::to_usize() == 0 {
            ensure!(lemma.len() > 2, "Invalid lemma length (short)");
            ensure!(
                lemma.len()
                    == get_merkle_proof_lemma_len(path.len() + 1, BaseTreeArity::to_usize()),
                "Invalid lemma length"
            );
        }

        Ok(Proof {
            sub_tree_proof,
            top_layer_nodes: TopLayerArity::to_usize(),
            sub_tree_layer_nodes: SubTreeArity::to_usize(),
            lemma,
            path,

            _u: PhantomData,
        })
    }

    /// Return proof target leaf
    pub fn item(&self) -> T {
        self.lemma
            .first()
            .expect("[item] requested element is empty")
            .clone()
    }

    /// Return sub tree root
    pub fn sub_tree_root(&self) -> T {
        assert!(self.sub_tree_layer_nodes > 0 && self.sub_tree_proof.is_some());
        // unwrap is safe as we checked sub_tree_proof to be initialised
        self.sub_tree_proof.as_ref().unwrap().root()
    }

    /// Return tree root
    pub fn root(&self) -> T {
        self.lemma
            .last()
            .expect("[root] requested element is empty")
            .clone()
    }

    /// Validates sub-tree proofs with the specified arity.
    fn validate_sub_tree_proof<A: Algorithm<T>>(&self, arity: usize) -> Result<bool> {
        // Ensure that the sub_tree validates to the root of that
        // sub_tree.
        let valid = self
            .sub_tree_proof
            .as_ref()
            .expect("[validate_sub_tree_proof] couldn't get sub_tree_proof value")
            .validate::<A>()?;
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

        let branches = BaseTreeArity::to_usize();
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
    use crate::test_legacy::{get_vec_tree_from_slice, Item, Sha256Hasher, XOR128};
    use typenum::{Unsigned, U0, U1, U2, U3, U4, U5, U8};

    // Break one element inside the proof's top layer (if available).
    // Otherwise, break the sub-proof.
    fn modify_proof<
        E: Element,
        A: Algorithm<E>,
        BaseTreeArity: Unsigned,
        SubTreeArity: Unsigned,
        TopTreeArity: Unsigned,
    >(
        proof: &mut Proof<E, BaseTreeArity>,
    ) {
        use rand::prelude::*;

        if TopTreeArity::to_usize() > 0 {
            assert!(proof.sub_tree_proof.is_some());
            assert!(proof
                .sub_tree_proof
                .as_ref()
                .unwrap() // safe
                .sub_tree_proof
                .is_some());
        } else if SubTreeArity::to_usize() > 0 {
            assert!(proof.sub_tree_proof.is_some());
        }

        let mut hasher_alg = A::default();
        let mut tmp = vec![0u8; E::byte_len()];

        if TopTreeArity::to_usize() > 0 || SubTreeArity::to_usize() > 0 {
            // unwrap is safe as we checked sub_tree_proof to be initialised
            let i = random::<usize>() % proof.sub_tree_proof.as_ref().unwrap().lemma().len();
            let j = random::<usize>();

            j.hash(&mut hasher_alg);

            // Break random sub-tree proof element

            // unwrap is safe as we checked sub_tree_proof to be initialised
            proof.sub_tree_proof.as_ref().unwrap().lemma()[i].copy_to_slice(&mut tmp);
            tmp.hash(&mut hasher_alg);
            // unwrap is safe as we checked sub_tree_proof to be initialised
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
            BaseTreeArity: Unsigned,
            SubTreeArity: Unsigned,
            TopTreeArity: Unsigned,
        >() {
            let leafs = 32768;
            let tree = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);

            for i in 0..tree.leafs() {
                let mut p = tree
                    .gen_proof(i)
                    .expect("[test_proofs] failed to generate Merkle proof");
                assert!(p.validate::<A>().expect("[test_proofs] failed to validate"));

                // Break the proof here and assert negative validation.
                modify_proof::<E, A, BaseTreeArity, SubTreeArity, TopTreeArity>(&mut p);
                assert!(!p.validate::<A>().expect("[test_proofs] failed to validate"));
            }
        }

        run_test::<Item, XOR128, U2, U0, U0>();
        run_test::<Item, Sha256Hasher, U2, U0, U0>();
    }

    #[test]
    fn test_compound_quad_broken_proofs() {
        fn run_test<
            E: Element,
            A: Algorithm<E>,
            BaseTreeArity: Unsigned,
            SubTreeArity: Unsigned,
            TopTreeArity: Unsigned,
        >() {
            let leafs = 16384;
            let mt1 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let mt2 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let mt3 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);

            let tree: MerkleTree<E, A, VecStore<E>, BaseTreeArity, SubTreeArity> =
                MerkleTree::from_trees(vec![mt1, mt2, mt3])
                    .expect("[test_compound_quad_broken_proofs] Failed to build compound tree");

            for i in 0..tree.leafs() {
                let mut p = tree
                    .gen_proof(i)
                    .expect("[test_compound_quad_broken_proofs] failed to generate Merkle proof");
                assert!(p
                    .validate::<A>()
                    .expect("[test_compound_quad_broken_proofs] failed to validate"));

                modify_proof::<E, A, BaseTreeArity, SubTreeArity, TopTreeArity>(&mut p);
                assert!(!p
                    .validate::<A>()
                    .expect("[test_compound_quad_broken_proofs] failed to validate"));
            }
        }
        run_test::<Item, XOR128, U4, U3, U0>();
        run_test::<Item, Sha256Hasher, U4, U3, U0>();
    }

    #[test]
    fn test_compound_single_octree_broken_proofs() {
        fn run_test<
            E: Element,
            A: Algorithm<E>,
            BaseTreeArity: Unsigned,
            SubTreeArity: Unsigned,
            TopTreeArity: Unsigned,
        >() {
            let leafs = 32768;
            let mt1 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);

            let tree: MerkleTree<E, A, VecStore<E>, BaseTreeArity, SubTreeArity> =
                MerkleTree::from_trees(vec![mt1]).expect(
                    "[test_compound_single_octree_broken_proofs] Failed to build compound tree",
                );

            for i in 0..tree.leafs() {
                let mut p = tree.gen_proof(i).expect(
                    "[test_compound_single_octree_broken_proofs] failed to generate Merkle proof",
                );
                assert!(p
                    .validate::<A>()
                    .expect("[test_compound_single_octree_broken_proofs] failed to validate"));

                modify_proof::<E, A, BaseTreeArity, SubTreeArity, TopTreeArity>(&mut p);
                assert!(!p
                    .validate::<A>()
                    .expect("[test_compound_single_octree_broken_proofs] failed to validate"));
            }
        }
        run_test::<Item, XOR128, U8, U1, U0>();
        run_test::<Item, Sha256Hasher, U8, U1, U0>();
    }

    #[test]
    #[ignore]
    fn test_compound_octree_broken_proofs() {
        fn run_test<
            E: Element,
            A: Algorithm<E>,
            BaseTreeArity: Unsigned,
            SubTreeArity: Unsigned,
            TopTreeArity: Unsigned,
        >() {
            let leafs = 32768;
            let mt1 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let mt2 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let mt3 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let mt4 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);

            let tree: MerkleTree<E, A, VecStore<E>, BaseTreeArity, SubTreeArity> =
                MerkleTree::from_trees(vec![mt1, mt2, mt3, mt4])
                    .expect("[test_compound_octree_broken_proofs] Failed to build compound tree");

            for i in 0..tree.leafs() {
                let mut p = tree
                    .gen_proof(i)
                    .expect("[test_compound_octree_broken_proofs] failed to generate Merkle proof");
                assert!(p
                    .validate::<A>()
                    .expect("[test_compound_octree_broken_proofs] failed to validate"));

                modify_proof::<E, A, BaseTreeArity, SubTreeArity, TopTreeArity>(&mut p);
                assert!(!p
                    .validate::<A>()
                    .expect("[test_compound_octree_broken_proofs] failed to validate"));
            }
        }
        run_test::<Item, XOR128, U8, U4, U0>();
        run_test::<Item, Sha256Hasher, U8, U4, U0>();
    }

    #[test]
    fn test_compound_compound_quad_broken_proofs() {
        fn run_test<
            E: Element,
            A: Algorithm<E>,
            BaseTreeArity: Unsigned,
            SubTreeArity: Unsigned,
            TopTreeArity: Unsigned,
        >() {
            let leafs = 16384;

            let mt1 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let mt2 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let mt3 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let cmt1: MerkleTree<E, A, VecStore<E>, BaseTreeArity, SubTreeArity> =
                MerkleTree::from_trees(vec![mt1, mt2, mt3])
                    .expect("[test_compound_compound_quad_broken_proofs] failed to build compound merkle tree");

            let mt4 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let mt5 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let mt6 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let cmt2: MerkleTree<E, A, VecStore<E>, BaseTreeArity, SubTreeArity> =
                MerkleTree::from_trees(vec![mt4, mt5, mt6])
                    .expect("[test_compound_compound_quad_broken_proofs] failed to build compound merkle tree");

            let mt7 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let mt8 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let mt9 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let cmt3: MerkleTree<E, A, VecStore<E>, BaseTreeArity, SubTreeArity> =
                MerkleTree::from_trees(vec![mt7, mt8, mt9])
                    .expect("[test_compound_compound_quad_broken_proofs] failed to build compound merkle tree");

            let tree: MerkleTree<E, A, VecStore<E>, BaseTreeArity, SubTreeArity, TopTreeArity> =
                MerkleTree::from_sub_trees(vec![cmt1, cmt2, cmt3])
                    .expect("[test_compound_compound_quad_broken_proofs] Failed to build compound-compound tree");

            for i in 0..tree.leafs() {
                let mut p = tree.gen_proof(i).expect("failed to generate Merkle proof");
                assert!(p
                    .validate::<A>()
                    .expect("[test_compound_compound_quad_broken_proofs] failed to validate"));

                modify_proof::<E, A, BaseTreeArity, SubTreeArity, TopTreeArity>(&mut p);
                assert!(!p
                    .validate::<A>()
                    .expect("[test_compound_compound_quad_broken_proofs] failed to validate"));
            }
        }

        run_test::<Item, XOR128, U4, U3, U3>();
        run_test::<Item, Sha256Hasher, U4, U3, U3>();
    }

    #[test]
    #[ignore]
    fn test_compound_compound_single_quad_broken_proofs() {
        fn run_test<
            E: Element,
            A: Algorithm<E>,
            BaseTreeArity: Unsigned,
            SubTreeArity: Unsigned,
            TopTreeArity: Unsigned,
        >() {
            let leafs = 16384;

            let mt1 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let mt2 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let mt3 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let cmt1: MerkleTree<E, A, VecStore<E>, BaseTreeArity, SubTreeArity> =
                MerkleTree::from_trees(vec![mt1, mt2, mt3])
                    .expect("[test_compound_compound_single_quad_broken_proofs] failed to build compound merkle tree");

            let tree: MerkleTree<E, A, VecStore<E>, BaseTreeArity, SubTreeArity, TopTreeArity> =
                MerkleTree::from_sub_trees(vec![cmt1])
                    .expect("[test_compound_compound_single_quad_broken_proofs] Failed to build compound-compound tree");

            for i in 0..tree.leafs() {
                let mut p = tree.gen_proof(i).expect("[test_compound_compound_single_quad_broken_proofs] failed to generate Merkle proof");
                assert!(p.validate::<A>().expect(
                    "[test_compound_compound_single_quad_broken_proofs] failed to validate"
                ));

                // TODO investigate why SubTree and TopTree are substituted (in origin test)
                modify_proof::<E, A, BaseTreeArity, TopTreeArity, SubTreeArity>(&mut p);
                assert!(!p.validate::<A>().expect(
                    "[test_compound_compound_single_quad_broken_proofs] failed to validate"
                ));
            }
        }
        run_test::<Item, XOR128, U4, U3, U1>();
        run_test::<Item, Sha256Hasher, U4, U3, U1>();
    }

    #[test]
    #[ignore]
    fn test_compound_compound_octree_broken_proofs() {
        fn run_test<
            E: Element,
            A: Algorithm<E>,
            BaseTreeArity: Unsigned,
            SubTreeArity: Unsigned,
            TopTreeArity: Unsigned,
        >() {
            let leafs = 32768;

            let mt1 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let mt2 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let mt3 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let mt4 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let cmt1: MerkleTree<E, A, VecStore<E>, BaseTreeArity, SubTreeArity> =
                MerkleTree::from_trees(vec![mt1, mt2, mt3, mt4]).expect(
                    "[test_compound_compound_octree_broken_proofs] Failed to build compound tree",
                );

            let mt5 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let mt6 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let mt7 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let mt8 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let cmt2: MerkleTree<E, A, VecStore<E>, BaseTreeArity, SubTreeArity> =
                MerkleTree::from_trees(vec![mt5, mt6, mt7, mt8]).expect(
                    "[test_compound_compound_octree_broken_proofs] Failed to build compound tree",
                );

            let mt9 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let mt10 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let mt11 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let mt12 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let cmt3: MerkleTree<E, A, VecStore<E>, BaseTreeArity, SubTreeArity> =
                MerkleTree::from_trees(vec![mt9, mt10, mt11, mt12]).expect(
                    "[test_compound_compound_octree_broken_proofs] Failed to build compound tree",
                );

            let mt13 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let mt14 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let mt15 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let mt16 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let cmt4: MerkleTree<E, A, VecStore<E>, BaseTreeArity, SubTreeArity> =
                MerkleTree::from_trees(vec![mt13, mt14, mt15, mt16]).expect(
                    "[test_compound_compound_octree_broken_proofs] Failed to build compound tree",
                );

            let mt17 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let mt18 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let mt19 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let mt20 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let cmt5: MerkleTree<E, A, VecStore<E>, BaseTreeArity, SubTreeArity> =
                MerkleTree::from_trees(vec![mt17, mt18, mt19, mt20]).expect(
                    "[test_compound_compound_octree_broken_proofs] Failed to build compound tree",
                );

            let tree: MerkleTree<E, A, VecStore<E>, BaseTreeArity, SubTreeArity, TopTreeArity> =
                MerkleTree::from_sub_trees(vec![cmt1, cmt2, cmt3, cmt4, cmt5])
                    .expect("[test_compound_compound_octree_broken_proofs] Failed to build compound-compound tree");

            for i in 0..tree.leafs() {
                let mut p = tree.gen_proof(i).expect(
                    "[test_compound_compound_octree_broken_proofs] failed to generate Merkle proof",
                );
                assert!(p
                    .validate::<A>()
                    .expect("[test_compound_compound_octree_broken_proofs] failed to validate"));

                // TODO investigate why SubTree and TopTree are substituted (in origin test)
                modify_proof::<E, A, BaseTreeArity, TopTreeArity, SubTreeArity>(&mut p);
                assert!(!p
                    .validate::<A>()
                    .expect("[test_compound_compound_octree_broken_proofs] failed to validate"));
            }
        }
        run_test::<Item, XOR128, U8, U4, U5>();
        run_test::<Item, Sha256Hasher, U8, U4, U5>();
    }

    #[test]
    #[ignore]
    fn test_compound_compound_single_octree_broken_proofs() {
        fn run_test<
            E: Element,
            A: Algorithm<E>,
            BaseTreeArity: Unsigned,
            SubTreeArity: Unsigned,
            TopTreeArity: Unsigned,
        >() {
            let leafs = 32768;

            let mt1 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let mt2 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let mt3 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let mt4 = get_vec_tree_from_slice::<E, A, BaseTreeArity>(leafs);
            let cmt1: MerkleTree<E, A, VecStore<E>, BaseTreeArity, SubTreeArity> =
                MerkleTree::from_trees(vec![mt1, mt2, mt3, mt4])
                    .expect("[test_compound_compound_single_octree_broken_proofs] Failed to build compound tree");

            let tree: MerkleTree<E, A, VecStore<E>, BaseTreeArity, SubTreeArity, TopTreeArity> =
                MerkleTree::from_sub_trees(vec![cmt1]).expect("[test_compound_compound_single_octree_broken_proofs] Failed to build ccompound tree");

            for i in 0..tree.leafs() {
                let mut p = tree.gen_proof(i).expect("[test_compound_compound_single_octree_broken_proofs] failed to generate Merkle proof");
                assert!(p.validate::<A>().expect(
                    "[test_compound_compound_single_octree_broken_proofs] failed to validate"
                ));

                // TODO investigate why SubTree and TopTree are substituted (in origin test)
                modify_proof::<E, A, BaseTreeArity, TopTreeArity, SubTreeArity>(&mut p);
                assert!(!p.validate::<A>().expect(
                    "[test_compound_compound_single_octree_broken_proofs] failed to validate"
                ));
            }
        }

        run_test::<Item, XOR128, U8, U4, U1>();
        run_test::<Item, Sha256Hasher, U8, U4, U1>();
    }
}
