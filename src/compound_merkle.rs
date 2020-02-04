use std::marker::PhantomData;

use anyhow::{Context, Result};

use crate::compound_proof::CompoundProof;
use crate::hash::Algorithm;
use crate::merkle::{get_merkle_tree_len, Element, MerkleTree};
use crate::proof::Proof;
use crate::store::{Store, StoreConfig};
use typenum::marker_traits::Unsigned;

#[derive(Debug, Clone, Eq, PartialEq, Default)]
pub struct CompoundMerkleTree<T, A, K, B, N>
where
    T: Element,
    A: Algorithm<T>,
    K: Store<T>,
    B: Unsigned, // Branching factor of sub-trees
    N: Unsigned, // Number of nodes at top layer
{
    trees: Vec<MerkleTree<T, A, K, B>>,
    top_layer_nodes: usize,
    len: usize,
    leafs: usize,
    height: usize,
    root: T,

    _n: PhantomData<N>,
}

impl<T: Element, A: Algorithm<T>, K: Store<T>, B: Unsigned, N: Unsigned>
    CompoundMerkleTree<T, A, K, B, N>
{
    /// Creates new compound merkle tree from a vector of merkle trees.
    pub fn new(trees: Vec<MerkleTree<T, A, K, B>>) -> Result<CompoundMerkleTree<T, A, K, B, N>> {
        let top_layer_nodes = <N as Unsigned>::to_usize();
        ensure!(
            trees.len() == top_layer_nodes,
            "Length of trees MUST equal the number of top layer nodes"
        );
        ensure!(
            trees.iter().all(|ref mt| mt.height() == trees[0].height()),
            "All passed in trees must have the same height"
        );
        ensure!(
            trees.iter().all(|ref mt| mt.len() == trees[0].len()),
            "All passed in trees must have the same length"
        );

        // Total number of leafs in the compound tree is the combined leafs total of all subtrees.
        let leafs = trees.iter().fold(0, |leafs, mt| leafs + mt.leafs());
        // Total length of the compound tree is the combined length of all subtrees plus the root.
        let len = trees.iter().fold(0, |len, mt| len + mt.len()) + 1;
        // Total height of the compound tree is the height of any of the sub-trees to top-layer plus root.
        let height = trees[0].height() + 1;
        // Calculate the compound root by hashing the top layer roots together.
        let roots: Vec<T> = trees.iter().map(|x| x.root()).collect();
        let root = A::default().multi_node(&roots, 1);

        Ok(CompoundMerkleTree {
            trees,
            top_layer_nodes,
            len,
            leafs,
            height,
            root,
            _n: PhantomData,
        })
    }

    /// Given an already constructed merkle tree contained as a slice,
    /// add it to this compound merkle tree.
    pub fn new_from_slice(
        tree_data: &[u8],
        leafs: usize,
    ) -> Result<CompoundMerkleTree<T, A, K, B, N>> {
        let mut trees = Vec::with_capacity(1);
        trees.push(MerkleTree::<T, A, K, B>::from_tree_slice(tree_data, leafs)?);

        CompoundMerkleTree::new(trees)
    }

    /// Given an already constructed merkle tree contained as a slice,
    /// add it to this compound merkle tree.
    pub fn new_from_slice_with_config(
        tree_data: &[u8],
        leafs: usize,
        config: StoreConfig,
    ) -> Result<CompoundMerkleTree<T, A, K, B, N>> {
        let mut trees = Vec::with_capacity(1);
        trees.push(MerkleTree::<T, A, K, B>::from_tree_slice_with_config(
            tree_data, leafs, config,
        )?);

        CompoundMerkleTree::new(trees)
    }

    /// Given an already constructed merkle tree contained as a slice,
    /// add it to this compound merkle tree.
    pub fn new_from_slices(
        tree_data: Vec<&[u8]>,
        leafs: usize,
    ) -> Result<CompoundMerkleTree<T, A, K, B, N>> {
        let mut trees = Vec::with_capacity(tree_data.len());
        for data in tree_data {
            trees.push(MerkleTree::<T, A, K, B>::from_tree_slice(data, leafs)?);
        }

        CompoundMerkleTree::new(trees)
    }

    /// Given an already constructed merkle tree contained as a slice,
    /// add it to this compound merkle tree.
    pub fn new_from_slices_with_configs(
        tree_data: Vec<&[u8]>,
        leafs: usize,
        configs: Vec<StoreConfig>,
    ) -> Result<CompoundMerkleTree<T, A, K, B, N>> {
        let mut trees = Vec::with_capacity(tree_data.len());
        for i in 0..tree_data.len() {
            trees.push(MerkleTree::<T, A, K, B>::from_tree_slice_with_config(
                tree_data[i],
                leafs,
                configs[i].clone(),
            )?);
        }

        CompoundMerkleTree::new(trees)
    }

    /// Given a vector of Stores (i.e. backing to MTs), instantiate
    /// each tree and return a compound merkle tree with them.
    pub fn new_from_stores(
        leafs: usize,
        stores: Vec<K>,
    ) -> Result<CompoundMerkleTree<T, A, K, B, N>> {
        let mut trees = Vec::with_capacity(stores.len());
        for store in stores {
            trees.push(MerkleTree::<T, A, K, B>::from_data_store(store, leafs)?);
        }

        CompoundMerkleTree::new(trees)
    }

    /// Given a vector of StoreConfig's (i.e on-disk references to
    /// stores), instantiate each tree and return a compound merkle
    /// tree with them.
    pub fn new_from_store_configs(
        leafs: usize,
        configs: Vec<StoreConfig>,
    ) -> Result<CompoundMerkleTree<T, A, K, B, N>> {
        let branches = <B as Unsigned>::to_usize();
        let mut trees = Vec::with_capacity(configs.len());
        for config in configs {
            let data = K::new_with_config(get_merkle_tree_len(leafs, branches), branches, config)
                .context("failed to create data store")?;
            trees.push(MerkleTree::<T, A, K, B>::from_data_store(data, leafs)?);
        }

        CompoundMerkleTree::new(trees)
    }

    /// Generate merkle tree inclusion proof for leaf `i`
    #[inline]
    pub fn gen_proof(&self, i: usize) -> Result<CompoundProof<T, B, N>> {
        ensure!(
            i < self.leafs,
            "{} is out of bounds (max: {})",
            i,
            self.leafs
        ); // i in [0 .. self.leafs)

        // Locate the sub-tree the leaf is contained in.
        let tree_index = i / (self.leafs / self.top_layer_nodes);
        let tree = &self.trees[tree_index];
        let tree_leafs = tree.leafs();

        // Get the leaf index within the sub-tree.
        let leaf_index = i % tree_leafs;

        // Generate the proof that will validate to the provided
        // sub-tree root (note the branching factor of B).
        let sub_tree_proof: Proof<T, B> = tree.gen_proof(leaf_index)?;

        let mut lemmas: Vec<T> = Vec::with_capacity(self.top_layer_nodes - 1);
        for i in 0..self.top_layer_nodes {
            if i != tree_index {
                lemmas.push(self.trees[i].root())
            }
        }

        // Construct the top layer proof (1 layer of sub-tree hashes)
        // with the final proof layer being the root.
        let mut lemma: Vec<Vec<T>> = Vec::with_capacity(2); // path + root
        let mut path: Vec<usize> = Vec::with_capacity(1); // path - 1

        lemma.push(lemmas);
        lemma.push(vec![self.root()]);

        path.push(tree_index);

        // Generate the final compound tree proof which is composed of
        // a sub-tree proof of branching factor B and a top-level
        // proof with a branching factor of N.
        Ok(CompoundProof::new(sub_tree_proof, lemma, path))
    }

    pub fn top_layer_nodes(&self) -> usize {
        self.top_layer_nodes
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn leafs(&self) -> usize {
        self.leafs
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn root(&self) -> T {
        self.root.clone()
    }
}
