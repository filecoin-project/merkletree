use std::marker::PhantomData;

use anyhow::Result;

use crate::ccompound_merkle_proof::CCompoundMerkleProof;
use crate::compound_merkle::CompoundMerkleTree;
use crate::compound_merkle_proof::CompoundMerkleProof;
use crate::hash::Algorithm;
use crate::merkle::Element;
use crate::store::Store;
use typenum::marker_traits::Unsigned;

/// Compound Compound Merkle Tree.
///
/// A compound compound merkle tree is a type of compound merkle tree
/// in which every non-leaf node is the hash of its child nodes.
#[derive(Debug, Clone, Eq, PartialEq, Default)]
pub struct CCompoundMerkleTree<T, A, K, B, N, R>
where
    T: Element,
    A: Algorithm<T>,
    K: Store<T>,
    B: Unsigned, // Branching factor of sub-trees
    N: Unsigned, // Number of nodes at sub-tree top layer
    R: Unsigned, // Number of nodes at top layer
{
    trees: Vec<CompoundMerkleTree<T, A, K, B, N>>,
    top_layer_nodes: usize,
    len: usize,
    leafs: usize,
    height: usize,
    root: T,

    _r: PhantomData<R>,
}

impl<T: Element, A: Algorithm<T>, K: Store<T>, B: Unsigned, N: Unsigned, R: Unsigned>
    CCompoundMerkleTree<T, A, K, B, N, R>
{
    /// Creates new compound merkle tree from a vector of merkle
    /// trees.  The ordering of the trees is significant, as trees are
    /// leaf indexed / addressable in the same sequence that they are
    /// provided here.
    pub fn from_trees(
        trees: Vec<CompoundMerkleTree<T, A, K, B, N>>,
    ) -> Result<CCompoundMerkleTree<T, A, K, B, N, R>> {
        let top_layer_nodes = R::to_usize();
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

        // If we are building a compound compound tree with only a single tree,
        // all properties revert to the single tree properties.  This
        // is done as an interface simplification where a
        // CCompoundMerkleTree can simply represent a CompoundMerkleTree.
        let (leafs, len, height, root) = if top_layer_nodes == 1 {
            (
                trees[0].leafs(),
                trees[0].len(),
                trees[0].height(),
                trees[0].root(),
            )
        } else {
            // Total number of leafs in the compound tree is the combined leafs total of all subtrees.
            let leafs = trees.iter().fold(0, |leafs, mt| leafs + mt.leafs());
            // Total length of the compound tree is the combined length of all subtrees plus the root.
            let len = trees.iter().fold(0, |len, mt| len + mt.len()) + 1;
            // Total height of the compound tree is the height of any of the sub-trees to top-layer plus root.
            let height = trees[0].height() + 1;
            // Calculate the compound root by hashing the top layer roots together.
            let roots: Vec<T> = trees.iter().map(|x| x.root()).collect();
            let root = A::default().multi_node(&roots, 1);

            (leafs, len, height, root)
        };

        Ok(CCompoundMerkleTree {
            trees,
            top_layer_nodes,
            len,
            leafs,
            height,
            root,
            _r: PhantomData,
        })
    }

    /// Generate merkle tree inclusion proof for leaf `i`
    pub fn gen_proof(&self, i: usize) -> Result<CCompoundMerkleProof<T, B, N, R>> {
        ensure!(
            i < self.leafs,
            "{} is out of bounds (max: {})",
            i,
            self.leafs
        ); // i in [0 .. self.leafs)

        ensure!(
            R::to_usize() == self.top_layer_nodes,
            "Invalid top layer node value"
        );

        // Locate the sub-tree the leaf is contained in.
        let tree_index = i / (self.leafs / self.top_layer_nodes);
        let tree = &self.trees[tree_index];
        let tree_leafs = tree.leafs();

        // Get the leaf index within the sub-tree.
        let leaf_index = i % tree_leafs;

        // Generate the proof that will validate to the provided
        // sub-tree root (note the branching factor of N).
        let sub_tree_proof: CompoundMerkleProof<T, B, N> = tree.gen_proof(leaf_index)?;
        if self.top_layer_nodes == 1 {
            return CCompoundMerkleProof::<T, B, N, R>::new(sub_tree_proof, Vec::new(), Vec::new());
        }

        // Construct the top layer proof.  'lemma' length is
        // top_layer_nodes - 1 + root == top_layer_nodes
        let mut path: Vec<usize> = Vec::with_capacity(1); // path - 1
        let mut lemma: Vec<T> = Vec::with_capacity(self.top_layer_nodes);
        for i in 0..self.top_layer_nodes {
            if i != tree_index {
                lemma.push(self.trees[i].root())
            }
        }

        lemma.push(self.root());
        path.push(tree_index);

        // Generate the final compound tree proof which is composed of
        // a sub-tree proof of branching factor B and a top-level
        // proof with a branching factor of R.
        CCompoundMerkleProof::<T, B, N, R>::new(sub_tree_proof, lemma, path)
    }

    /// Generate merkle tree inclusion proof for leaf `i` using partial trees built from cached data.
    pub fn gen_proof_from_cached_tree(
        &self,
        i: usize,
        levels: usize,
    ) -> Result<CCompoundMerkleProof<T, B, N, R>> {
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
        // sub-tree root (note the branching factor of N).
        let sub_tree_proof = tree.gen_proof_from_cached_tree(leaf_index, levels)?;
        if self.top_layer_nodes == 1 {
            return CCompoundMerkleProof::<T, B, N, R>::new(sub_tree_proof, Vec::new(), Vec::new());
        }

        // Construct the top layer proof.  'lemma' length is
        // top_layer_nodes - 1 + root == top_layer_nodes
        let mut path: Vec<usize> = Vec::with_capacity(1); // path - 1
        let mut lemma: Vec<T> = Vec::with_capacity(self.top_layer_nodes);
        for i in 0..self.top_layer_nodes {
            if i != tree_index {
                lemma.push(self.trees[i].root())
            }
        }

        lemma.push(self.root());
        path.push(tree_index);

        // Generate the final compound tree proof which is composed of
        // a sub-tree proof of branching factor B and a top-level
        // proof with a branching factor of N.
        CCompoundMerkleProof::<T, B, N, R>::new(sub_tree_proof, lemma, path)
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

    /// Returns merkle leaf element
    #[inline]
    pub fn read_at(&self, i: usize) -> Result<T> {
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

        tree.read_at(leaf_index)
    }
}
