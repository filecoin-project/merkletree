#![cfg(not(tarpaulin_include))]
pub mod common;

use typenum::{Unsigned, U0, U2, U8};

use merkletree::hash::Algorithm;
use merkletree::merkle::{get_merkle_tree_len_generic, Element, MerkleTree};
use merkletree::store::{DiskStore, MmapStore, Store, StoreConfig, VecStore};

use crate::common::{
    get_vector_of_base_trees, instantiate_new, instantiate_new_with_config,
    test_disk_mmap_vec_tree_functionality, TestItem, TestItemType, TestSha256Hasher, TestXOR128,
};

/// Compound-compound tree constructors
fn instantiate_cctree_from_sub_trees<
    E: Element,
    A: Algorithm<E>,
    S: Store<E>,
    BaseTreeArity: Unsigned,
    SubTreeArity: Unsigned,
    TopTreeArity: Unsigned,
>(
    base_tree_leaves: usize,
) -> MerkleTree<E, A, S, BaseTreeArity, SubTreeArity, TopTreeArity> {
    let compound_trees = (0..TopTreeArity::to_usize())
        .map(|_| {
            let base_trees =
                get_vector_of_base_trees::<E, A, S, BaseTreeArity, SubTreeArity>(base_tree_leaves);
            MerkleTree::from_trees(base_trees)
                .expect("failed to instantiate compound tree [instantiate_cctree_from_sub_trees]")
        })
        .collect();

    MerkleTree::from_sub_trees(compound_trees)
        .expect("failed to instantiate compound-compound tree from compound trees [instantiate_cctree_from_sub_trees]")
}

fn instantiate_cctree_from_sub_trees_as_trees<
    E: Element,
    A: Algorithm<E>,
    S: Store<E>,
    BaseTreeArity: Unsigned,
    SubTreeArity: Unsigned,
    TopTreeArity: Unsigned,
>(
    base_tree_leaves: usize,
) -> MerkleTree<E, A, S, BaseTreeArity, SubTreeArity, TopTreeArity> {
    let base_trees = (0..TopTreeArity::to_usize())
        .flat_map(|_| {
            (0..SubTreeArity::to_usize())
                .map(|_| instantiate_new(base_tree_leaves, None))
                .collect::<Vec<MerkleTree<E, A, S, BaseTreeArity, U0, U0>>>()
        })
        .collect();

    MerkleTree::from_sub_trees_as_trees(base_trees)
        .expect("failed to instantiate compound-compound tree from set of base trees [instantiate_cctree_from_sub_trees_as_trees]")
}

fn instantiate_cctree_from_sub_tree_store_configs<
    E: Element,
    A: Algorithm<E>,
    S: Store<E>,
    BaseTreeArity: Unsigned,
    SubTreeArity: Unsigned,
    TopTreeArity: Unsigned,
>(
    base_tree_leaves: usize,
) -> MerkleTree<E, A, S, BaseTreeArity, SubTreeArity, TopTreeArity> {
    let distinguisher = "instantiate_ctree_from_store_configs";
    let temp_dir = tempfile::Builder::new()
        .prefix(distinguisher)
        .tempdir()
        .expect("can't create temp dir [instantiate_cctree_from_sub_tree_store_configs]");

    // compute len for base tree as we are going to instantiate compound tree from set of base trees
    let len = get_merkle_tree_len_generic::<BaseTreeArity, U0, U0>(base_tree_leaves)
        .expect("can't get tree len [instantiate_cctree_from_sub_tree_store_configs]");

    let configs = (0..TopTreeArity::to_usize())
        .flat_map(|j| {
            (0..SubTreeArity::to_usize())
                .map(|i| {
                    let replica = format!(
                        "{}-{}-{}-{}-{}-replica",
                        distinguisher, i, j, base_tree_leaves, len,
                    );

                    // we attempt to discard all intermediate layers, except bottom one (set of leaves) and top-level root of base tree
                    let config = StoreConfig::new(temp_dir.path(), replica, 0);
                    // we need to instantiate a tree in order to dump tree data into Disk-based storages and bind them to configs
                    instantiate_new_with_config::<E, A, S, BaseTreeArity>(
                        base_tree_leaves,
                        Some(config.clone()),
                    );
                    config
                })
                .collect::<Vec<StoreConfig>>()
        })
        .collect::<Vec<StoreConfig>>();

    MerkleTree::from_sub_tree_store_configs(base_tree_leaves, &configs)
        .expect("failed to instantiate compound-compound tree [instantiate_cctree_from_sub_tree_store_configs]")
}

/// Test executor
fn run_test_compound_compound_tree<
    E: Element,
    A: Algorithm<E>,
    S: Store<E>,
    BaseTreeArity: Unsigned,
    SubTreeArity: Unsigned,
    TopTreeArity: Unsigned,
>(
    constructor: fn(usize) -> MerkleTree<E, A, S, BaseTreeArity, SubTreeArity, TopTreeArity>,
    base_tree_leaves: usize,
    expected_leaves: usize,
    expected_len: usize,
    expected_root: E,
) {
    let compound_tree: MerkleTree<E, A, S, BaseTreeArity, SubTreeArity, TopTreeArity> =
        constructor(base_tree_leaves);
    test_disk_mmap_vec_tree_functionality::<E, A, S, BaseTreeArity, SubTreeArity, TopTreeArity>(
        compound_tree,
        expected_leaves,
        expected_len,
        expected_root,
    );
}

/// Ultimately we cover following list of constructors for compound-compound trees
/// - from_sub_trees
/// - from_sub_trees_as_trees
/// - from_sub_tree_store_configs
#[test]
fn test_compound_compound_constructors() {
    fn run_test<E: Element + Copy, A: Algorithm<E>, S: Store<E>>(root: E) {
        let base_tree_leaves = 64;
        let expected_total_leaves = base_tree_leaves * 8 * 2;
        let len = get_merkle_tree_len_generic::<U8, U8, U2>(base_tree_leaves)
            .expect("[test_compound_compound_constructors] couldn't compute Merkle Tree len");

        run_test_compound_compound_tree::<E, A, S, U8, U8, U2>(
            instantiate_cctree_from_sub_trees,
            base_tree_leaves,
            expected_total_leaves,
            len,
            root,
        );

        run_test_compound_compound_tree::<E, A, S, U8, U8, U2>(
            instantiate_cctree_from_sub_trees_as_trees,
            base_tree_leaves,
            expected_total_leaves,
            len,
            root,
        );
    }

    let root_xor128 = TestItem::from_slice(&[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    run_test::<TestItemType, TestXOR128, VecStore<TestItemType>>(root_xor128);

    let root_sha256 = TestItem::from_slice(&[
        52, 152, 123, 224, 174, 42, 152, 12, 199, 4, 105, 245, 176, 59, 230, 86,
    ]);
    run_test::<TestItemType, TestSha256Hasher, VecStore<TestItemType>>(root_sha256);

    let base_tree_leaves = 64;
    let expected_total_leaves = base_tree_leaves * 8 * 2;
    let len = get_merkle_tree_len_generic::<U8, U8, U2>(base_tree_leaves)
        .expect("[test_compound_compound_constructors] couldn't compute Merkle Tree len");

    // this instantiator works only with DiskStore / MmapStore trees
    run_test_compound_compound_tree::<TestItemType, TestXOR128, DiskStore<TestItemType>, U8, U8, U2>(
        instantiate_cctree_from_sub_tree_store_configs,
        base_tree_leaves,
        expected_total_leaves,
        len,
        root_xor128,
    );

    run_test_compound_compound_tree::<
        TestItemType,
        TestSha256Hasher,
        DiskStore<TestItemType>,
        U8,
        U8,
        U2,
    >(
        instantiate_cctree_from_sub_tree_store_configs,
        base_tree_leaves,
        expected_total_leaves,
        len,
        root_sha256,
    );

    // same instantiator for MmapStore..
    run_test_compound_compound_tree::<TestItemType, TestXOR128, MmapStore<TestItemType>, U8, U8, U2>(
        instantiate_cctree_from_sub_tree_store_configs,
        base_tree_leaves,
        expected_total_leaves,
        len,
        root_xor128,
    );

    run_test_compound_compound_tree::<
        TestItemType,
        TestSha256Hasher,
        MmapStore<TestItemType>,
        U8,
        U8,
        U2,
    >(
        instantiate_cctree_from_sub_tree_store_configs,
        base_tree_leaves,
        expected_total_leaves,
        len,
        root_sha256,
    );
}
