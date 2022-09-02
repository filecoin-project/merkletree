#![cfg(not(tarpaulin_include))]
pub mod common;

use typenum::{Unsigned, U0, U8};

use merkletree::hash::Algorithm;
use merkletree::merkle::{
    get_merkle_tree_len_generic, get_merkle_tree_row_count, Element, MerkleTree,
};
use merkletree::store::{DiskStore, MmapStore, Store, StoreConfig, VecStore};

use common::{
    get_vector_of_base_trees, get_vector_of_base_trees_as_slices, instantiate_new_with_config,
    serialize_tree, test_disk_mmap_vec_tree_functionality, TestItem, TestItemType,
    TestSha256Hasher, TestXOR128,
};

/// Compound tree constructors
fn instantiate_ctree_from_trees<
    E: Element,
    A: Algorithm<E>,
    S: Store<E>,
    BaseTreeArity: Unsigned,
    SubTreeArity: Unsigned,
>(
    base_tree_leaves: usize,
) -> MerkleTree<E, A, S, BaseTreeArity, SubTreeArity> {
    let base_trees =
        get_vector_of_base_trees::<E, A, S, BaseTreeArity, SubTreeArity>(base_tree_leaves);
    MerkleTree::from_trees(base_trees).expect("failed to instantiate compound tree [from_trees]")
}

fn instantiate_ctree_from_stores<
    E: Element,
    A: Algorithm<E>,
    S: Store<E>,
    BaseTreeArity: Unsigned,
    SubTreeArity: Unsigned,
>(
    base_tree_leaves: usize,
) -> MerkleTree<E, A, S, BaseTreeArity, SubTreeArity> {
    let base_trees =
        get_vector_of_base_trees::<E, A, S, BaseTreeArity, SubTreeArity>(base_tree_leaves);
    let mut stores = Vec::new();
    for tree in base_trees {
        let serialized_tree = serialize_tree(tree);
        stores.push(
            S::new_from_slice(serialized_tree.len(), &serialized_tree)
                .expect("can't create new store over existing one [from_stores]"),
        );
    }

    MerkleTree::from_stores(base_tree_leaves, stores)
        .expect("failed to instantiate compound tree [from_stores]")
}

fn instantiate_ctree_from_slices<
    E: Element,
    A: Algorithm<E>,
    S: Store<E>,
    BaseTreeArity: Unsigned,
    SubTreeArity: Unsigned,
>(
    base_tree_leaves: usize,
) -> MerkleTree<E, A, S, BaseTreeArity, SubTreeArity> {
    let base_trees = get_vector_of_base_trees_as_slices::<E, A, S, BaseTreeArity, SubTreeArity>(
        base_tree_leaves,
    );
    let vec_of_slices: Vec<&[u8]> = base_trees.iter().map(|x| &x[..]).collect();

    MerkleTree::<E, A, S, BaseTreeArity, SubTreeArity>::from_slices(
        &vec_of_slices[..],
        base_tree_leaves,
    )
    .expect("failed to instantiate compound tree from set of base trees [from_slices]")
}

fn instantiate_ctree_from_slices_with_configs<
    E: Element,
    A: Algorithm<E>,
    S: Store<E>,
    BaseTreeArity: Unsigned,
    SubTreeArity: Unsigned,
>(
    base_tree_leaves: usize,
) -> MerkleTree<E, A, S, BaseTreeArity, SubTreeArity> {
    let base_trees = get_vector_of_base_trees_as_slices::<E, A, S, BaseTreeArity, SubTreeArity>(
        base_tree_leaves,
    );
    let vec_of_slices: Vec<&[u8]> = base_trees.iter().map(|x| &x[..]).collect();

    let vec_of_configs = (0..vec_of_slices.len())
        .map(|_| StoreConfig::default())
        .collect::<Vec<StoreConfig>>();

    MerkleTree::<E, A, S, BaseTreeArity, SubTreeArity>::from_slices_with_configs(
        &vec_of_slices[..],
        base_tree_leaves,
        &vec_of_configs[..],
    )
    .expect("failed to instantiate compound tree [from_slices_with_configs]")
}

fn instantiate_ctree_from_store_configs<
    E: Element,
    A: Algorithm<E>,
    S: Store<E>,
    BaseTreeArity: Unsigned,
    SubTreeArity: Unsigned,
>(
    base_tree_leaves: usize,
) -> MerkleTree<E, A, S, BaseTreeArity, SubTreeArity> {
    let distinguisher = "instantiate_ctree_from_store_configs";
    let temp_dir = tempfile::Builder::new()
        .prefix(distinguisher)
        .tempdir()
        .expect("can't create temp dir [from_store_configs]");

    // compute len for base tree as we are going to instantiate compound tree from set of base trees
    let len = get_merkle_tree_len_generic::<BaseTreeArity, U0, U0>(base_tree_leaves)
        .expect("can't get tree len [from_store_configs]");
    let row_count = get_merkle_tree_row_count(base_tree_leaves, BaseTreeArity::to_usize());

    let vec_of_configs = (0..SubTreeArity::to_usize())
        .map(|index| {
            let replica = format!(
                "{}-{}-{}-{}-{}-replica",
                distinguisher, index, base_tree_leaves, len, row_count,
            );

            let config = StoreConfig::new(
                temp_dir.path(),
                replica,
                StoreConfig::default_rows_to_discard(base_tree_leaves, BaseTreeArity::to_usize()),
            );
            // we need to instantiate a tree in order to dump tree data into Disk-based storages and bind them to configs
            instantiate_new_with_config::<E, A, S, BaseTreeArity>(
                base_tree_leaves,
                Some(config.clone()),
            );
            config
        })
        .collect::<Vec<StoreConfig>>();

    MerkleTree::from_store_configs(base_tree_leaves, &vec_of_configs)
        .expect("failed to instantiate compound tree [from_store_configs]")
}

/// Test executor
fn run_test_compound_tree<
    E: Element,
    A: Algorithm<E>,
    S: Store<E>,
    BaseTreeArity: Unsigned,
    SubTreeArity: Unsigned,
>(
    constructor: fn(usize) -> MerkleTree<E, A, S, BaseTreeArity, SubTreeArity>,
    base_tree_leaves: usize,
    expected_leaves: usize,
    expected_len: usize,
    expected_root: E,
) {
    let compound_tree: MerkleTree<E, A, S, BaseTreeArity, SubTreeArity, U0> =
        constructor(base_tree_leaves);
    test_disk_mmap_vec_tree_functionality::<E, A, S, BaseTreeArity, SubTreeArity, U0>(
        compound_tree,
        expected_leaves,
        expected_len,
        expected_root,
    );
}

/// Ultimately we cover following list of constructors for compound trees
/// - from_trees
/// - from_stores
/// - from_slices
/// - from_slices_with_configs
/// - from_store_configs
#[test]
fn test_compound_constructors() {
    fn run_tests<E: Element + Copy, A: Algorithm<E>, S: Store<E>>(root: E) {
        let base_tree_leaves = 64;
        let expected_total_leaves = base_tree_leaves * 8;
        let len = get_merkle_tree_len_generic::<U8, U8, U0>(base_tree_leaves)
            .expect("[test_compound_constructors] couldn't compute Merkle Tree len");

        run_test_compound_tree::<E, A, S, U8, U8>(
            instantiate_ctree_from_trees,
            base_tree_leaves,
            expected_total_leaves,
            len,
            root,
        );

        run_test_compound_tree::<E, A, S, U8, U8>(
            instantiate_ctree_from_stores,
            base_tree_leaves,
            expected_total_leaves,
            len,
            root,
        );

        run_test_compound_tree::<E, A, S, U8, U8>(
            instantiate_ctree_from_slices,
            base_tree_leaves,
            expected_total_leaves,
            len,
            root,
        );

        run_test_compound_tree::<E, A, S, U8, U8>(
            instantiate_ctree_from_slices_with_configs,
            base_tree_leaves,
            expected_total_leaves,
            len,
            root,
        );
    }
    let root_xor128 = TestItem::from_slice(&[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    run_tests::<TestItemType, TestXOR128, VecStore<TestItemType>>(root_xor128);
    // TODO: investigate why these tests fail
    //run_tests::<TestItemType, TestXOR128, DiskStore<TestItemType>>(root_xor128);
    //run_tests::<TestItemType, TestXOR128, MmapStore<TestItemType>>(root_xor128);

    let root_sha256 = TestItem::from_slice(&[
        32, 129, 168, 134, 58, 233, 155, 225, 88, 230, 247, 63, 18, 38, 194, 230,
    ]);
    run_tests::<TestItemType, TestSha256Hasher, VecStore<TestItemType>>(root_sha256);
    // TODO: investigate why these tests fail
    //run_tests::<TestItemType, TestSha256Hasher, DiskStore<TestItemType>>(root_sha256);
    //run_tests::<TestItemType, TestSha256Hasher, MmapStore<TestItemType>>(root_sha256);

    let base_tree_leaves = 64;
    let expected_total_leaves = base_tree_leaves * 8;
    let len = get_merkle_tree_len_generic::<U8, U8, U0>(base_tree_leaves)
        .expect("[test_compound_constructors] couldn't compute Merkle Tree len");

    // this instantiator works only with DiskStore / MmapStore trees
    run_test_compound_tree::<TestItemType, TestXOR128, DiskStore<TestItemType>, U8, U8>(
        instantiate_ctree_from_store_configs,
        base_tree_leaves,
        expected_total_leaves,
        len,
        root_xor128,
    );
    run_test_compound_tree::<TestItemType, TestSha256Hasher, DiskStore<TestItemType>, U8, U8>(
        instantiate_ctree_from_store_configs,
        base_tree_leaves,
        expected_total_leaves,
        len,
        root_sha256,
    );

    // same instantiator for MmapStore..
    run_test_compound_tree::<TestItemType, TestXOR128, MmapStore<TestItemType>, U8, U8>(
        instantiate_ctree_from_store_configs,
        base_tree_leaves,
        expected_total_leaves,
        len,
        root_xor128,
    );
    run_test_compound_tree::<TestItemType, TestSha256Hasher, MmapStore<TestItemType>, U8, U8>(
        instantiate_ctree_from_store_configs,
        base_tree_leaves,
        expected_total_leaves,
        len,
        root_sha256,
    );
}
