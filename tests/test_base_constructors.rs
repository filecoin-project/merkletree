#![cfg(not(tarpaulin_include))]
pub mod common;

use rayon::iter::IntoParallelIterator;
use typenum::{Unsigned, U0, U2, U8};

use merkletree::hash::Algorithm;
use merkletree::merkle::{
    get_merkle_tree_len_generic, get_merkle_tree_row_count, Element, FromIndexedParallelIterator,
    MerkleTree,
};
use merkletree::store::{
    DiskStore, LevelCacheStore, MmapStore, Store, StoreConfig, VecStore, SMALL_TREE_BUILD,
};

use crate::common::{
    generate_vector_of_usizes, instantiate_new, instantiate_new_with_config,
    test_disk_mmap_vec_tree_functionality, TestItem, TestItemType, TestSha256Hasher, TestXOR128,
};

/// Base tree constructors
fn instantiate_try_from_iter<E: Element, A: Algorithm<E>, S: Store<E>, U: Unsigned>(
    leaves: usize,
    _config: Option<StoreConfig>,
) -> MerkleTree<E, A, S, U> {
    let dataset = common::generate_vector_of_elements::<E>(leaves);
    MerkleTree::try_from_iter(dataset.into_iter().map(Ok))
        .expect("failed to instantiate tree [try_from_iter]")
}

fn instantiate_from_par_iter<E: Element, A: Algorithm<E>, S: Store<E>, U: Unsigned>(
    leaves: usize,
    _config: Option<StoreConfig>,
) -> MerkleTree<E, A, S, U> {
    let dataset = common::generate_vector_of_elements::<E>(leaves);
    MerkleTree::from_par_iter(dataset.into_par_iter())
        .expect("failed to instantiate tree [try_from_par_iter]")
}

fn instantiate_try_from_iter_with_config<E: Element, A: Algorithm<E>, S: Store<E>, U: Unsigned>(
    leaves: usize,
    config: Option<StoreConfig>,
) -> MerkleTree<E, A, S, U> {
    let dataset = common::generate_vector_of_elements::<E>(leaves);
    MerkleTree::try_from_iter_with_config(
        dataset.into_iter().map(Ok),
        config.expect("can't get tree's config [try_from_iter_with_config]"),
    )
    .expect("failed to instantiate tree [try_from_iter_with_config]")
}

fn instantiate_from_par_iter_with_config<E: Element, A: Algorithm<E>, S: Store<E>, U: Unsigned>(
    leaves: usize,
    config: Option<StoreConfig>,
) -> MerkleTree<E, A, S, U> {
    let dataset = common::generate_vector_of_elements::<E>(leaves);
    MerkleTree::from_par_iter_with_config(
        dataset,
        config.expect("can't get tree's config [from_par_iter_with_config]"),
    )
    .expect("failed to instantiate tree [from_par_iter_with_config]")
}

fn instantiate_from_data<E: Element, A: Algorithm<E>, S: Store<E>, U: Unsigned>(
    leaves: usize,
    _config: Option<StoreConfig>,
) -> MerkleTree<E, A, S, U> {
    let dataset = generate_vector_of_usizes(leaves);
    MerkleTree::from_data(dataset.as_slice()).expect("failed to instantiate tree [from_data]")
}

fn instantiate_from_data_with_config<E: Element, A: Algorithm<E>, S: Store<E>, U: Unsigned>(
    leaves: usize,
    config: Option<StoreConfig>,
) -> MerkleTree<E, A, S, U> {
    let dataset = generate_vector_of_usizes(leaves);
    MerkleTree::from_data_with_config(
        dataset.as_slice(),
        config.expect("can't get tree's config [from_data_with_config]"),
    )
    .expect("failed to instantiate tree [from_data_with_config]")
}

fn instantiate_from_data_store<E: Element, A: Algorithm<E>, S: Store<E>, U: Unsigned>(
    leaves: usize,
    _config: Option<StoreConfig>,
) -> MerkleTree<E, A, S, U> {
    let tree = instantiate_from_data::<E, A, S, U>(leaves, None);
    let serialized_tree = common::serialize_tree(tree);
    let store = Store::new_from_slice(serialized_tree.len(), &serialized_tree)
        .expect("can't create new store over existing one [from_data_store]");
    MerkleTree::from_data_store(store, leaves)
        .expect("failed to instantiate tree [from_data_store]")
}

fn instantiate_from_tree_slice<E: Element, A: Algorithm<E>, S: Store<E>, U: Unsigned>(
    leaves: usize,
    _config: Option<StoreConfig>,
) -> MerkleTree<E, A, S, U> {
    let tree = instantiate_from_data::<E, A, S, U>(leaves, None);
    let serialized_tree = common::serialize_tree(tree);
    MerkleTree::from_tree_slice(serialized_tree.as_slice(), leaves)
        .expect("failed to instantiate tree [from_tree_slice]")
}

fn instantiate_from_byte_slice<E: Element, A: Algorithm<E>, S: Store<E>, U: Unsigned>(
    leaves: usize,
    _config: Option<StoreConfig>,
) -> MerkleTree<E, A, S, U> {
    let dataset = common::generate_byte_slice_tree::<E, A>(leaves);
    MerkleTree::from_byte_slice(dataset.as_slice())
        .expect("failed to instantiate tree [from_byte_slice]")
}

fn instantiate_from_byte_slice_with_config<
    E: Element,
    A: Algorithm<E>,
    S: Store<E>,
    U: Unsigned,
>(
    leaves: usize,
    config: Option<StoreConfig>,
) -> MerkleTree<E, A, S, U> {
    let dataset = common::generate_byte_slice_tree::<E, A>(leaves);
    MerkleTree::from_byte_slice_with_config(
        dataset.as_slice(),
        config.expect("from_byte_slice_with_config"),
    )
    .expect("failed to instantiate tree [from_byte_slice_with_config]")
}

fn instantiate_from_tree_slice_with_config<
    E: Element,
    A: Algorithm<E>,
    S: Store<E>,
    U: Unsigned,
>(
    leaves: usize,
    config: Option<StoreConfig>,
) -> MerkleTree<E, A, S, U> {
    let tmp_tree = instantiate_from_data::<E, A, S, U>(leaves, None);
    let serialized_tree = common::serialize_tree(tmp_tree);
    MerkleTree::from_tree_slice_with_config(
        serialized_tree.as_slice(),
        leaves,
        config.expect("can't get tree's config [from_tree_slice_with_config]"),
    )
    .expect("failed to instantiate tree [from_tree_slice_with_config]")
}

/// Test executor
fn run_test_base_tree<E: Element, A: Algorithm<E>, S: Store<E>, BaseTreeArity: Unsigned>(
    constructor: fn(usize, Option<StoreConfig>) -> MerkleTree<E, A, S, BaseTreeArity>,
    leaves_in_tree: usize,
    config: Option<StoreConfig>,
    expected_leaves: usize,
    expected_len: usize,
    expected_root: E,
) {
    // base tree has SubTreeArity and TopTreeArity parameters equal to zero
    let tree: MerkleTree<E, A, S, BaseTreeArity, U0, U0> = constructor(leaves_in_tree, config);
    test_disk_mmap_vec_tree_functionality(tree, expected_leaves, expected_len, expected_root);
}

/// Ultimately we have a list of constructors for base trees
/// that we can divide by actual dataset generator and organize
/// complex integration tests that evaluate correct instantiation
/// of base tree (with fixing base tree arity parameter to U8 - oct tree)
/// using distinct hashers
///
/// [Iterable]
/// - new
/// - try_from_iter
/// - from_par_iter
/// - new_with_config
/// - try_from_iter_with_config
/// - from_par_iter_with_config
///
/// [Iterable+Hashable, Serialization]
/// - from_data
/// - from_data_with_config
/// - from_data_store
/// - from_tree_slice
/// - from_byte_slice
/// - from_tree_slice_with_config
/// - from_byte_slice_with_config

#[test]
fn test_iterable() {
    fn run_tests<E: Element + Copy, A: Algorithm<E>, S: Store<E>>(root: E) {
        let base_tree_leaves = 64;
        let expected_total_leaves = base_tree_leaves;
        let len = get_merkle_tree_len_generic::<U8, U0, U0>(base_tree_leaves)
            .expect("[test_iterable] couldn't compute Merkle Tree len");

        run_test_base_tree::<E, A, S, U8>(
            instantiate_new,
            base_tree_leaves,
            None,
            expected_total_leaves,
            len,
            root,
        );

        run_test_base_tree::<E, A, S, U8>(
            instantiate_try_from_iter,
            base_tree_leaves,
            None,
            expected_total_leaves,
            len,
            root,
        );

        run_test_base_tree::<E, A, S, U8>(
            instantiate_from_par_iter,
            base_tree_leaves,
            None,
            expected_total_leaves,
            len,
            root,
        );

        let distinguisher = "instantiate_new_with_config";
        let temp_dir = tempfile::Builder::new()
            .prefix(distinguisher)
            .tempdir()
            .expect("[test_iterable] couldn't create temp_dir");
        run_test_base_tree::<E, A, S, U8>(
            instantiate_new_with_config,
            base_tree_leaves,
            Some(StoreConfig::new(
                temp_dir.into_path(),
                String::from(distinguisher),
                0,
            )),
            expected_total_leaves,
            len,
            root,
        );

        let distinguisher = "instantiate_try_from_iter_with_config";
        let temp_dir = tempfile::Builder::new()
            .prefix(distinguisher)
            .tempdir()
            .expect("[test_iterable] couldn't create temp_dir");
        run_test_base_tree::<E, A, S, U8>(
            instantiate_try_from_iter_with_config,
            base_tree_leaves,
            Some(StoreConfig::new(
                temp_dir.into_path(),
                String::from(distinguisher),
                0,
            )),
            expected_total_leaves,
            len,
            root,
        );

        let distinguisher = "instantiate_from_par_iter_with_config";
        let temp_dir = tempfile::Builder::new()
            .prefix(distinguisher)
            .tempdir()
            .expect("[test_iterable] couldn't create temp_dir");
        run_test_base_tree::<E, A, S, U8>(
            instantiate_from_par_iter_with_config,
            base_tree_leaves,
            Some(StoreConfig::new(
                temp_dir.into_path(),
                String::from(distinguisher),
                0,
            )),
            expected_total_leaves,
            len,
            root,
        );
    }

    // Run set of tests over XOR128-based hasher
    let root_xor128 =
        TestItemType::from_slice(&[65, 0, 64, 0, 64, 0, 64, 0, 64, 0, 64, 0, 64, 0, 64, 0]);
    run_tests::<TestItemType, TestXOR128, VecStore<TestItemType>>(root_xor128);
    run_tests::<TestItemType, TestXOR128, DiskStore<TestItemType>>(root_xor128);
    run_tests::<TestItemType, TestXOR128, MmapStore<TestItemType>>(root_xor128);

    // Run set of tests over SHA256-based hasher
    let root_sha256 = TestItem::from_slice(&[
        252, 61, 163, 229, 140, 223, 198, 165, 200, 137, 59, 43, 83, 136, 197, 63,
    ]);
    run_tests::<TestItemType, TestSha256Hasher, VecStore<TestItemType>>(root_sha256);
    run_tests::<TestItemType, TestSha256Hasher, DiskStore<TestItemType>>(root_sha256);
    run_tests::<TestItemType, TestSha256Hasher, MmapStore<TestItemType>>(root_sha256);
}

#[test]
fn test_iterable_hashable_and_serialization() {
    fn run_tests<E: Element + Copy, A: Algorithm<E>, S: Store<E>>(root: E) {
        let base_tree_leaves = 64;
        let expected_total_leaves = base_tree_leaves;
        let len = get_merkle_tree_len_generic::<U8, U0, U0>(base_tree_leaves)
            .expect("[test_iterable_hashable_and_serialization] couldn't compute Merkle Tree len");

        run_test_base_tree::<E, A, S, U8>(
            instantiate_from_data,
            base_tree_leaves,
            None,
            expected_total_leaves,
            len,
            root,
        );

        let distinguisher = "instantiate_from_data_with_config";
        let temp_dir = tempfile::Builder::new()
            .prefix(distinguisher)
            .tempdir()
            .expect("[test_iterable_hashable_and_serialization] couldn't create temp_dir");
        run_test_base_tree::<E, A, S, U8>(
            instantiate_from_data_with_config,
            base_tree_leaves,
            Some(StoreConfig::new(
                temp_dir.into_path(),
                String::from(distinguisher),
                0,
            )),
            expected_total_leaves,
            len,
            root,
        );

        run_test_base_tree::<E, A, S, U8>(
            instantiate_from_data_store,
            base_tree_leaves,
            None,
            expected_total_leaves,
            len,
            root,
        );

        run_test_base_tree::<E, A, S, U8>(
            instantiate_from_tree_slice,
            base_tree_leaves,
            None,
            expected_total_leaves,
            len,
            root,
        );

        run_test_base_tree::<E, A, S, U8>(
            instantiate_from_byte_slice,
            base_tree_leaves,
            None,
            expected_total_leaves,
            len,
            root,
        );

        let distinguisher = "instantiate_from_byte_slice_with_config";
        let temp_dir = tempfile::Builder::new()
            .prefix(distinguisher)
            .tempdir()
            .expect("[test_iterable_hashable_and_serialization] couldn't create temp_dir");
        run_test_base_tree::<E, A, S, U8>(
            instantiate_from_byte_slice_with_config,
            base_tree_leaves,
            Some(StoreConfig::new(
                temp_dir.into_path(),
                String::from(distinguisher),
                0,
            )),
            expected_total_leaves,
            len,
            root,
        );

        let distinguisher = "instantiate_from_tree_slice_with_config";
        let temp_dir = tempfile::Builder::new()
            .prefix(distinguisher)
            .tempdir()
            .expect("[test_iterable_hashable_and_serialization] couldn't create temp_dir");
        run_test_base_tree::<E, A, S, U8>(
            instantiate_from_tree_slice_with_config,
            base_tree_leaves,
            Some(StoreConfig::new(
                temp_dir.into_path(),
                String::from(distinguisher),
                0,
            )),
            expected_total_leaves,
            len,
            root,
        );
    }

    // Run set of tests over XOR128-based hasher
    let root_xor128 = TestItemType::from_slice(&[1, 0, 0, 0, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    run_tests::<TestItemType, TestXOR128, VecStore<TestItemType>>(root_xor128);
    run_tests::<TestItemType, TestXOR128, DiskStore<TestItemType>>(root_xor128);
    run_tests::<TestItemType, TestXOR128, MmapStore<TestItemType>>(root_xor128);

    // Run set of tests over SHA256-based hasher
    let root_sha256 = TestItem::from_slice(&[
        98, 103, 202, 101, 121, 179, 6, 237, 133, 39, 253, 169, 173, 63, 89, 188,
    ]);
    run_tests::<TestItemType, TestSha256Hasher, VecStore<TestItemType>>(root_sha256);
    run_tests::<TestItemType, TestSha256Hasher, DiskStore<TestItemType>>(root_sha256);
    run_tests::<TestItemType, TestSha256Hasher, MmapStore<TestItemType>>(root_sha256);
}

/// Test executor
///
/// Logically this test only checks that created tree has expected storage.
/// Since Rust doesn't provide special tools for comparing types (only some unstable tools
/// exist - https://stackoverflow.com/questions/60138397/how-to-test-for-type-equality-in-rust)
/// we just compare partial strings from formatted storages
fn run_base_tree_storage_test<E: Element, A: Algorithm<E>, S: Store<E>, BaseTreeArity: Unsigned>(
    constructor: fn(usize, Option<StoreConfig>) -> MerkleTree<E, A, S, BaseTreeArity>,
    leaves_in_tree: usize,
    config: Option<StoreConfig>,
    expected_storage: S,
) {
    // it should be enough for our current storages
    const SYMBOLS_TO_TRUNCATE: usize = 5;

    let tree = constructor(leaves_in_tree, config);
    let actual_storage = tree.data().expect("can't get type of tree's storage");

    let mut expected = format!("{:?}", expected_storage);
    let mut actual = format!("{:?}", actual_storage);

    expected.truncate(SYMBOLS_TO_TRUNCATE);
    actual.truncate(SYMBOLS_TO_TRUNCATE);

    assert_eq!(expected, actual);
}

/// This integration test evaluates that base tree of any storage (Disk, LevelCache and Mmap;
/// we don't check here VecStore as it is already evaluated in previous test) can be correctly
/// instantiated with expected data storage type
///
#[test]
fn test_storage_types() {
    let base_tree_leaves = 64;
    let expected_total_leaves = base_tree_leaves;
    let branches = 8;

    // Disk
    type DiskStorage = DiskStore<TestItemType>;
    let distinguisher = "instantiate_new_with_config-disk";
    let temp_dir = tempfile::Builder::new()
        .prefix(distinguisher)
        .tempdir()
        .expect("[test_storage_types] couldn't create temp_dir");
    run_base_tree_storage_test::<TestItemType, TestXOR128, DiskStorage, U8>(
        instantiate_new_with_config,
        base_tree_leaves,
        Some(StoreConfig::new(
            temp_dir.into_path(),
            String::from(distinguisher),
            StoreConfig::default_rows_to_discard(expected_total_leaves, branches),
        )),
        DiskStorage::new(1).expect("[test_storage_types] couldn't create DiskStorage"),
    );

    // Mmap
    type MmapStorage = MmapStore<TestItemType>;
    let distinguisher = "instantiate_new_with_config-mmap";
    let temp_dir = tempfile::Builder::new()
        .prefix(distinguisher)
        .tempdir()
        .expect("[test_storage_types] couldn't create temp_dir");
    run_base_tree_storage_test::<TestItemType, TestXOR128, MmapStorage, U8>(
        instantiate_new_with_config,
        base_tree_leaves,
        Some(StoreConfig::new(
            temp_dir.into_path(),
            String::from(distinguisher),
            StoreConfig::default_rows_to_discard(expected_total_leaves, branches),
        )),
        MmapStorage::new(1).expect("[test_storage_types] couldn't instantiate MmapStorage"),
    );

    // Level-cache
    type LevelCacheStorage = LevelCacheStore<TestItemType, std::fs::File>;
    let distinguisher = "instantiate_new_with_config-level-cache";
    let temp_dir = tempfile::Builder::new()
        .prefix(distinguisher)
        .tempdir()
        .expect("[test_storage_types] couldn't create temp_dir");
    run_base_tree_storage_test::<TestItemType, TestXOR128, LevelCacheStorage, U8>(
        instantiate_new_with_config,
        base_tree_leaves,
        Some(StoreConfig::new(
            temp_dir.into_path(),
            String::from(distinguisher),
            StoreConfig::default_rows_to_discard(expected_total_leaves, branches),
        )),
        LevelCacheStorage::new(1)
            .expect("[test_storage_types] couldn't instantiate LevelCacheStorage"),
    );
}

// big test moved from test_xor128.rs
#[test]
#[ignore]
fn test_large_base_trees() {
    fn run_test<BaseTreeArity: Unsigned>(
        leaves: usize,
        len: usize,
        row_count: usize,
        num_challenges: usize,
    ) {
        let big_tree =
            instantiate_new::<TestItemType, TestXOR128, VecStore<TestItemType>, BaseTreeArity>(
                leaves, None,
            );

        assert_eq!(big_tree.row_count(), row_count);
        assert_eq!(big_tree.len(), len);

        // Selectively verify that proving works.
        for i in 0..num_challenges {
            let index = i * (leaves / num_challenges);
            let proof = big_tree.gen_proof(index).expect("Failed to generate proof");
            assert!(proof
                .validate::<TestXOR128>()
                .expect("failed to validate proof"));
        }
    }

    let (leaves, len, row_count, num_challenges) = { (16777216, 19173961, 9, 1024) };
    run_test::<U8>(leaves, len, row_count, num_challenges);

    let leaves = SMALL_TREE_BUILD * 2;
    let num_challenges = SMALL_TREE_BUILD * 2;
    let branches = 2;
    run_test::<U2>(
        leaves,
        get_merkle_tree_len_generic::<U2, U0, U0>(leaves).expect("can't get tree len"),
        get_merkle_tree_row_count(leaves, branches),
        num_challenges,
    );
}
