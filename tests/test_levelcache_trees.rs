#![cfg(not(tarpaulin_include))]

/// LevelCacheStore trees have significant differences, so we use separate integration tests
/// for their evaluation. Fortunately, only 'with-config' constructors (and couple of specific
/// replica-constructors) can be used for instantiation of LevelCacheStore trees, so we group
/// tests for base, compound and compound-compound trees into single suite.
///
/// Ultimately following constructors are covered:
/// [base trees]
/// - new_with_config
/// - try_from_iter_with_config
/// - from_par_iter_with_config
/// - from_data_with_config
/// - from_byte_slice_with_config
/// - from_tree_slice_with_config
/// [compound trees]
/// - from_store_configs_and_replica
/// [compound-compound trees]
/// - from_sub_tree_store_configs_and_replica
///
/// Each instantiation of LevelCacheStore tree requires preparing valid configuration and replica file
/// that point to actual tree data stored in filesystem, so most of the instantiators' logic is preparing
/// those items and dumping data to filesystem
pub mod common;

use std::path::Path;

use merkletree::hash::Algorithm;
use merkletree::merkle::{
    get_merkle_tree_len_generic, Element, FromIndexedParallelIterator, MerkleTree,
};
use merkletree::store::{DiskStore, LevelCacheStore, ReplicaConfig, StoreConfig, VecStore};
use typenum::{Unsigned, U0, U2, U8};

use crate::common::{
    dump_tree_data_to_replica, generate_byte_slice_tree, generate_vector_of_elements,
    generate_vector_of_usizes, instantiate_new_with_config, serialize_tree,
    test_levelcache_tree_functionality, TestItem, TestItemType, TestSha256Hasher, TestXOR128,
};

/// LevelCacheStore constructors of base trees
fn lc_instantiate_new_with_config<E: Element, A: Algorithm<E>, BaseTreeArity: Unsigned>(
    leaves: usize,
    temp_dir_path: &Path,
    rows_to_discard: usize,
) -> MerkleTree<E, A, LevelCacheStore<E, std::fs::File>, BaseTreeArity> {
    let dataset = generate_vector_of_elements::<E>(leaves);

    let replica_path = StoreConfig::data_path(temp_dir_path, "replica_path");
    let mut replica_file = std::fs::File::create(&replica_path)
        .expect("failed to create replica file [lc_instantiate_new_with_config]");

    // prepare replica file content
    let config = StoreConfig::new(
        temp_dir_path,
        "lc_instantiate_new_with_config",
        rows_to_discard,
    );

    // instantiation of this temp tree is required for binding config to actual file on disk for subsequent dumping the data to replica
    let tree = MerkleTree::<E, A, DiskStore<E>, BaseTreeArity>::new_with_config(
        dataset.clone(),
        config.clone(),
    )
    .expect("failed to instantiate tree [lc_instantiate_new_with_config]");

    dump_tree_data_to_replica::<E, BaseTreeArity>(
        tree.leafs(),
        tree.len(),
        &config,
        &mut replica_file,
    );

    // generate LC tree from dumped data
    let lc_config = StoreConfig::from_config(
        &config,
        format!("{}-{}", "lc_instantiate_new_with_config", "lc"),
        Some(tree.len()),
    );
    let mut tree =
        MerkleTree::<E, A, LevelCacheStore<E, std::fs::File>, BaseTreeArity>::new_with_config(
            dataset, lc_config,
        )
        .expect("failed to instantiate LC tree [lc_instantiate_new_with_config]");
    tree.set_external_reader_path(&replica_path)
        .expect("can't set external reader path [lc_instantiate_new_with_config]");
    tree
}

fn lc_instantiate_try_from_iter_with_config<
    E: Element,
    A: Algorithm<E>,
    BaseTreeArity: Unsigned,
>(
    leaves: usize,
    temp_dir_path: &Path,
    rows_to_discard: usize,
) -> MerkleTree<E, A, LevelCacheStore<E, std::fs::File>, BaseTreeArity> {
    let dataset = generate_vector_of_elements::<E>(leaves);

    let replica_path = StoreConfig::data_path(temp_dir_path, "replica_path");
    let mut replica_file = std::fs::File::create(&replica_path)
        .expect("failed to create replica file [lc_instantiate_try_from_iter_with_config]");

    // prepare replica file content
    let config = StoreConfig::new(
        temp_dir_path,
        "lc_instantiate_try_from_iter_with_config",
        rows_to_discard,
    );

    // instantiation of this temp tree is required for binding config to actual file on disk for subsequent dumping the data to replica
    let tree = MerkleTree::<E, A, DiskStore<E>, BaseTreeArity>::try_from_iter_with_config(
        dataset.clone().into_iter().map(Ok),
        config.clone(),
    )
    .expect("failed to instantiate tree [lc_instantiate_try_from_iter_with_config]");

    dump_tree_data_to_replica::<E, BaseTreeArity>(
        tree.leafs(),
        tree.len(),
        &config,
        &mut replica_file,
    );

    // generate LC tree from dumped data
    let lc_config = StoreConfig::from_config(
        &config,
        format!("{}-{}", "lc_instantiate_try_from_iter_with_config", "lc"),
        Some(tree.len()),
    );
    let mut tree = MerkleTree::<E, A, LevelCacheStore<E, std::fs::File>, BaseTreeArity>::try_from_iter_with_config(dataset.into_iter().map(Ok), lc_config).expect("failed to instantiate LC tree [lc_instantiate_try_from_iter_with_config]");
    tree.set_external_reader_path(&replica_path)
        .expect("can't set external reader path [lc_instantiate_try_from_iter_with_config]");
    tree
}

fn lc_instantiate_from_par_iter_with_config<
    E: Element,
    A: Algorithm<E>,
    BaseTreeArity: Unsigned,
>(
    leaves: usize,
    temp_dir_path: &Path,
    rows_to_discard: usize,
) -> MerkleTree<E, A, LevelCacheStore<E, std::fs::File>, BaseTreeArity> {
    let dataset = generate_vector_of_elements::<E>(leaves);

    let replica_path = StoreConfig::data_path(temp_dir_path, "replica_path");
    let mut replica_file = std::fs::File::create(&replica_path)
        .expect("failed to create replica file [lc_instantiate_from_par_iter_with_config]");

    // prepare replica file content
    let config = StoreConfig::new(
        temp_dir_path,
        "lc_instantiate_from_par_iter_with_config",
        rows_to_discard,
    );

    // instantiation of this temp tree is required for binding config to actual file on disk for subsequent dumping the data to replica
    let tree = MerkleTree::<E, A, DiskStore<E>, BaseTreeArity>::from_par_iter_with_config(
        dataset.clone(),
        config.clone(),
    )
    .expect("failed to instantiate tree [lc_instantiate_from_par_iter_with_config]");

    dump_tree_data_to_replica::<E, BaseTreeArity>(
        tree.leafs(),
        tree.len(),
        &config,
        &mut replica_file,
    );

    // generate LC tree from dumped data
    let lc_config = StoreConfig::from_config(
        &config,
        format!("{}-{}", "lc_instantiate_from_par_iter_with_config", "lc"),
        Some(tree.len()),
    );
    let mut tree = MerkleTree::<E, A, LevelCacheStore<E, std::fs::File>, BaseTreeArity>::from_par_iter_with_config(dataset, lc_config).expect("failed to instantiate LC tree [lc_instantiate_from_par_iter_with_config]");
    tree.set_external_reader_path(&replica_path)
        .expect("can't set external reader path [lc_instantiate_from_par_iter_with_config]");
    tree
}

fn lc_instantiate_from_data_with_config<E: Element, A: Algorithm<E>, BaseTreeArity: Unsigned>(
    leaves: usize,
    temp_dir_path: &Path,
    rows_to_discard: usize,
) -> MerkleTree<E, A, LevelCacheStore<E, std::fs::File>, BaseTreeArity> {
    let dataset = generate_vector_of_usizes(leaves);

    let replica_path = StoreConfig::data_path(temp_dir_path, "replica_path");
    let mut replica_file = std::fs::File::create(&replica_path)
        .expect("failed to create replica file [lc_instantiate_from_data_with_config]");

    // prepare replica file content
    let config = StoreConfig::new(
        temp_dir_path,
        "lc_instantiate_from_data_with_config",
        rows_to_discard,
    );

    // instantiation of this temp tree is required for binding config to actual file on disk for subsequent dumping the data to replica
    let tree = MerkleTree::<E, A, DiskStore<E>, BaseTreeArity>::from_data_with_config(
        dataset.as_slice(),
        config.clone(),
    )
    .expect("failed to instantiate tree [lc_instantiate_from_data_with_config]");

    dump_tree_data_to_replica::<E, BaseTreeArity>(
        tree.leafs(),
        tree.len(),
        &config,
        &mut replica_file,
    );

    // generate LC tree from dumped data
    let lc_config = StoreConfig::from_config(
        &config,
        format!("{}-{}", "lc_instantiate_from_data_with_config", "lc"),
        Some(tree.len()),
    );
    let mut tree = MerkleTree::<E, A, LevelCacheStore<E, std::fs::File>, BaseTreeArity>::from_data_with_config(dataset.as_slice(), lc_config).expect("failed to instantiate LC tree [lc_instantiate_from_data_with_config]");
    tree.set_external_reader_path(&replica_path)
        .expect("can't set external reader path [lc_instantiate_from_data_with_config]");
    tree
}

fn lc_instantiate_from_byte_slice_with_config<
    E: Element,
    A: Algorithm<E>,
    BaseTreeArity: Unsigned,
>(
    leaves: usize,
    temp_dir_path: &Path,
    rows_to_discard: usize,
) -> MerkleTree<E, A, LevelCacheStore<E, std::fs::File>, BaseTreeArity> {
    let dataset = generate_byte_slice_tree::<E, A>(leaves);

    let replica_path = StoreConfig::data_path(temp_dir_path, "replica_path");
    let mut replica_file = std::fs::File::create(&replica_path)
        .expect("failed to create replica file [lc_instantiate_from_byte_slice_with_config]");

    // prepare replica file content
    let config = StoreConfig::new(
        temp_dir_path,
        "lc_instantiate_from_byte_slice_with_config",
        rows_to_discard,
    );

    // instantiation of this temp tree is required for binding config to actual file on disk for subsequent dumping the data to replica
    let tree = MerkleTree::<E, A, DiskStore<E>, BaseTreeArity>::from_byte_slice_with_config(
        dataset.as_slice(),
        config.clone(),
    )
    .expect("failed to instantiate tree [lc_instantiate_from_byte_slice_with_config]");

    dump_tree_data_to_replica::<E, BaseTreeArity>(
        tree.leafs(),
        tree.len(),
        &config,
        &mut replica_file,
    );

    // generate LC tree from dumped data
    let lc_config = StoreConfig::from_config(
        &config,
        format!("{}-{}", "lc_instantiate_from_byte_slice_with_config", "lc"),
        Some(tree.len()),
    );
    let mut tree = MerkleTree::<E, A, LevelCacheStore<E, std::fs::File>, BaseTreeArity>::from_byte_slice_with_config(dataset.as_slice(), lc_config).expect("failed to instantiate LC tree [lc_instantiate_from_byte_slice_with_config]");
    tree.set_external_reader_path(&replica_path)
        .expect("can't set external reader path [lc_instantiate_from_byte_slice_with_config]");
    tree
}

#[allow(dead_code)]
fn lc_instantiate_from_tree_slice_with_config<
    E: Element,
    A: Algorithm<E>,
    BaseTreeArity: Unsigned,
>(
    leaves: usize,
    temp_dir_path: &Path,
    rows_to_discard: usize,
) -> MerkleTree<E, A, LevelCacheStore<E, std::fs::File>, BaseTreeArity> {
    let dataset = generate_vector_of_usizes(leaves);
    let tmp_tree = MerkleTree::<E, A, VecStore<E>, BaseTreeArity>::from_data(dataset.as_slice())
        .expect("failed to instantiate tree [lc_instantiate_from_tree_slice_with_config]");
    let dataset = serialize_tree(tmp_tree);

    let replica_path = StoreConfig::data_path(temp_dir_path, "replica_path");
    let mut replica_file = std::fs::File::create(&replica_path)
        .expect("failed to create replica file [lc_instantiate_from_tree_slice_with_config]");

    // prepare replica file content
    let config = StoreConfig::new(
        temp_dir_path,
        "lc_instantiate_from_tree_slice_with_config",
        rows_to_discard,
    );

    // instantiation of this temp tree is required for binding config to actual file on disk for subsequent dumping the data to replica
    let tree = MerkleTree::<E, A, DiskStore<E>, BaseTreeArity>::from_tree_slice_with_config(
        dataset.as_slice(),
        leaves,
        config.clone(),
    )
    .expect("failed to instantiate tree [lc_instantiate_from_tree_slice_with_config]");

    dump_tree_data_to_replica::<E, BaseTreeArity>(
        tree.leafs(),
        tree.len(),
        &config,
        &mut replica_file,
    );

    // generate LC tree from dumped data
    let lc_config = StoreConfig::from_config(
        &config,
        format!("{}-{}", "lc_instantiate_from_tree_slice_with_config", "lc"),
        Some(tree.len()),
    );
    let mut tree = MerkleTree::<E, A, LevelCacheStore<E, std::fs::File>, BaseTreeArity>::from_tree_slice_with_config(dataset.as_slice(), leaves, lc_config).expect("failed to instantiate LC tree [lc_instantiate_from_tree_slice_with_config]");
    tree.set_external_reader_path(&replica_path)
        .expect("can't set external reader path [lc_instantiate_from_tree_slice_with_config]");
    tree
}

/// Test executor
#[allow(clippy::type_complexity)]
fn run_test_base_lc_tree<E: Element, A: Algorithm<E>, BaseTreeArity: Unsigned>(
    constructor: fn(
        usize,
        &Path,
        usize,
    ) -> MerkleTree<E, A, LevelCacheStore<E, std::fs::File>, BaseTreeArity>,
    leaves_in_tree: usize,
    temp_dir_path: &Path,
    rows_to_discard: usize,
    expected_leaves: usize,
    expected_len: usize,
    expected_root: E,
) {
    // base tree has SubTreeArity and TopTreeArity parameters equal to zero
    let tree: MerkleTree<E, A, LevelCacheStore<E, std::fs::File>, BaseTreeArity> =
        constructor(leaves_in_tree, temp_dir_path, rows_to_discard);
    test_levelcache_tree_functionality(
        tree,
        Some(rows_to_discard),
        expected_leaves,
        expected_len,
        expected_root,
    );
}

/// LevelCache (base) trees can be instantiated only with constructors that take
/// valid configuration as input parameter, so we use only 'with_config' constructors
/// for testing. Again dataset is critical for root computation, so we use distinct
/// integration tests for different datasets
#[test]
fn test_base_levelcache_trees_iterable() {
    fn run_tests<E: Element + Copy, A: Algorithm<E>>(root: E) {
        let base_tree_leaves = 64;
        let expected_total_leaves = base_tree_leaves;
        let len = get_merkle_tree_len_generic::<U8, U0, U0>(base_tree_leaves).unwrap();
        let rows_to_discard = 0;

        let distinguisher = "lc_instantiate_new_with_config";
        let temp_dir = tempdir::TempDir::new(distinguisher).unwrap();
        run_test_base_lc_tree::<E, A, U8>(
            lc_instantiate_new_with_config,
            base_tree_leaves,
            temp_dir.path(),
            rows_to_discard,
            expected_total_leaves,
            len,
            root,
        );

        let distinguisher = "lc_instantiate_try_from_iter_with_config";
        let temp_dir = tempdir::TempDir::new(distinguisher).unwrap();
        run_test_base_lc_tree::<E, A, U8>(
            lc_instantiate_try_from_iter_with_config,
            base_tree_leaves,
            temp_dir.path(),
            rows_to_discard,
            expected_total_leaves,
            len,
            root,
        );

        let distinguisher = "lc_instantiate_from_par_iter_with_config";
        let temp_dir = tempdir::TempDir::new(distinguisher).unwrap();
        run_test_base_lc_tree::<E, A, U8>(
            lc_instantiate_from_par_iter_with_config,
            base_tree_leaves,
            temp_dir.path(),
            rows_to_discard,
            expected_total_leaves,
            len,
            root,
        );
    }

    let root_xor128 =
        TestItemType::from_slice(&[65, 0, 64, 0, 64, 0, 64, 0, 64, 0, 64, 0, 64, 0, 64, 0]);
    run_tests::<TestItemType, TestXOR128>(root_xor128);

    let root_sha256 = TestItemType::from_slice(&[
        252, 61, 163, 229, 140, 223, 198, 165, 200, 137, 59, 43, 83, 136, 197, 63,
    ]);
    run_tests::<TestItemType, TestSha256Hasher>(root_sha256);
}

#[test]
fn test_base_levelcache_trees_iterable_hashable_and_serialization() {
    fn run_tests<E: Element + Copy, A: Algorithm<E>>(root: E) {
        let base_tree_leaves = 64;
        let expected_total_leaves = base_tree_leaves;
        let len = get_merkle_tree_len_generic::<U8, U0, U0>(base_tree_leaves).unwrap();
        let rows_to_discard = 0;

        let distinguisher = "lc_instantiate_from_data_with_config";
        let temp_dir = tempdir::TempDir::new(distinguisher).unwrap();
        run_test_base_lc_tree::<E, A, U8>(
            lc_instantiate_from_data_with_config,
            base_tree_leaves,
            temp_dir.path(),
            rows_to_discard,
            expected_total_leaves,
            len,
            root,
        );

        let distinguisher = "lc_instantiate_from_byte_slice_with_config";
        let temp_dir = tempdir::TempDir::new(distinguisher).unwrap();
        run_test_base_lc_tree::<E, A, U8>(
            lc_instantiate_from_byte_slice_with_config,
            base_tree_leaves,
            temp_dir.path(),
            rows_to_discard,
            expected_total_leaves,
            len,
            root,
        );

        /* TODO investigate, why this test fails
        let distinguisher = "lc_instantiate_from_tree_slice_with_config";
        let temp_dir = tempdir::TempDir::new(distinguisher).unwrap();
        run_test_base_lc_tree::<E, A, U8>(
            lc_instantiate_from_tree_slice_with_config,
            base_tree_leaves,
            temp_dir.path(),
            rows_to_discard,
            expected_total_leaves,
            len,
            root,
        );
        */
    }

    let root_xor128 = TestItemType::from_slice(&[1, 0, 0, 0, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    run_tests::<TestItemType, TestXOR128>(root_xor128);

    let root_sha256 = TestItemType::from_slice(&[
        98, 103, 202, 101, 121, 179, 6, 237, 133, 39, 253, 169, 173, 63, 89, 188,
    ]);
    run_tests::<TestItemType, TestSha256Hasher>(root_sha256);
}

/// Compound LevelCacheStore trees can be instantiated via specific 'from_store_configs_and_replica' constructor
fn lc_instantiate_ctree_from_store_configs_and_replica<
    E: Element,
    A: Algorithm<E>,
    BaseTreeArity: Unsigned,
    SubTreeArity: Unsigned,
>(
    base_tree_leaves: usize,
    temp_dir_path: &Path,
    rows_to_discard: Option<usize>,
) -> MerkleTree<E, A, LevelCacheStore<E, std::fs::File>, BaseTreeArity, SubTreeArity> {
    let replica_path = StoreConfig::data_path(temp_dir_path, "replica_path");
    let mut replica_file = std::fs::File::create(&replica_path).expect(
        "failed to create replica file [lc_instantiate_ctree_from_store_configs_and_replica]",
    );

    let sub_tree_arity = SubTreeArity::to_usize();

    let offsets = (0..sub_tree_arity)
        .map(|index| index * E::byte_len() * base_tree_leaves)
        .collect();

    let configs = (0..sub_tree_arity)
        .map(|index| {
            // prepare replica file content
            let config = StoreConfig::new(
                temp_dir_path,
                format!("config_id{}", index),
                rows_to_discard
                    .expect("can't get rows_to_discard [lc_instantiate_ctree_from_store_configs_and_replica]"),
            );
            // instantiation of this temp tree is required for binding config to actual file on disk for subsequent dumping the data to replica
            let tree = instantiate_new_with_config::<E, A, DiskStore<E>, BaseTreeArity>(
                base_tree_leaves,
                Some(config.clone()),
            );
            dump_tree_data_to_replica::<E, BaseTreeArity>(
                tree.leafs(),
                tree.len(),
                &config,
                &mut replica_file,
            );

            // generate valid configs and bind them each to actual data of the tree
            let lc_config = StoreConfig::from_config(
                &config,
                format!("lc_config_id{}", index),
                Some(tree.len()),
            );
            let mut tree = instantiate_new_with_config::<
                E,
                A,
                LevelCacheStore<E, std::fs::File>,
                BaseTreeArity,
            >(base_tree_leaves, Some(lc_config.clone()));
            tree.set_external_reader_path(&replica_path)
                .expect("can't set external reader path [lc_instantiate_ctree_from_store_configs_and_replica]");
            lc_config
        })
        .collect::<Vec<StoreConfig>>();

    MerkleTree::from_store_configs_and_replica(
        base_tree_leaves,
        &configs,
        &ReplicaConfig::new(&replica_path, offsets),
    )
    .expect(
        "failed to instantiate compound tree [lc_instantiate_ctree_from_store_configs_and_replica]",
    )
}

/// We don't add generic test runner function, since we test only single specific constructor of compound tree.
/// If more constructors appear, then creating and using runner would be beneficial
#[test]
fn test_compound_levelcache_trees() {
    fn run_tests<E: Element + Copy, A: Algorithm<E>>(root: E) {
        let base_tree_leaves = 64;
        let expected_total_leaves = base_tree_leaves * 8;
        let len = get_merkle_tree_len_generic::<U8, U8, U0>(base_tree_leaves).unwrap();

        let distinguisher = "instantiate_cctree_from_sub_tree_store_configs_and_replica";
        let temp_dir = tempdir::TempDir::new(distinguisher).unwrap();
        let rows_to_discard = 0;
        let tree = lc_instantiate_ctree_from_store_configs_and_replica::<E, A, U8, U8>(
            base_tree_leaves,
            temp_dir.path(),
            Some(rows_to_discard),
        );

        test_levelcache_tree_functionality::<E, A, U8, U8, U0>(
            tree,
            Some(rows_to_discard),
            expected_total_leaves,
            len,
            root,
        );
    }

    let root_xor128 = TestItem::from_slice(&[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    run_tests::<TestItemType, TestXOR128>(root_xor128);

    let root_sha256 = TestItem::from_slice(&[
        32, 129, 168, 134, 58, 233, 155, 225, 88, 230, 247, 63, 18, 38, 194, 230,
    ]);
    run_tests::<TestItemType, TestSha256Hasher>(root_sha256);
}

/// Compound-compound LevelCacheStore tree can be instantiated via specific 'from_sub_tree_store_configs_and_replica' constructor
fn lc_instantiate_cctree_from_sub_tree_store_configs_and_replica<
    E: Element,
    A: Algorithm<E>,
    BaseTreeArity: Unsigned,
    SubTreeArity: Unsigned,
    TopTreeArity: Unsigned,
>(
    base_tree_leaves: usize,
    temp_dir_path: &Path,
    rows_to_discard: Option<usize>,
) -> MerkleTree<E, A, LevelCacheStore<E, std::fs::File>, BaseTreeArity, SubTreeArity, TopTreeArity>
{
    let replica_path = StoreConfig::data_path(temp_dir_path, "replica_path");
    let mut replica_file = std::fs::File::create(&replica_path)
        .expect("failed to create replica file [lc_instantiate_cctree_from_sub_tree_store_configs_and_replica]");

    let sub_tree_arity = SubTreeArity::to_usize();
    let top_tree_arity = TopTreeArity::to_usize();

    let offsets = (0..sub_tree_arity * top_tree_arity)
        .map(|index| index * E::byte_len() * base_tree_leaves)
        .collect();

    let configs = (0..TopTreeArity::to_usize())
        .flat_map(|j| {
            (0..SubTreeArity::to_usize())
                .map(|i| {
                    // prepare replica file content
                    let config = StoreConfig::new(
                        temp_dir_path,
                        format!(
                            "config_id{}{}",
                            i,
                            j
                        ),
                        rows_to_discard.expect(
                            "can't get rows_to_discard [lc_instantiate_cctree_from_sub_tree_store_configs_and_replica]",
                        ),
                    );
                    // instantiation of this temp tree is required for binding config to actual file on disk for subsequent dumping the data to replica
                    let tree =
                        instantiate_new_with_config::<E, A, DiskStore<E>, BaseTreeArity>(
                            base_tree_leaves,
                            Some(config.clone()),
                        );
                    dump_tree_data_to_replica::<E, BaseTreeArity>(
                        tree.leafs(),
                        tree.len(),
                        &config,
                        &mut replica_file,
                    );

                    // generate valid configs and bind them each to actual data of the tree
                    let lc_config = StoreConfig::from_config(
                        &config,
                        format!(
                            "lc_config_id{}{}",
                            i,
                            j
                        ),
                        Some(tree.len()),
                    );
                    let mut tree = instantiate_new_with_config::<
                        E,
                        A,
                        LevelCacheStore<E, std::fs::File>,
                        BaseTreeArity,
                    >(base_tree_leaves, Some(lc_config.clone()));
                    tree.set_external_reader_path(&replica_path).expect(
                        "can't set external reader path [lc_instantiate_cctree_from_sub_tree_store_configs_and_replica]",
                    );
                    lc_config
                })
                .collect::<Vec<StoreConfig>>()
        })
        .collect::<Vec<StoreConfig>>();

    MerkleTree::from_sub_tree_store_configs_and_replica(
        base_tree_leaves,
        &configs,
        &ReplicaConfig::new(&replica_path, offsets),
    )
        .expect(
            "failed to instantiate compound-compound tree [lc_instantiate_cctree_from_sub_tree_store_configs_and_replica]",
        )
}

/// We don't add generic test runner function, since we test only single specific constructor of compound-compound tree.
/// If more constructors appear, then creating and using runner would be beneficial
#[test]
fn test_compound_compound_levelcache_trees() {
    fn run_tests<E: Element + Copy, A: Algorithm<E>>(root: E) {
        let base_tree_leaves = 64;
        let expected_total_leaves = base_tree_leaves * 8 * 2;
        let len = get_merkle_tree_len_generic::<U8, U8, U2>(base_tree_leaves).unwrap();

        let distinguisher = "instantiate_cctree_from_sub_tree_store_configs_and_replica";
        let temp_dir = tempdir::TempDir::new(distinguisher).unwrap();
        let rows_to_discard = 0;
        let tree = lc_instantiate_cctree_from_sub_tree_store_configs_and_replica::<E, A, U8, U8, U2>(
            base_tree_leaves,
            temp_dir.path(),
            Some(rows_to_discard),
        );

        test_levelcache_tree_functionality::<E, A, U8, U8, U2>(
            tree,
            Some(rows_to_discard),
            expected_total_leaves,
            len,
            root,
        );
    }

    let root_xor128 = TestItem::from_slice(&[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    run_tests::<TestItemType, TestXOR128>(root_xor128);

    let root_sha256 = TestItem::from_slice(&[
        52, 152, 123, 224, 174, 42, 152, 12, 199, 4, 105, 245, 176, 59, 230, 86,
    ]);
    run_tests::<TestItemType, TestSha256Hasher>(root_sha256);
}
