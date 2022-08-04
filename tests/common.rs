#![cfg(not(tarpaulin_include))]
use std::fmt;
use std::hash::Hasher;
use std::io::Write;

use sha2::{Digest, Sha256};
use typenum::Unsigned;

use merkletree::hash::{Algorithm, Hashable};
use merkletree::merkle::{Element, MerkleTree};
use merkletree::store::{DiskStore, LevelCacheStore, Store, StoreConfig};

/// This is the common utilities that we use for integration tests
///
/// In order to check that particular merkle tree will work as expected we need following stuff:
///
/// - actual logic of test, that will evaluate if implemented functionality works as expected;
/// - implementation of Element, that we will use as elements of the tree while testing;
/// - implementations of Hasher and Algorithm, that we will use for computing leafs, nodes, root
/// and inclusion proofs while testing;
/// - generator of some arbitrary dataset, that can be used as a source of data for building a tree over it;
///
/// Implementation of MerkleTree abstraction is rather dense. Trees can be instantiated via 23 different
/// constructors (each constructor is part of public API), while tree can have various type (base, compound,
/// compound-compound), while each type can have arbitrary arity; additionally tree can be backed by 4 different
/// storages, each with own specifics.
///
/// Having that in mind, and considering that writing tests for all possible combination of parameters can lead to
/// huge amount of code that we need to support, we provide integration tests that cover following cases:
///
/// - testing instantiation of tree via all constructors that are part of public API;
/// - ensuring that each tree has expected amount of leaves, expected length and expected root;
/// - ensuring that inclusion proof can be successfully created and verified for each tree leaf;
///
/// What is not covered / evaluated:
///
/// - checking that arities' tests work for each base / compound / compound-compound constructor;
/// - checking that arities' tests work for Disk / Mmap / LevelCache storages;
/// - instantiation of compound tree using each base constructor;
/// - instantiation of compound-compound tree using each base and each compound constructor;
/// - instantiation of DiskStore and MmapStore compound trees;
/// - instantiation of DiskStore and MmapStore compound-compound trees;
/// - instantiation of compound tree using 'from_store_configs' with custom configurations
/// - instantiation of compound-compound tree using 'from_sub_tree_store_configs' with custom configurations
/// - instantiation of LevelCacheStore base tree using 'from_tree_slice_with_config' constructor
/// - instantiation of LevelCacheStore compound trees using "regular" compound constructors ('from_slices_with_configs', 'from_store_configs');

/// Implementation of Element abstraction that we use in our integration tests
#[derive(PartialEq, Eq, PartialOrd, Ord, Copy, Clone, Debug, Default)]
pub struct TestItem([u8; SIZE]);
pub const SIZE: usize = 0x10;

// We introduce this wrapper-type and implement Element actually for it
// just to avoid writing <item>.clone() all the time in tests
pub type TestItemType = TestItem;

impl AsRef<[u8]> for TestItem {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl Element for TestItemType {
    fn byte_len() -> usize {
        SIZE
    }
    fn from_slice(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), Self::byte_len());
        let mut el = [0u8; SIZE];
        el[..].copy_from_slice(bytes);
        TestItem(el)
    }
    fn copy_to_slice(&self, bytes: &mut [u8]) {
        bytes.copy_from_slice(&self.0);
    }
}

/// XOR128 implementation of Algorithm abstraction that we use in our integration tests
pub struct TestXOR128 {
    data: TestItem,
    i: usize,
}

impl TestXOR128 {
    pub fn new() -> TestXOR128 {
        TestXOR128 {
            data: TestItem([0u8; SIZE]),
            i: 0,
        }
    }
}

impl Hasher for TestXOR128 {
    fn finish(&self) -> u64 {
        // FIXME: contract is broken by design
        unimplemented!(
            "Hasher's contract (finish function is not used) is deliberately broken by design"
        )
    }
    fn write(&mut self, bytes: &[u8]) {
        for x in bytes {
            self.data.0[self.i & 15] ^= *x;
            self.i += 1;
        }
    }
}

impl Default for TestXOR128 {
    fn default() -> Self {
        TestXOR128::new()
    }
}

impl Algorithm<TestItem> for TestXOR128 {
    fn hash(&mut self) -> TestItem {
        self.data
    }
}

/// SHA256 implementations of Algorithm abstraction that we use in our integration tests
pub struct TestSha256Hasher {
    engine: Sha256,
}

impl TestSha256Hasher {
    pub fn new() -> TestSha256Hasher {
        TestSha256Hasher {
            engine: Sha256::new(),
        }
    }
}

impl fmt::Debug for TestSha256Hasher {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("Sha256Hasher")
    }
}

impl Default for TestSha256Hasher {
    fn default() -> Self {
        TestSha256Hasher::new()
    }
}

impl Hasher for TestSha256Hasher {
    // FIXME: contract is broken by design
    fn finish(&self) -> u64 {
        unimplemented!(
            "Hasher's contract (finish function is not used) is deliberately broken by design"
        )
    }

    fn write(&mut self, bytes: &[u8]) {
        self.engine.update(bytes)
    }
}

impl Algorithm<TestItem> for TestSha256Hasher {
    fn hash(&mut self) -> TestItem {
        let mut result = TestItem::default();
        let item_size = result.0.len();
        let hash_output = self.engine.clone().finalize().to_vec();
        self.engine.reset();
        if item_size < hash_output.len() {
            result
                .0
                .copy_from_slice(&hash_output.as_slice()[0..item_size]);
        } else {
            result.0.copy_from_slice(hash_output.as_slice())
        }
        result
    }
}

/// Dataset generators
///
/// We need to provide 3 different datasets:
/// - <I: IntoIterator<Item = E>>
/// - <O: Hashable<A>, I: IntoIterator<Item = O>>
/// - datasets based on raw serialization to byte-slice;
///
/// because various MerkleTree constructors have specific requirements

// generate dataset of iterable elements
pub fn generate_vector_of_elements<E: Element>(leaves: usize) -> Vec<E> {
    let result = (0..leaves).map(|index| {
        // we are ok with usize -> u8 conversion problems, since we need just predictable dataset
        let vector: Vec<u8> = (0..E::byte_len()).map(|x| (index + x) as u8).collect();
        E::from_slice(vector.as_slice())
    });
    result.collect()
}

// generate dataset of iterable and hashable elements
pub fn generate_vector_of_usizes(leaves: usize) -> Vec<usize> {
    (0..leaves).map(|index| index * 93).collect()
}

// generate dataset of hashable (usize) elements and serialize it at once
pub fn generate_byte_slice_tree<E: Element, A: Algorithm<E>>(leaves: usize) -> Vec<u8> {
    let mut a = A::default();
    let mut a2 = A::default();

    let dataset: Vec<u8> = generate_vector_of_usizes(leaves)
        .iter()
        .map(|x| {
            a.reset();
            x.hash(&mut a);
            a.hash()
        })
        .take(leaves)
        .flat_map(|item| {
            a2.reset();
            a2.leaf(item).as_ref().to_vec()
        })
        .collect();

    dataset
}

/// Actual tests
pub fn test_disk_mmap_vec_tree_functionality<
    E: Element,
    A: Algorithm<E>,
    S: Store<E>,
    BaseTreeArity: Unsigned,
    SubTreeArity: Unsigned,
    TopTreeArity: Unsigned,
>(
    tree: MerkleTree<E, A, S, BaseTreeArity, SubTreeArity, TopTreeArity>,
    expected_leaves: usize,
    expected_len: usize,
    expected_root: E,
) {
    assert_eq!(tree.leafs(), expected_leaves);
    assert_eq!(tree.len(), expected_len);
    assert_eq!(tree.root(), expected_root);

    for index in 0..tree.leafs() {
        let p = tree.gen_proof(index).unwrap();
        assert!(p.validate::<A>().expect("failed to validate"));
    }
}

pub fn test_levelcache_tree_functionality<
    E: Element,
    A: Algorithm<E>,
    BaseTreeArity: Unsigned,
    SubTreeArity: Unsigned,
    TopTreeArity: Unsigned,
>(
    tree: MerkleTree<
        E,
        A,
        LevelCacheStore<E, std::fs::File>,
        BaseTreeArity,
        SubTreeArity,
        TopTreeArity,
    >,
    rows_to_discard: Option<usize>,
    expected_leaves: usize,
    expected_len: usize,
    expected_root: E,
) {
    assert_eq!(tree.leafs(), expected_leaves);
    assert_eq!(tree.len(), expected_len);
    assert_eq!(tree.root(), expected_root);

    for index in 0..tree.leafs() {
        let p = tree.gen_cached_proof(index, rows_to_discard).unwrap();
        assert!(p.validate::<A>().expect("failed to validate"));
    }
}

/// Utilities
pub fn serialize_tree<E: Element, A: Algorithm<E>, S: Store<E>, U: Unsigned>(
    tree: MerkleTree<E, A, S, U>,
) -> Vec<u8> {
    let data = tree.data().expect("can't get tree's data [serialize_tree]");
    let data: Vec<E> = data
        .read_range(0..data.len())
        .expect("can't read actual data [serialize_tree]");
    let mut serialized_tree = vec![0u8; E::byte_len() * data.len()];
    let mut start = 0;
    let mut end = E::byte_len();
    for element in data {
        element.copy_to_slice(&mut serialized_tree[start..end]);
        start += E::byte_len();
        end += E::byte_len();
    }
    serialized_tree
}

pub fn instantiate_new<E: Element, A: Algorithm<E>, S: Store<E>, U: Unsigned>(
    leaves: usize,
    _config: Option<StoreConfig>,
) -> MerkleTree<E, A, S, U> {
    let dataset = generate_vector_of_elements::<E>(leaves);
    MerkleTree::new(dataset).expect("failed to instantiate tree [new]")
}

pub fn instantiate_new_with_config<E: Element, A: Algorithm<E>, S: Store<E>, U: Unsigned>(
    leaves: usize,
    config: Option<StoreConfig>,
) -> MerkleTree<E, A, S, U> {
    let dataset = generate_vector_of_elements::<E>(leaves);
    MerkleTree::new_with_config(
        dataset,
        config.expect("can't get tree's config [new_with_config]"),
    )
    .expect("failed to instantiate tree [new_with_config]")
}

pub fn dump_tree_data_to_replica<E: Element, BaseTreeArity: Unsigned>(
    leaves: usize,
    len: usize,
    config: &StoreConfig,
    replica_file: &mut std::fs::File,
) {
    // Dump tree data to disk
    let store = DiskStore::new_with_config(len, BaseTreeArity::to_usize(), config.clone())
        .expect("failed to open store [dump_tree_data_to_replica]");

    // Use that data store as the replica (concat the data to the replica_path)
    let data: Vec<E> = store
        .read_range(std::ops::Range {
            start: 0,
            end: leaves,
        })
        .expect("failed to read store [dump_tree_data_to_replica]");
    for element in data {
        let mut vector = vec![0u8; E::byte_len()];
        element.copy_to_slice(vector.as_mut_slice());
        replica_file
            .write_all(vector.as_slice())
            .expect("failed to write replica data [dump_tree_data_to_replica]");
    }
}

pub fn get_vector_of_base_trees_as_slices<
    E: Element,
    A: Algorithm<E>,
    S: Store<E>,
    BaseTreeArity: Unsigned,
    SubTreeArity: Unsigned,
>(
    base_tree_leaves: usize,
) -> Vec<Vec<u8>> {
    (0..SubTreeArity::to_usize())
        .map(|_| {
            let base_tree = instantiate_new::<E, A, S, BaseTreeArity>(base_tree_leaves, None);
            serialize_tree(base_tree)
        })
        .collect()
}

pub fn get_vector_of_base_trees<
    E: Element,
    A: Algorithm<E>,
    S: Store<E>,
    BaseTreeArity: Unsigned,
    SubTreeArity: Unsigned,
>(
    base_tree_leaves: usize,
) -> Vec<MerkleTree<E, A, S, BaseTreeArity>> {
    (0..SubTreeArity::to_usize())
        .map(|_| instantiate_new(base_tree_leaves, None))
        .collect()
}
