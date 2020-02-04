#![cfg(test)]

use crate::compound_merkle::CompoundMerkleTree;
use crate::compound_proof::CompoundProof;
use crate::hash::*;
use crate::merkle::FromIndexedParallelIterator;
use crate::merkle::{
    get_merkle_tree_height, get_merkle_tree_len, log2_pow2, next_pow2, Element, MerkleTree,
};
use crate::store::{
    DiskStore, DiskStoreProducer, ExternalReader, LevelCacheStore, MmapStore, Store, StoreConfig,
    StoreConfigDataVersion, VecStore, SMALL_TREE_BUILD,
};
use rayon::iter::{plumbing::*, IntoParallelIterator, ParallelIterator};
use std::fmt;
use std::fs::OpenOptions;
use std::hash::Hasher;
use std::os::unix::prelude::FileExt;
use typenum::marker_traits::Unsigned;
use typenum::{U2, U3, U4, U5, U7, U8};

const SIZE: usize = 0x10;
const DEFAULT_NUM_BRANCHES: usize = 2;

type Item = [u8; SIZE];

#[derive(Debug, Copy, Clone, Default)]
struct XOR128 {
    data: Item,
    i: usize,
}

impl XOR128 {
    fn new() -> XOR128 {
        XOR128 {
            data: [0; SIZE],
            i: 0,
        }
    }
}

impl Hasher for XOR128 {
    fn write(&mut self, bytes: &[u8]) {
        for x in bytes {
            self.data[self.i & (SIZE - 1)] ^= *x;
            self.i += 1;
        }
    }

    fn finish(&self) -> u64 {
        unimplemented!()
    }
}

impl Algorithm<Item> for XOR128 {
    #[inline]
    fn hash(&mut self) -> [u8; 16] {
        self.data
    }

    #[inline]
    fn reset(&mut self) {
        *self = XOR128::new();
    }
}

impl fmt::UpperHex for XOR128 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if f.alternate() {
            if let Err(e) = f.write_str("0x") {
                return Err(e);
            }
        }
        for b in self.data.as_ref() {
            if let Err(e) = write!(f, "{:02X}", b) {
                return Err(e);
            }
        }
        Ok(())
    }
}

impl Element for [u8; 16] {
    fn byte_len() -> usize {
        16
    }

    fn from_slice(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), Self::byte_len());
        let mut el = [0u8; 16];
        el[..].copy_from_slice(bytes);
        el
    }

    fn copy_to_slice(&self, bytes: &mut [u8]) {
        bytes.copy_from_slice(self);
    }
}

fn get_vec_tree_from_slice<U: Unsigned>(
    leafs: usize,
) -> MerkleTree<[u8; 16], XOR128, VecStore<[u8; 16]>, U> {
    let mut x = Vec::with_capacity(leafs);
    for i in 0..leafs {
        x.push(i * 93);
    }
    MerkleTree::from_data(&x).expect("failed to create tree from slice")
}

fn test_vec_tree_from_slice<U: Unsigned>(
    leafs: usize,
    len: usize,
    height: usize,
    num_challenges: usize,
) {
    let mut x = [0; 16];
    for i in 0..leafs {
        x[i] = i * 93;
    }
    let mt: MerkleTree<[u8; 16], XOR128, VecStore<_>, U> =
        MerkleTree::from_data(&x).expect("failed to create tree from slice");
    assert_eq!(mt.len(), len);
    assert_eq!(mt.leafs(), leafs);
    assert_eq!(mt.height(), height);

    for i in 0..num_challenges {
        let index = i * (leafs / num_challenges);
        let p = mt.gen_proof(index).unwrap();
        assert!(p.validate::<XOR128>());
    }
}

fn test_vec_tree_from_iter<U: Unsigned>(
    leafs: usize,
    len: usize,
    height: usize,
    num_challenges: usize,
) {
    let branches = <U as Unsigned>::to_usize();
    assert_eq!(len, get_merkle_tree_len(leafs, branches));
    assert_eq!(height, get_merkle_tree_height(leafs, branches));

    let mut a = XOR128::new();
    let mt: MerkleTree<[u8; 16], XOR128, VecStore<_>, U> =
        MerkleTree::try_from_iter((0..leafs).map(|x| {
            a.reset();
            (x * 3).hash(&mut a);
            leafs.hash(&mut a);
            Ok(a.hash())
        }))
        .expect("failed to create octree from iter");

    assert_eq!(mt.len(), len);
    assert_eq!(mt.leafs(), leafs);
    assert_eq!(mt.height(), height);

    for i in 0..num_challenges {
        let index = i * (leafs / num_challenges);
        let p = mt.gen_proof(index).unwrap();
        assert!(p.validate::<XOR128>());
    }
}

fn get_disk_tree_from_slice<U: Unsigned>(
    leafs: usize,
    config: StoreConfig,
) -> MerkleTree<[u8; 16], XOR128, DiskStore<[u8; 16]>, U> {
    let mut x = Vec::with_capacity(leafs);
    for i in 0..leafs {
        x.push(i * 93);
    }
    MerkleTree::from_data_with_config(&x, config).expect("failed to create tree from slice")
}

fn build_disk_tree_from_iter<U: Unsigned>(
    leafs: usize,
    len: usize,
    height: usize,
    config: &StoreConfig,
) {
    let branches = <U as Unsigned>::to_usize();
    assert_eq!(len, get_merkle_tree_len(leafs, branches));
    assert_eq!(height, get_merkle_tree_height(leafs, branches));

    let mut a = XOR128::new();

    // Construct and store an MT using a named DiskStore.
    let mt: MerkleTree<[u8; 16], XOR128, DiskStore<_>, U> = MerkleTree::try_from_iter_with_config(
        (0..leafs).map(|x| {
            a.reset();
            (x * 3).hash(&mut a);
            leafs.hash(&mut a);
            Ok(a.hash())
        }),
        config.clone(),
    )
    .expect("failed to create quad tree");

    assert_eq!(mt.len(), len);
    assert_eq!(mt.leafs(), leafs);
    assert_eq!(mt.height(), height);
}

fn test_disk_tree_from_iter<U: Unsigned>(
    leafs: usize,
    len: usize,
    height: usize,
    num_challenges: usize,
) {
    let branches = <U as Unsigned>::to_usize();

    let name = format!("test_disk_tree_from_iter-{}-{}-{}", leafs, len, height);
    let temp_dir = tempdir::TempDir::new(&name).unwrap();

    // Construct and store an MT using a named DiskStore.
    let config = StoreConfig::new(temp_dir.path(), String::from(name), 2);
    build_disk_tree_from_iter::<U>(leafs, len, height, &config);

    // Sanity check loading the store from disk and then re-creating
    // the MT from it.
    let store = DiskStore::new_from_disk(len, branches, &config).unwrap();
    let mt_cache: MerkleTree<[u8; 16], XOR128, DiskStore<_>, U> =
        MerkleTree::from_data_store(store, leafs).unwrap();

    assert_eq!(mt_cache.len(), len);
    assert_eq!(mt_cache.leafs(), leafs);
    assert_eq!(mt_cache.height(), height);

    for i in 0..num_challenges {
        let index = i * (leafs / num_challenges);
        let p = mt_cache.gen_proof(index).unwrap();
        assert!(p.validate::<XOR128>());
    }
}

fn test_levelcache_v1_tree_from_iter<U: Unsigned>(
    leafs: usize,
    len: usize,
    height: usize,
    num_challenges: usize,
    cached_above_base: usize,
) {
    let branches = <U as Unsigned>::to_usize();

    let name = format!(
        "test_levelcache_v1_tree_from_iter-{}-{}-{}",
        leafs, len, height
    );
    let temp_dir = tempdir::TempDir::new(&name).unwrap();

    // Construct and store an MT using a named DiskStore.
    let config = StoreConfig::new(temp_dir.path(), String::from(name), cached_above_base);
    build_disk_tree_from_iter::<U>(leafs, len, height, &config);

    // Sanity check loading the store from disk and then re-creating
    // the MT from it.
    let store = DiskStore::new_from_disk(len, branches, &config).unwrap();
    let mut mt_cache: MerkleTree<[u8; 16], XOR128, DiskStore<_>, U> =
        MerkleTree::from_data_store(store, leafs).unwrap();

    assert_eq!(mt_cache.len(), len);
    assert_eq!(mt_cache.leafs(), leafs);
    assert_eq!(mt_cache.height(), height);

    match mt_cache.compact(config.clone(), StoreConfigDataVersion::One as u32) {
        Ok(x) => assert_eq!(x, true),
        Err(_) => panic!("Compaction failed"),
    }

    // Then re-create an MT using LevelCacheStore and generate all proofs.
    let level_cache_store: LevelCacheStore<[u8; 16], std::fs::File> =
        LevelCacheStore::new_from_disk(len, branches, &config).unwrap();

    let mt_level_cache: MerkleTree<[u8; 16], XOR128, LevelCacheStore<_, _>, U> =
        MerkleTree::from_data_store(level_cache_store, leafs)
            .expect("Failed to create MT from data store");

    assert_eq!(mt_level_cache.len(), len);
    assert_eq!(mt_level_cache.leafs(), leafs);
    assert_eq!(mt_level_cache.height(), height);

    // Verify all proofs are still working.
    for i in 0..num_challenges {
        let index = i * (leafs / num_challenges);
        let (proof, _) = mt_level_cache
            .gen_proof_and_partial_tree(index, config.levels)
            .expect("Failed to generate proof and partial tree");
        assert!(proof.validate::<XOR128>());
    }
}

#[test]
fn test_hasher_light() {
    let mut h = XOR128::new();
    "1234567812345678".hash(&mut h);
    h.reset();
    String::from("1234567812345678").hash(&mut h);
    assert_eq!(format!("{:#X}", h), "0x31323334353637383132333435363738");
    String::from("1234567812345678").hash(&mut h);
    assert_eq!(format!("{:#X}", h), "0x00000000000000000000000000000000");
    String::from("1234567812345678").hash(&mut h);
    assert_eq!(format!("{:#X}", h), "0x31323334353637383132333435363738");
}

#[test]
fn test_vec_from_slice() {
    let x = [String::from("ars"), String::from("zxc")];
    let mt: MerkleTree<[u8; 16], XOR128, VecStore<_>> =
        MerkleTree::from_data(&x).expect("failed to create tree");
    assert_eq!(
        mt.read_range(0, 3).unwrap(),
        [
            [0, 97, 114, 115, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 122, 120, 99, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 27, 10, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    );
    assert_eq!(mt.len(), 3);
    assert_eq!(mt.leafs(), 2);
    assert_eq!(mt.height(), 2);
    assert_eq!(
        mt.root(),
        [1, 0, 27, 10, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    );

    for i in 0..mt.leafs() {
        let p = mt.gen_proof(i).unwrap();
        assert!(p.validate::<XOR128>());
    }
}

// B: Branching factor of sub-trees
// N: Branching factor of top-layer
fn test_compound_tree_from_slices<B: Unsigned, N: Unsigned>(sub_tree_leafs: usize) {
    let branches = <B as Unsigned>::to_usize();
    let sub_tree_count = <N as Unsigned>::to_usize();
    let mut sub_trees = Vec::with_capacity(sub_tree_count);
    for _ in 0..sub_tree_count {
        sub_trees.push(get_vec_tree_from_slice::<B>(sub_tree_leafs));
    }

    let tree: CompoundMerkleTree<[u8; 16], XOR128, VecStore<_>, B, N> =
        CompoundMerkleTree::new(sub_trees).expect("Failed to build compound tree");

    assert_eq!(
        tree.len(),
        (get_merkle_tree_len(sub_tree_leafs, branches) * sub_tree_count) + 1
    );
    assert_eq!(tree.leafs(), sub_tree_count * sub_tree_leafs);

    for i in 0..tree.leafs() {
        let p: CompoundProof<[u8; 16], B, N> = tree.gen_proof(i).unwrap();
        assert!(p.validate::<XOR128>());
    }
}

// B: Branching factor of sub-trees
// N: Branching factor of top-layer
fn test_compound_tree_from_store_configs<B: Unsigned, N: Unsigned>(sub_tree_leafs: usize) {
    let branches = <B as Unsigned>::to_usize();
    let sub_tree_count = <N as Unsigned>::to_usize();
    let mut sub_tree_configs = Vec::with_capacity(sub_tree_count);

    let temp_dir = tempdir::TempDir::new("test_read_into").unwrap();

    for i in 0..sub_tree_count {
        let config = StoreConfig::new(
            temp_dir.path(),
            format!("test-compound-tree-from-store-{}", i),
            StoreConfig::default_cached_above_base_layer(sub_tree_leafs, branches),
        );
        get_disk_tree_from_slice::<B>(sub_tree_leafs, config.clone());
        sub_tree_configs.push(config);
    }

    let tree: CompoundMerkleTree<[u8; 16], XOR128, DiskStore<_>, B, N> =
        CompoundMerkleTree::new_from_store_configs(sub_tree_leafs, sub_tree_configs)
            .expect("Failed to build compound tree");

    assert_eq!(
        tree.len(),
        (get_merkle_tree_len(sub_tree_leafs, branches) * sub_tree_count) + 1
    );
    assert_eq!(tree.leafs(), sub_tree_count * sub_tree_leafs);

    for i in 0..tree.leafs() {
        let p = tree.gen_proof(i).unwrap();
        assert!(p.validate::<XOR128>());
    }
}

#[test]
fn test_compound_quad_trees_from_slices() {
    // 3 quad trees each with 4 leafs joined by top layer
    test_compound_tree_from_slices::<U4, U3>(4);

    // 5 quad trees each with 16 leafs joined by top layer
    test_compound_tree_from_slices::<U4, U5>(16);

    // 7 quad trees each with 64 leafs joined by top layer
    test_compound_tree_from_slices::<U4, U7>(64);
}

#[test]
fn test_compound_quad_trees_from_store_configs() {
    // 3 quad trees each with 4 leafs joined by top layer
    test_compound_tree_from_store_configs::<U4, U3>(4);

    // 5 quad trees each with 16 leafs joined by top layer
    test_compound_tree_from_store_configs::<U4, U5>(16);

    // 7 quad trees each with 64 leafs joined by top layer
    test_compound_tree_from_store_configs::<U4, U7>(64);
}

#[test]
fn test_compound_octrees_from_slices() {
    // 3 octrees each with 8 leafs joined by top layer
    test_compound_tree_from_slices::<U8, U3>(8);

    // 5 octrees each with 64 leafs joined by top layer
    test_compound_tree_from_slices::<U8, U5>(64);

    // 7 octrees each with 320 leafs joined by top layer
    test_compound_tree_from_slices::<U8, U7>(512);
}

#[test]
fn test_compound_octrees_from_store_configs() {
    // 3 octrees each with 8 leafs joined by top layer
    test_compound_tree_from_store_configs::<U8, U3>(8);

    // 5 octrees each with 64 leafs joined by top layer
    test_compound_tree_from_store_configs::<U8, U5>(64);

    // 7 octrees each with 320 leafs joined by top layer
    test_compound_tree_from_store_configs::<U8, U7>(512);
}

#[test]
fn test_compound_quad_tree_from_slices() {
    // This tests a compound merkle tree that consists of 3 quad trees
    // with 4 leafs each.  The compound tree will have 12 leaves.
    let leafs = 4;
    let mt1 = get_vec_tree_from_slice::<U4>(leafs);
    let mt2 = get_vec_tree_from_slice::<U4>(leafs);
    let mt3 = get_vec_tree_from_slice::<U4>(leafs);

    let tree: CompoundMerkleTree<[u8; 16], XOR128, VecStore<_>, U4, U3> =
        CompoundMerkleTree::new(vec![mt1, mt2, mt3]).expect("Failed to build compound tree");
    assert_eq!(tree.len(), 16);
    assert_eq!(tree.leafs(), 12);
    assert_eq!(tree.height(), 3);

    for i in 0..tree.leafs() {
        let p = tree.gen_proof(i).unwrap();
        assert!(p.validate::<XOR128>());
    }
}

#[test]
fn test_compound_octree_from_slices() {
    // This tests a compound merkle tree that consists of 5 octrees
    // with 64 leafs each.  The compound tree will have 320 leaves.
    let leafs = 64;
    let mt1 = get_vec_tree_from_slice::<U8>(leafs);
    let mt2 = get_vec_tree_from_slice::<U8>(leafs);
    let mt3 = get_vec_tree_from_slice::<U8>(leafs);
    let mt4 = get_vec_tree_from_slice::<U8>(leafs);
    let mt5 = get_vec_tree_from_slice::<U8>(leafs);

    let tree: CompoundMerkleTree<[u8; 16], XOR128, VecStore<_>, U8, U5> =
        CompoundMerkleTree::new(vec![mt1, mt2, mt3, mt4, mt5])
            .expect("Failed to build compound tree");

    assert_eq!(tree.len(), 366);
    assert_eq!(tree.leafs(), 320);
    assert_eq!(tree.height(), 4);

    for i in 0..tree.leafs() {
        let p = tree.gen_proof(i).unwrap();
        assert!(p.validate::<XOR128>());
    }
}

#[test]
fn test_quad_from_slice() {
    let (leafs, len, height, num_challenges) = { (16, 21, 3, 16) };
    test_vec_tree_from_slice::<U4>(leafs, len, height, num_challenges);
}

#[test]
fn test_quad_from_iter() {
    let (leafs, len, height, num_challenges) = { (16384, 21845, 8, 16384) };
    test_vec_tree_from_iter::<U4>(leafs, len, height, num_challenges);
}

#[test]
#[ignore]
fn test_xlarge_quad_with_disk_store() {
    let (leafs, len, height, num_challenges) = { (1073741824, 1431655765, 16, 2048) };
    test_disk_tree_from_iter::<U4>(leafs, len, height, num_challenges);
}

#[test]
fn test_small_quad_with_partial_cache() {
    let (leafs, len, height, num_challenges) = { (256, 341, 5, 256) };
    for cached_above_base in 1..height - 1 {
        test_levelcache_v1_tree_from_iter::<U4>(
            leafs,
            len,
            height,
            num_challenges,
            cached_above_base,
        );
    }
}

#[test]
fn test_large_quad_with_partial_cache() {
    let (leafs, len, height, num_challenges) = { (1048576, 1398101, 11, 2048) };
    for cached_above_base in 5..7 {
        test_levelcache_v1_tree_from_iter::<U4>(
            leafs,
            len,
            height,
            num_challenges,
            cached_above_base,
        );
    }
}

#[test]
#[ignore]
fn test_large_quad_with_partial_cache_full() {
    let (leafs, len, height, num_challenges, cached_above_base) =
        { (1048576, 1398101, 11, 1048576, 5) };
    test_levelcache_v1_tree_from_iter::<U4>(leafs, len, height, num_challenges, cached_above_base);
}

#[test]
fn test_octo_from_iter() {
    let (leafs, len, height, num_challenges) = { (64, 73, 3, 64) };
    test_vec_tree_from_iter::<U8>(leafs, len, height, num_challenges);
}

#[test]
fn test_large_octo_with_disk_store() {
    let (leafs, len, height, num_challenges) = { (2097152, 2396745, 8, 2048) };
    test_disk_tree_from_iter::<U8>(leafs, len, height, num_challenges);
}

#[test]
fn test_large_octo_with_partial_cache() {
    let (leafs, len, height, num_challenges) = { (2097152, 2396745, 8, 2048) };
    for cached_above_base in 5..7 {
        test_levelcache_v1_tree_from_iter::<U8>(
            leafs,
            len,
            height,
            num_challenges,
            cached_above_base,
        );
    }
}

#[test]
#[ignore]
fn test_large_octo_with_partial_cache_full() {
    let (leafs, len, height, num_challenges, cached_above_base) =
        { (2097152, 2396745, 8, 2048, 3) };
    test_levelcache_v1_tree_from_iter::<U8>(leafs, len, height, num_challenges, cached_above_base);
}

#[test]
#[ignore]
fn test_xlarge_octo_with_disk_store() {
    let (leafs, len, height, num_challenges) = { (1073741824, 1227133513, 11, 2048) };
    test_disk_tree_from_iter::<U8>(leafs, len, height, num_challenges);
}

#[test]
#[ignore]
fn test_xlarge_octo_with_partial_cache() {
    let (leafs, len, height, num_challenges, cached_above_base) =
        { (1073741824, 1227133513, 11, 2048, 6) };
    test_levelcache_v1_tree_from_iter::<U8>(leafs, len, height, num_challenges, cached_above_base);
}

#[test]
fn test_read_into() {
    let x = [String::from("ars"), String::from("zxc")];
    let mt: MerkleTree<[u8; 16], XOR128, VecStore<_>> =
        MerkleTree::from_data(&x).expect("failed to create tree");
    let target_data = [
        [0, 97, 114, 115, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 122, 120, 99, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 27, 10, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ];

    let mut read_buffer: [u8; 16] = [0; 16];
    for (pos, &data) in target_data.iter().enumerate() {
        mt.read_into(pos, &mut read_buffer).unwrap();
        assert_eq!(read_buffer, data);
    }

    let temp_dir = tempdir::TempDir::new("test_read_into").unwrap();
    let config = StoreConfig::new(
        temp_dir.path(),
        "test-read-into",
        StoreConfig::default_cached_above_base_layer(x.len(), DEFAULT_NUM_BRANCHES),
    );

    let mt2: MerkleTree<[u8; 16], XOR128, DiskStore<_>> =
        MerkleTree::from_data_with_config(&x, config).expect("failed to create tree");
    for (pos, &data) in target_data.iter().enumerate() {
        mt2.read_into(pos, &mut read_buffer).unwrap();
        assert_eq!(read_buffer, data);
    }
}

#[test]
fn test_from_iter() {
    let mut a = XOR128::new();

    let mt: MerkleTree<[u8; 16], XOR128, VecStore<_>> =
        MerkleTree::try_from_iter(["a", "b", "c", "d"].iter().map(|x| {
            a.reset();
            x.hash(&mut a);
            Ok(a.hash())
        }))
        .unwrap();
    assert_eq!(mt.len(), 7);
    assert_eq!(mt.height(), 3);
}

#[test]
fn test_simple_tree() {
    let answer: Vec<Vec<[u8; 16]>> = vec![
        vec![
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        vec![
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        vec![
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        vec![
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        vec![
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        vec![
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
    ];

    // pow 2 only supported
    for items in [2, 4].iter() {
        let mut a = XOR128::new();
        let mt_base: MerkleTree<[u8; 16], XOR128, VecStore<_>> = MerkleTree::try_from_iter(
            [1, 2, 3, 4, 5, 6, 7, 8]
                .iter()
                .map(|x| {
                    a.reset();
                    x.hash(&mut a);
                    Ok(a.hash())
                })
                .take(*items),
        )
        .unwrap();

        assert_eq!(mt_base.leafs(), *items);
        assert_eq!(mt_base.height(), log2_pow2(next_pow2(mt_base.len())));
        assert_eq!(
            mt_base.read_range(0, mt_base.len()).unwrap(),
            answer[*items - 2].as_slice()
        );
        assert_eq!(mt_base.read_at(0).unwrap(), mt_base.read_at(0).unwrap());

        for i in 0..mt_base.leafs() {
            let p = mt_base.gen_proof(i).unwrap();
            assert!(p.validate::<XOR128>());
        }

        let mut a2 = XOR128::new();
        let leafs: Vec<u8> = [1, 2, 3, 4, 5, 6, 7, 8]
            .iter()
            .map(|x| {
                a.reset();
                x.hash(&mut a);
                a.hash()
            })
            .take(*items)
            .map(|item| {
                a2.reset();
                a2.leaf(item).as_ref().to_vec()
            })
            .flatten()
            .collect();
        {
            let mt1: MerkleTree<[u8; 16], XOR128, VecStore<_>> =
                MerkleTree::from_byte_slice(&leafs).unwrap();
            assert_eq!(mt1.leafs(), *items);
            assert_eq!(mt1.height(), log2_pow2(next_pow2(mt1.len())));
            assert_eq!(
                mt_base.read_range(0, mt_base.len()).unwrap(),
                answer[*items - 2].as_slice()
            );

            for i in 0..mt1.leafs() {
                let p = mt1.gen_proof(i).unwrap();
                assert!(p.validate::<XOR128>());
            }
        }

        {
            let mt2: MerkleTree<[u8; 16], XOR128, DiskStore<_>> =
                MerkleTree::from_byte_slice(&leafs).unwrap();
            assert_eq!(mt2.leafs(), *items);
            assert_eq!(mt2.height(), log2_pow2(next_pow2(mt2.len())));
            for i in 0..mt2.leafs() {
                let p = mt2.gen_proof(i).unwrap();
                assert!(p.validate::<XOR128>());
            }
        }
    }
}

#[test]
fn test_large_tree() {
    let count = SMALL_TREE_BUILD * 2;
    test_vec_tree_from_iter::<U2>(count, 2 * count - 1, log2_pow2(2 * count), 0);
    test_disk_tree_from_iter::<U2>(count, 2 * count - 1, log2_pow2(2 * count), 0);
}

#[test]
fn test_large_tree_disk() {
    let a = XOR128::new();
    let count = SMALL_TREE_BUILD * SMALL_TREE_BUILD * 8;

    let mt_disk: MerkleTree<[u8; 16], XOR128, DiskStore<_>> =
        MerkleTree::from_par_iter((0..count).into_par_iter().map(|x| {
            let mut xor_128 = a.clone();
            xor_128.reset();
            x.hash(&mut xor_128);
            93.hash(&mut xor_128);
            xor_128.hash()
        }))
        .unwrap();
    assert_eq!(mt_disk.len(), 2 * count - 1);
}

#[test]
fn test_mmap_tree() {
    use std::{thread, time};

    let mut a = XOR128::new();
    let count = SMALL_TREE_BUILD * SMALL_TREE_BUILD * 128;

    let mut mt_map: MerkleTree<[u8; 16], XOR128, MmapStore<_>> =
        MerkleTree::try_from_iter((0..count).map(|x| {
            a.reset();
            x.hash(&mut a);
            93.hash(&mut a);
            Ok(a.hash())
        }))
        .unwrap();
    assert_eq!(mt_map.len(), 2 * count - 1);

    let config = {
        let temp_dir = tempdir::TempDir::new("test_mmap_tree").unwrap();
        let temp_path = temp_dir.path();
        StoreConfig::new(
            &temp_path,
            String::from("test-mmap-tree"),
            StoreConfig::default_cached_above_base_layer(count, DEFAULT_NUM_BRANCHES),
        )
    };

    println!("Sleeping ... (high mem usage is visible)");
    thread::sleep(time::Duration::from_secs(5));

    println!("Compacting ...");
    let res = mt_map
        .compact(config.clone(), 1)
        .expect("Compaction failed");
    assert_eq!(res, true);

    println!("Sleeping ... (reduced mem usage is visible)");
    thread::sleep(time::Duration::from_secs(10));

    mt_map.reinit().expect("Failed to re-init the mmap");

    for i in 0..100 {
        let p = mt_map.gen_proof(i * (count / 100)).unwrap();
        assert!(p.validate::<XOR128>());
    }
}

#[test]
fn test_level_cache_tree_v1() {
    let cached_above_base = 4;
    let count = SMALL_TREE_BUILD * 2;
    test_levelcache_v1_tree_from_iter::<U2>(
        count,
        2 * count - 1,
        log2_pow2(2 * count),
        count,
        cached_above_base,
    );
}

#[test]
fn test_level_cache_tree_v2() {
    let a = XOR128::new();
    let count = SMALL_TREE_BUILD * 2;

    let temp_dir = tempdir::TempDir::new("test_level_cache_tree_v2").unwrap();
    let temp_path = temp_dir.path();

    // Construct and store an MT using a named DiskStore.
    let config = StoreConfig::new(
        &temp_path,
        String::from("test-cache-v2"),
        StoreConfig::default_cached_above_base_layer(count, DEFAULT_NUM_BRANCHES),
    );

    let mut mt_disk: MerkleTree<[u8; 16], XOR128, DiskStore<_>> =
        MerkleTree::from_par_iter_with_config(
            (0..count).into_par_iter().map(|x| {
                let mut xor_128 = a.clone();
                xor_128.reset();
                x.hash(&mut xor_128);
                99.hash(&mut xor_128);
                xor_128.hash()
            }),
            config.clone(),
        )
        .expect("Failed to create MT");
    assert_eq!(mt_disk.len(), 2 * count - 1);

    // Generate proofs on tree.
    for j in 0..mt_disk.leafs() {
        // First generate and validate the proof using the full
        // range of data we have stored on disk (no partial tree
        // is built or used in this case).
        let p = mt_disk.gen_proof(j).unwrap();
        assert!(p.validate::<XOR128>());
    }

    // Copy the base data from the store to a separate file that
    // is not managed by the store (for use later with an
    // ExternalReader).
    let reader = OpenOptions::new()
        .read(true)
        .open(StoreConfig::data_path(&config.path, &config.id))
        .expect("Failed to open base layer data");
    let mut base_layer = vec![0; count * 16];
    reader
        .read_exact_at(&mut base_layer, 0)
        .expect("Failed to read");

    let output_file = temp_path.join("base-data-only");
    std::fs::write(&output_file, &base_layer).expect("Failed to write output file");

    // Re-open the reader for the newly created output file.
    let reader = OpenOptions::new()
        .read(true)
        .open(&output_file)
        .expect("Failed to open base layer data");

    // Compact the disk store for use as a LevelCacheStore (v2
    // stores only the cached data and requires the ExternalReader
    // for base data retrieval).
    match mt_disk.compact(config.clone(), StoreConfigDataVersion::Two as u32) {
        Ok(x) => assert_eq!(x, true),
        Err(_) => panic!("Compaction failed"), // Could not do any compaction with this configuration.
    }

    // Then re-create an MT using LevelCacheStore and generate all proofs.
    let external_reader = ExternalReader {
        source: reader,
        read_fn: |start, end, buf: &mut [u8], reader: &std::fs::File| {
            reader
                .read_exact_at(&mut buf[0..end - start], start as u64)
                .expect("Failed to read");

            Ok(end - start)
        },
    };

    let level_cache_store: LevelCacheStore<[u8; 16], _> =
        LevelCacheStore::new_from_disk_with_reader(
            2 * count - 1,
            DEFAULT_NUM_BRANCHES,
            &config,
            external_reader,
        )
        .unwrap();

    let mt_level_cache: MerkleTree<[u8; 16], XOR128, LevelCacheStore<_, _>> =
        MerkleTree::from_data_store(level_cache_store, count)
            .expect("Failed to create MT from data store");
    assert_eq!(mt_level_cache.len(), 2 * count - 1);

    // Generate proofs on tree.
    for j in 0..mt_level_cache.leafs() {
        let (proof, _) = mt_level_cache
            .gen_proof_and_partial_tree(j, config.levels)
            .expect("Failed to generate proof and partial tree");
        assert!(proof.validate::<XOR128>());
    }
}

#[test]
fn test_various_trees_with_partial_cache_v2_only() {
    env_logger::init();
    let mut a = XOR128::new();

    // Attempt to allow this test to move along relatively quickly.
    let min_count = SMALL_TREE_BUILD / 4;
    let max_count = SMALL_TREE_BUILD * 4;
    let mut count = min_count;

    // Test a range of tree sizes, given a range of leaf elements.
    while count <= max_count {
        let pow = next_pow2(count);
        let height = log2_pow2(2 * pow);

        // Test a range of heights to cache above the base (for
        // different partial tree sizes).
        //
        // compaction correctly fails at 0 and height
        for i in 1..height - 1 {
            let temp_dir = tempdir::TempDir::new("test_various_trees_with_partial_cache").unwrap();
            let temp_path = temp_dir.path();

            // Construct and store an MT using a named DiskStore.
            let config = StoreConfig::new(
                &temp_path,
                String::from(format!("test-partial-cache-{}", i)),
                i,
            );

            let mut mt_cache: MerkleTree<[u8; 16], XOR128, DiskStore<_>> =
                MerkleTree::try_from_iter_with_config(
                    (0..count).map(|x| {
                        a.reset();
                        x.hash(&mut a);
                        count.hash(&mut a);
                        Ok(a.hash())
                    }),
                    config.clone(),
                )
                .expect("failed to create merkle tree from iter with config");

            // Sanity check loading the store from disk and then
            // re-creating the MT from it.
            let store =
                DiskStore::new_from_disk(2 * count - 1, DEFAULT_NUM_BRANCHES, &config).unwrap();
            let mt_cache2: MerkleTree<[u8; 16], XOR128, DiskStore<_>> =
                MerkleTree::from_data_store(store, count).unwrap();

            assert_eq!(mt_cache.len(), mt_cache2.len());
            assert_eq!(mt_cache.leafs(), mt_cache2.leafs());

            assert_eq!(mt_cache.len(), 2 * count - 1);
            assert_eq!(mt_cache.leafs(), count);

            // Generate and validate proof on the first element.
            //let p = mt_cache.gen_proof(0).unwrap();
            //assert!(p.validate::<XOR128>());

            /*
            // This is commented out because it's no longer necessary.
            // The idea below is that we generate 2 partial merkle
            // trees and then all of the proofs re-using those trees.
            // With the optimal partial tree generation imlemented
            // now, this use case is not as appealing as it once was
            // envisioned to be.

            // Generate and validate proof on the first element and also
            // retrieve the partial tree needed for future proof
            // generation.  This is an optimization that lets us re-use
            // the partially generated tree, given the known access
            // pattern.
            //
            // NOTE: Using partial tree proof generation with a DiskStore
            // does not generally make sense (just use gen_proof), but it
            // does provide a proof of concept implementation to show that
            // we can generate proofs only using certain segments of the
            // on-disk data.
            let pat1 = mt_cache.gen_proof_and_partial_tree(0, i).unwrap();
            assert!(pat1.proof.validate::<XOR128>());

            // Same as above, but generate and validate the proof on the
            // first element of the second data half and retrieve the
            // partial tree needed for future proofs in that range.
            let pat2 = mt_cache
                .gen_proof_and_partial_tree(mt_cache.leafs() / 2, i)
                .unwrap();
            assert!(pat2.proof.validate::<XOR128>());
            */

            for j in 0..mt_cache.leafs() {
                // First generate and validate the proof using the full
                // range of data we have stored on disk (no partial tree
                // is built or used in this case).
                let p = mt_cache.gen_proof(j).unwrap();
                assert!(p.validate::<XOR128>());

                /*
                // See comment above on why this is no longer necessary.

                // Then generate proofs using a combination of data in the
                // partial tree generated outside of this loop, and data
                // on disk (simulating a partial cache since we do not use
                // the full range of data stored on disk in these cases).
                if j < mt_cache.leafs() / 2 {
                    let p1 = mt_cache.gen_proof_with_partial_tree(j, i, &pat1.merkle_tree);
                    assert!(p1.validate::<XOR128>());
                } else {
                    let p2 = mt_cache.gen_proof_with_partial_tree(j, i, &pat2.merkle_tree);
                    assert!(p2.validate::<XOR128>());
                }
                */
            }

            // Once we have the full on-disk MT data, we can optimize
            // space for future access by compacting it into the partially
            // cached data format.
            //
            // Before store compaction, save the mt_cache.len() so that we
            // can assert after rebuilding the MT from the compacted data
            // that it matches.
            let mt_cache_len = mt_cache.len();

            // Copy the base data from the store to a separate file that
            // is not managed by the store (for use later with an
            // ExternalReader).
            let reader = OpenOptions::new()
                .read(true)
                .open(StoreConfig::data_path(&config.path, &config.id))
                .expect("Failed to open base layer data");
            let mut base_layer = vec![0; count * 16];
            reader
                .read_exact_at(&mut base_layer, 0)
                .expect("Failed to read");

            let output_file = temp_path.join("base-data-only");
            std::fs::write(&output_file, &base_layer).expect("Failed to write output file");

            // Re-open the reader for the newly created output file.
            let reader = OpenOptions::new()
                .read(true)
                .open(&output_file)
                .expect("Failed to open base layer data");

            // Compact the newly created DiskStore into the
            // LevelCacheStore format.  This uses information from the
            // Config to properly shape the compacted data for later
            // access using the LevelCacheStore interface.
            //
            // NOTE: If we were v1 compacting here instead of v2, it's
            // possible that the cache would result in a larger data
            // file than the original tree data, in which case
            // compaction could fail.  It does NOT panic here because
            // for v2 compaction, we only store the cached data.
            match mt_cache.compact(config.clone(), StoreConfigDataVersion::Two as u32) {
                Ok(x) => assert_eq!(x, true),
                Err(_) => panic!("Compaction failed"), // Could not do any compaction with this configuration.
            }

            // Then re-create an MT using LevelCacheStore and generate all proofs.
            let external_reader = ExternalReader {
                source: reader,
                read_fn: |start, end, buf: &mut [u8], reader: &std::fs::File| {
                    reader
                        .read_exact_at(&mut buf[0..end - start], start as u64)
                        .expect("Failed to read");

                    Ok(end - start)
                },
            };

            let level_cache_store: LevelCacheStore<[u8; 16], _> =
                LevelCacheStore::new_from_disk_with_reader(
                    2 * count - 1,
                    DEFAULT_NUM_BRANCHES,
                    &config,
                    external_reader,
                )
                .unwrap();

            let mt_level_cache: MerkleTree<[u8; 16], XOR128, LevelCacheStore<_, _>> =
                MerkleTree::from_data_store(level_cache_store, count)
                    .expect("Failed to revive LevelCacheStore after compaction");

            // Sanity check that after rebuild, the new MT properties match the original.
            assert_eq!(mt_level_cache.len(), mt_cache_len);
            assert_eq!(mt_level_cache.leafs(), mt_cache.leafs());

            // This is the proper way to generate a single proof using
            // the LevelCacheStore.  The optimal partial tree is
            // built, given the cached parameters and the properly
            // recorded LevelCacheStore.
            for j in 0..mt_level_cache.leafs() {
                let (proof, _) = mt_level_cache
                    .gen_proof_and_partial_tree(j, i)
                    .expect("Failed to generate proof and partial tree");
                assert!(proof.validate::<XOR128>());
            }

            /*
            // This is commented out because it's no longer necessary.
            // The idea below is that we generate 2 partial merkle
            // trees and then all of the proofs re-using those trees.
            // With the optimal partial tree generation imlemented
            // now, this use case is not as appealing as it once was
            // envisioned to be.

            // Optimized proof generation based on simple generation pattern:
            let pat1 = mt_level_cache.gen_proof_and_partial_tree(0, i).unwrap();
            assert!(pat1.proof.validate::<XOR128>());

            // Same as above, but generate and validate the proof on the
            // first element of the second data half and retrieve the
            // partial tree needed for future proofs in that range.
            let pat2 = mt_level_cache
                .gen_proof_and_partial_tree(mt_level_cache.leafs() / 2, i)
                .unwrap();
            assert!(pat2.proof.validate::<XOR128>());

            for j in 1..mt_level_cache.leafs() {
                // Generate proofs using a combination of data in the
                // partial tree generated outside of this loop, and data
                // on disk (which now only contains the base layer and
                // cached range).
                if j < mt_level_cache.leafs() / 2 {
                    let p1 = mt_level_cache.gen_proof_with_partial_tree(j, i, &pat1.merkle_tree);
                    assert!(p1.validate::<XOR128>());
                } else {
                    let p2 = mt_level_cache.gen_proof_with_partial_tree(j, i, &pat2.merkle_tree);
                    assert!(p2.validate::<XOR128>());
                }
            }
            */

            // Delete the single store backing this MT (for this test,
            // the DiskStore is compacted and then shared with the
            // LevelCacheStore, so it's still a single store on disk).
            mt_level_cache
                .delete(config.clone())
                .expect("Failed to delete test store");

            // This also works (delete the store directly)
            //LevelCacheStore::<[u8; 16]>::delete(config.clone())
            //    .expect("Failed to delete test store");
        }

        count <<= 1;
    }
}

#[test]
fn test_parallel_iter_disk_1() {
    let data = vec![1u8; 16 * 128];
    let store: DiskStore<[u8; 16]> = DiskStore::new_from_slice(128, &data).unwrap();

    let p = DiskStoreProducer {
        current: 0,
        end: 128,
        store: &store,
    };

    assert_eq!(p.len(), 128);

    let collected: Vec<[u8; 16]> = p.clone().into_iter().collect();
    for (a, b) in collected.iter().zip(data.chunks_exact(16)) {
        assert_eq!(a, b);
    }

    let (a1, b1) = p.clone().split_at(64);
    assert_eq!(a1.len(), 64);
    assert_eq!(b1.len(), 64);

    let (a2, b2) = a1.split_at(32);
    assert_eq!(a2.len(), 32);
    assert_eq!(b2.len(), 32);

    let (a3, b3) = a2.split_at(16);
    assert_eq!(a3.len(), 16);
    assert_eq!(b3.len(), 16);

    let (a4, b4) = a3.split_at(8);
    assert_eq!(a4.len(), 8);
    assert_eq!(b4.len(), 8);

    let (a5, b5) = a4.split_at(4);
    assert_eq!(a5.len(), 4);
    assert_eq!(b5.len(), 4);

    let (a6, b6) = a5.split_at(2);
    assert_eq!(a6.len(), 2);
    assert_eq!(b6.len(), 2);

    let (a7, b7) = a6.split_at(1);
    assert_eq!(a7.len(), 1);
    assert_eq!(b7.len(), 1);

    // nothing happens
    let (a8, b8) = a7.clone().split_at(1);
    assert_eq!(a8.len(), 1);
    assert_eq!(b8.len(), 0);

    let (a8, b8) = a7.split_at(10);
    assert_eq!(a8.len(), 1);
    assert_eq!(b8.len(), 0);

    let (a, b) = p.clone().split_at(10);

    for (a, b) in a.into_iter().zip(data.chunks_exact(16).take(10)) {
        assert_eq!(a, b);
    }

    for (a, b) in b.into_iter().zip(data.chunks_exact(16).skip(10)) {
        assert_eq!(a, b);
    }

    let mut disk_iter = p.into_iter();
    let mut i = 128;
    while let Some(_el) = disk_iter.next_back() {
        i -= 1;
    }

    assert_eq!(i, 0);
}

#[test]
fn test_parallel_iter_disk_2() {
    for size in &[2, 4, 5, 99, 128] {
        let size = *size;
        println!(" --- {}", size);

        let data = vec![1u8; 16 * size];
        let store: DiskStore<[u8; 16]> = DiskStore::new_from_slice(size, &data).unwrap();

        let p = DiskStoreProducer {
            current: 0,
            end: size,
            store: &store,
        };

        assert_eq!(p.len(), size);

        let par_iter = store.into_par_iter();
        assert_eq!(Store::len(&par_iter), size);

        let collected: Vec<[u8; 16]> = par_iter.collect();
        for (a, b) in collected.iter().zip(data.chunks_exact(16)) {
            assert_eq!(a, b);
        }
    }
}
