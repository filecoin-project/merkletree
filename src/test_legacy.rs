#![cfg(test)]
#![cfg(not(tarpaulin_include))]

use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::fmt::{Debug, Formatter};
use std::fs::OpenOptions;
use std::hash::Hasher;
use std::io::Write as ioWrite;
use std::mem;
use std::num::ParseIntError;
use std::os::unix::prelude::FileExt;
use std::path::Path;
use std::slice;

use byteorder::{ByteOrder, NativeEndian};
use rayon::iter::{plumbing::Producer, IntoParallelIterator, ParallelIterator};
use sha2::{Digest, Sha256};
use typenum::marker_traits::Unsigned;
use typenum::{U2, U3, U4, U5, U7, U8};

use crate::hash::{Algorithm, Hashable};
use crate::merkle::{
    get_merkle_tree_len, get_merkle_tree_row_count, is_merkle_tree_size_valid, log2_pow2,
    next_pow2, Element, FromIndexedParallelIterator, MerkleTree,
};
use crate::store::{
    DiskStore, DiskStoreProducer, ExternalReader, LevelCacheStore, ReplicaConfig, Store,
    StoreConfig, StoreConfigDataVersion, VecStore, SMALL_TREE_BUILD,
};

fn build_disk_tree_from_iter<U: Unsigned>(
    leafs: usize,
    len: usize,
    row_count: usize,
    config: &StoreConfig,
) {
    let branches = U::to_usize();
    assert_eq!(
        len,
        get_merkle_tree_len(leafs, branches)
            .expect("[build_disk_tree_from_iter] failed to get merkle len")
    );
    assert_eq!(row_count, get_merkle_tree_row_count(leafs, branches));

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
    .expect("[build_disk_tree_from_iter] failed to create tree");

    assert_eq!(mt.len(), len);
    assert_eq!(mt.leafs(), leafs);
    assert_eq!(mt.row_count(), row_count);
}

pub fn get_levelcache_tree_from_slice<U: Unsigned>(
    leafs: usize,
    len: usize,
    row_count: usize,
    config: &StoreConfig,
    replica_path: &Path,
) -> MerkleTree<[u8; 16], XOR128, LevelCacheStore<[u8; 16], std::fs::File>, U> {
    let branches = U::to_usize();
    assert_eq!(
        len,
        get_merkle_tree_len(leafs, branches)
            .expect("[get_levelcache_tree_from_slice] failed to get merkle len")
    );
    assert_eq!(row_count, get_merkle_tree_row_count(leafs, branches));

    let mut x = Vec::with_capacity(leafs);
    for i in 0..leafs {
        x.push(i * 3);
    }

    let mut mt = MerkleTree::from_data_with_config(&x, config.clone())
        .expect("[get_levelcache_tree_from_slice] failed to create tree from slice");

    assert_eq!(mt.len(), len);
    assert_eq!(mt.leafs(), leafs);
    assert_eq!(mt.row_count(), row_count);

    mt.set_external_reader_path(replica_path)
        .expect("[get_levelcache_tree_from_slice] Failed to set external reader");

    mt
}

fn get_levelcache_tree_from_iter<U: Unsigned>(
    leafs: usize,
    len: usize,
    row_count: usize,
    config: &StoreConfig,
    replica_path: &Path,
) -> MerkleTree<[u8; 16], XOR128, LevelCacheStore<[u8; 16], std::fs::File>, U> {
    let branches = U::to_usize();
    assert_eq!(
        len,
        get_merkle_tree_len(leafs, branches)
            .expect("[get_levelcache_tree_from_iter] failed to get merkle len")
    );
    assert_eq!(row_count, get_merkle_tree_row_count(leafs, branches));

    let mut a = XOR128::new();

    // Construct and store an MT using a named LevelCacheStore.
    let mut mt: MerkleTree<[u8; 16], XOR128, LevelCacheStore<_, std::fs::File>, U> =
        MerkleTree::try_from_iter_with_config(
            (0..leafs).map(|x| {
                a.reset();
                (x * 3).hash(&mut a);
                leafs.hash(&mut a);
                Ok(a.hash())
            }),
            config.clone(),
        )
        .expect("[get_levelcache_tree_from_iter] failed to create tree");

    assert_eq!(mt.len(), len);
    assert_eq!(mt.leafs(), leafs);
    assert_eq!(mt.row_count(), row_count);

    mt.set_external_reader_path(replica_path)
        .expect("[get_levelcache_tree_from_iter] Failed to set external reader");

    mt
}

fn test_levelcache_v1_tree_from_iter<U: Unsigned>(
    leafs: usize,
    len: usize,
    row_count: usize,
    num_challenges: usize,
    rows_to_discard: usize,
) {
    let branches = U::to_usize();

    let name = format!(
        "test_levelcache_v1_tree_from_iter-{}-{}-{}",
        leafs, len, row_count
    );
    let temp_dir = tempfile::Builder::new()
        .prefix(&name)
        .tempdir()
        .expect("[test_levelcache_v1_tree_from_iter] couldn't create temp_dir");

    // Construct and store an MT using a named DiskStore.
    let config = StoreConfig::new(temp_dir.path(), name, rows_to_discard);
    build_disk_tree_from_iter::<U>(leafs, len, row_count, &config);

    // Sanity check loading the store from disk and then re-creating
    // the MT from it.
    assert!(DiskStore::<[u8; 16]>::is_consistent(len, branches, &config)
        .expect("[test_levelcache_v1_tree_from_iter] can't check if store is consistent"));
    let store = DiskStore::new_from_disk(len, branches, &config)
        .expect("[test_levelcache_v1_tree_from_iter] couldn't instantiate DiskStore");
    let mut mt_cache: MerkleTree<[u8; 16], XOR128, DiskStore<_>, U> =
        MerkleTree::from_data_store(store, leafs)
            .expect("[test_levelcache_v1_tree_from_iter] couldn't create Merkle Tree");

    assert_eq!(mt_cache.len(), len);
    assert_eq!(mt_cache.leafs(), leafs);
    assert_eq!(mt_cache.row_count(), row_count);

    match mt_cache.compact(config.clone(), StoreConfigDataVersion::One as u32) {
        Ok(x) => assert!(x),
        Err(_) => panic!("Compaction failed"),
    }

    // Then re-create an MT using LevelCacheStore and generate all proofs.
    assert!(
        LevelCacheStore::<[u8; 16], std::fs::File>::is_consistent_v1(len, branches, &config)
            .expect(
            "[test_levelcache_v1_tree_from_iter] couldn't check if LevelCacheStore is consistent"
        )
    );
    let level_cache_store: LevelCacheStore<[u8; 16], std::fs::File> =
        LevelCacheStore::new_from_disk(len, branches, &config)
            .expect("[test_levelcache_v1_tree_from_iter] couldn't instantiate LevelCacheStore");

    let mt_level_cache: MerkleTree<[u8; 16], XOR128, LevelCacheStore<_, _>, U> =
        MerkleTree::from_data_store(level_cache_store, leafs)
            .expect("[test_levelcache_v1_tree_from_iter] Failed to create MT from data store");

    assert_eq!(mt_level_cache.len(), len);
    assert_eq!(mt_level_cache.leafs(), leafs);
    assert_eq!(mt_level_cache.row_count(), row_count);

    // Verify all proofs are still working.
    for i in 0..num_challenges {
        let index = i * (leafs / num_challenges);
        let proof = mt_level_cache
            .gen_cached_proof(index, Some(config.rows_to_discard))
            .expect(
                "[test_levelcache_v1_tree_from_iter] Failed to generate proof and partial tree",
            );
        assert!(proof
            .validate::<XOR128>()
            .expect("[test_levelcache_v1_tree_from_iter] failed to validate"));
    }
}

fn test_levelcache_direct_build_from_slice<U: Unsigned>(
    leafs: usize,
    len: usize,
    row_count: usize,
    num_challenges: usize,
    rows_to_discard: Option<usize>,
) {
    assert!(is_merkle_tree_size_valid(leafs, U::to_usize()));

    let test_name = "test_levelcache_direct_build_from_slice";
    let replica = format!("{}-{}-{}-{}-replica", test_name, leafs, len, row_count);
    let lc_name = format!("{}-{}-{}-{}", test_name, leafs, len, row_count);
    let temp_dir = tempfile::Builder::new()
        .prefix(&test_name)
        .tempdir()
        .expect("[test_levelcache_direct_build_from_slice] couldn't create temp_dir");

    let rows_to_discard = match rows_to_discard {
        Some(x) => x,
        None => StoreConfig::default_rows_to_discard(leafs, U::to_usize()),
    };
    // Construct and store an MT using a named DiskStore.
    let config = StoreConfig::new(temp_dir.path(), String::from(&replica), rows_to_discard);
    build_disk_tree_from_iter::<U>(leafs, len, row_count, &config);

    // Use that data store as the replica.
    let replica_path = StoreConfig::data_path(&config.path, &config.id);

    // Construct level cache tree/store directly, using the above replica.
    let lc_config = StoreConfig::from_config(&config, lc_name, Some(len));
    let lc_tree =
        get_levelcache_tree_from_slice::<U>(leafs, len, row_count, &lc_config, &replica_path);

    // Verify all proofs are working.
    for i in 0..num_challenges {
        let index = i * (leafs / num_challenges);
        let proof = lc_tree
            .gen_cached_proof(index, Some(rows_to_discard))
            .expect(
            "[test_levelcache_direct_build_from_slice] Failed to generate proof and partial tree",
        );
        assert!(proof
            .validate::<XOR128>()
            .expect("[test_levelcache_direct_build_from_slice] failed to validate"));
    }
}

fn test_levelcache_direct_build_from_iter<U: Unsigned>(
    leafs: usize,
    len: usize,
    row_count: usize,
    num_challenges: usize,
    rows_to_discard: Option<usize>,
) {
    assert!(is_merkle_tree_size_valid(leafs, U::to_usize()));

    let test_name = "test_levelcache_direct_build_from_iter";
    let replica = format!("{}-{}-{}-{}-replica", test_name, leafs, len, row_count);
    let lc_name = format!("{}-{}-{}-{}", test_name, leafs, len, row_count);
    let temp_dir = tempfile::Builder::new()
        .prefix(&test_name)
        .tempdir()
        .expect("[test_levelcache_direct_build_from_iter] couldn't create temp_dir");

    let rows_to_discard = match rows_to_discard {
        Some(x) => x,
        None => StoreConfig::default_rows_to_discard(leafs, U::to_usize()),
    };
    // Construct and store an MT using a named DiskStore.
    let config = StoreConfig::new(temp_dir.path(), String::from(&replica), rows_to_discard);
    build_disk_tree_from_iter::<U>(leafs, len, row_count, &config);

    // Use that data store as the replica.
    let replica_path = StoreConfig::data_path(&config.path, &config.id);

    // Construct level cache tree/store directly, using the above replica.
    let lc_config = StoreConfig::from_config(&config, lc_name, Some(len));
    let lc_tree =
        get_levelcache_tree_from_iter::<U>(leafs, len, row_count, &lc_config, &replica_path);

    // Verify all proofs are working.
    for i in 0..num_challenges {
        let index = i * (leafs / num_challenges);
        let proof = lc_tree.gen_cached_proof(index, None).expect(
            "[test_levelcache_direct_build_from_iter] Failed to generate proof and partial tree",
        );
        assert!(proof
            .validate::<XOR128>()
            .expect("[test_levelcache_direct_build_from_iter] failed to validate"));
    }
}

#[test]
#[ignore]
fn test_levelcache_direct_build_quad() {
    let (leafs, len, row_count, num_challenges) = { (1048576, 1398101, 11, 2048) };

    test_levelcache_direct_build_from_iter::<U4>(leafs, len, row_count, num_challenges, None);

    test_levelcache_direct_build_from_slice::<U4>(leafs, len, row_count, num_challenges, None);
}

#[test]
#[ignore]
fn test_levelcache_direct_build_octo() {
    let (leafs, len, row_count, num_challenges, rows_to_discard) =
        { (262144, 299593, 7, 262144, 2) };

    test_levelcache_direct_build_from_iter::<U8>(
        leafs,
        len,
        row_count,
        num_challenges,
        Some(rows_to_discard),
    );

    test_levelcache_direct_build_from_slice::<U8>(
        leafs,
        len,
        row_count,
        num_challenges,
        Some(rows_to_discard),
    );
}

#[test]
fn test_hasher_light() {
    fn decode_hex(s: &str) -> Result<Vec<u8>, ParseIntError> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16))
            .collect()
    }
    fn run_test<E: Element, A: Algorithm<E>>(
        number_of_hashing: usize,
        input: Vec<u8>,
        expected_output: Vec<u8>,
    ) {
        let mut h = A::default();
        let mut result = vec![0u8; E::byte_len()];

        for _ in 0..number_of_hashing {
            input.hash(&mut h);
        }

        let element = h.hash();
        element.copy_to_slice(&mut result);
        assert_eq!(result, expected_output);
    }

    /*
       For XOR128 hasher: if we perform odd number of hashing operations,
       we always obtain output which is equal to input. Otherwise - output
       is equal to array of all zeroes
    */
    run_test::<Item, XOR128>(
        2,
        decode_hex("000102030405060708090a0b0c0d0e0f").unwrap(), // safe
        decode_hex("00000000000000000000000000000000").unwrap(), // safe
    );
    run_test::<Item, XOR128>(
        5,
        decode_hex("000102030405060708090a0b0c0d0e0f").unwrap(), // safe
        decode_hex("000102030405060708090a0b0c0d0e0f").unwrap(), // safe
    );
    run_test::<Item, XOR128>(
        10,
        decode_hex("000102030405060708090a0b0c0d0e0f").unwrap(), // safe
        decode_hex("00000000000000000000000000000000").unwrap(), // safe
    );
    run_test::<Item, XOR128>(
        371,
        decode_hex("000102030405060708090a0b0c0d0e0f").unwrap(), // safe
        decode_hex("000102030405060708090a0b0c0d0e0f").unwrap(), // safe
    );

    let mut sha256 = Sha256::new();
    let number_of_hashing = 951;
    let input = decode_hex("000102030405060708090a0b0c0d0e0f").unwrap(); // safe
    for _ in 0..number_of_hashing {
        sha256.update(input.as_slice());
    }
    let mut expected_output = sha256.finalize().to_vec();
    expected_output.truncate(Item::byte_len());

    run_test::<Item, Sha256Hasher>(number_of_hashing, input, expected_output);
}

// B: Branching factor of sub-trees
// N: Branching factor of top-layer
fn test_compound_levelcache_tree_from_store_configs<B: Unsigned, N: Unsigned>(
    sub_tree_leafs: usize,
) {
    let branches = B::to_usize();
    assert!(is_merkle_tree_size_valid(sub_tree_leafs, branches));

    let sub_tree_count = N::to_usize();
    let mut replica_offsets = Vec::with_capacity(sub_tree_count);
    let mut sub_tree_configs = Vec::with_capacity(sub_tree_count);

    let test_name = "test_compound_levelcache_tree_from_store_configs";
    let temp_dir = tempfile::Builder::new()
        .prefix("test_compound_levelcache_tree")
        .tempdir()
        .expect("[test_compound_levelcache_tree_from_store_configs] couldn't create temp_dir");
    let len = get_merkle_tree_len(sub_tree_leafs, branches)
        .expect("[test_compound_levelcache_tree_from_store_configs] failed to get merkle len");
    let row_count = get_merkle_tree_row_count(sub_tree_leafs, branches);

    let replica_path = StoreConfig::data_path(
        temp_dir.path(),
        &format!(
            "{}-{}-{}-{}-replica",
            test_name, sub_tree_leafs, len, row_count
        ),
    );
    let mut f_replica = std::fs::File::create(&replica_path)
        .expect("[test_compound_levelcache_tree_from_store_configs] failed to create replica file");

    for i in 0..sub_tree_count {
        let lc_name = format!(
            "{}-{}-{}-{}-lc-{}",
            test_name, sub_tree_leafs, len, row_count, i
        );
        let replica = format!(
            "{}-{}-{}-{}-replica-{}",
            test_name, sub_tree_leafs, len, row_count, i
        );
        let config = StoreConfig::new(
            temp_dir.path(),
            String::from(&replica),
            StoreConfig::default_rows_to_discard(sub_tree_leafs, branches),
        );
        build_disk_tree_from_iter::<B>(sub_tree_leafs, len, row_count, &config);
        let store = DiskStore::new_with_config(len, branches, config.clone())
            .expect("[test_compound_levelcache_tree_from_store_configs] failed to open store");

        // Use that data store as the replica (concat the data to the replica_path)
        let data: Vec<[u8; 16]> = store
            .read_range(std::ops::Range {
                start: 0,
                end: sub_tree_leafs,
            })
            .expect("[test_compound_levelcache_tree_from_store_configs] failed to read store");
        for element in data {
            f_replica.write_all(&element).expect(
                "[test_compound_levelcache_tree_from_store_configs] failed to write replica data",
            );
        }
        replica_offsets.push(i * (16 * sub_tree_leafs));

        let lc_config = StoreConfig::from_config(&config, lc_name, Some(len));
        get_levelcache_tree_from_slice::<B>(
            sub_tree_leafs,
            len,
            row_count,
            &lc_config,
            &replica_path,
        );

        sub_tree_configs.push(lc_config);
    }

    let replica_config = ReplicaConfig::new(replica_path, replica_offsets);
    let tree =
        MerkleTree::<[u8; 16], XOR128, LevelCacheStore<[u8; 16], std::fs::File>, B, N>::from_store_configs_and_replica(sub_tree_leafs, &sub_tree_configs, &replica_config)
            .expect("[test_compound_levelcache_tree_from_store_configs] Failed to build compound levelcache tree");

    assert_eq!(
        tree.len(),
        (get_merkle_tree_len(sub_tree_leafs, branches)
            .expect("[test_compound_levelcache_tree_from_store_configs] failed to get merkle len")
            * sub_tree_count)
            + 1
    );
    assert_eq!(tree.leafs(), sub_tree_count * sub_tree_leafs);

    for i in 0..tree.leafs() {
        // Make sure all elements are accessible.
        let _ = tree.read_at(i).expect(
            "[test_compound_levelcache_tree_from_store_configs] Failed to read tree element",
        );

        // Make sure all proofs validate.
        let p = tree.gen_cached_proof(i, None).expect("[test_compound_levelcache_tree_from_store_configs] couldn't generate cached Merkle Proof");
        assert!(p
            .validate::<XOR128>()
            .expect("[test_compound_levelcache_tree_from_store_configs] failed to validate"));
    }
}

#[test]
fn test_compound_levelcache_quad_trees_from_store_configs() {
    // 3 quad trees each with 16 leafs joined by top layer
    test_compound_levelcache_tree_from_store_configs::<U4, U3>(16);

    // 5 quad trees each with 64 leafs joined by top layer
    test_compound_levelcache_tree_from_store_configs::<U4, U5>(64);

    // 7 quad trees each with 256 leafs joined by top layer
    test_compound_levelcache_tree_from_store_configs::<U4, U7>(256);
}

#[test]
fn test_compound_levelcache_octrees_trees_from_store_configs() {
    // 3 octrees trees each with 64 leafs joined by top layer
    test_compound_levelcache_tree_from_store_configs::<U8, U3>(64);

    // 5 octrees trees each with 256 leafs joined by top layer
    test_compound_levelcache_tree_from_store_configs::<U8, U5>(512);

    // 7 octrees trees each with 2048 leafs joined by top layer
    test_compound_levelcache_tree_from_store_configs::<U8, U7>(4096);
}

#[test]
fn test_small_quad_with_partial_cache() {
    let (leafs, len, row_count, num_challenges) = { (256, 341, 5, 256) };
    for rows_to_discard in 1..row_count - 1 {
        test_levelcache_v1_tree_from_iter::<U4>(
            leafs,
            len,
            row_count,
            num_challenges,
            rows_to_discard,
        );
    }
}

#[test]
#[ignore]
fn test_large_quad_with_partial_cache() {
    let (leafs, len, row_count, num_challenges) = { (1048576, 1398101, 11, 2048) };
    for rows_to_discard in 5..7 {
        test_levelcache_v1_tree_from_iter::<U4>(
            leafs,
            len,
            row_count,
            num_challenges,
            std::cmp::min(
                rows_to_discard,
                StoreConfig::default_rows_to_discard(leafs, OCT_ARITY),
            ),
        );
    }
}

#[test]
#[ignore]
fn test_large_quad_with_partial_cache_full() {
    let (leafs, len, row_count, num_challenges, rows_to_discard) =
        { (1048576, 1398101, 11, 1048576, 5) };
    test_levelcache_v1_tree_from_iter::<U4>(leafs, len, row_count, num_challenges, rows_to_discard);
}

#[test]
#[ignore]
fn test_large_octo_with_partial_cache() {
    let (leafs, len, row_count, num_challenges) = { (2097152, 2396745, 8, 2048) };
    for rows_to_discard in 5..7 {
        test_levelcache_v1_tree_from_iter::<U8>(
            leafs,
            len,
            row_count,
            num_challenges,
            std::cmp::min(
                rows_to_discard,
                StoreConfig::default_rows_to_discard(leafs, OCT_ARITY),
            ),
        );
    }
}

#[test]
#[ignore]
fn test_large_octo_with_partial_cache_full() {
    let (leafs, len, row_count, num_challenges, rows_to_discard) =
        { (2097152, 2396745, 8, 2048, 3) };
    test_levelcache_v1_tree_from_iter::<U8>(
        leafs,
        len,
        row_count,
        num_challenges,
        std::cmp::min(
            rows_to_discard,
            StoreConfig::default_rows_to_discard(leafs, OCT_ARITY),
        ),
    );
}

#[test]
#[ignore]
fn test_xlarge_octo_with_partial_cache() {
    let (leafs, len, row_count, num_challenges, rows_to_discard) =
        { (1073741824, 1227133513, 11, 2048, 6) };
    test_levelcache_v1_tree_from_iter::<U8>(
        leafs,
        len,
        row_count,
        num_challenges,
        std::cmp::min(
            rows_to_discard,
            StoreConfig::default_rows_to_discard(leafs, OCT_ARITY),
        ),
    );
}

#[test]
fn test_read_into() {
    let x = [String::from("ars"), String::from("zxc")];
    let mt: MerkleTree<[u8; 16], XOR128, VecStore<_>> =
        MerkleTree::from_data(&x).expect("[test_read_into] failed to create tree");
    let target_data = [
        [0, 97, 114, 115, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 122, 120, 99, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 27, 10, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ];

    let mut read_buffer: [u8; 16] = [0; 16];
    for (pos, &data) in target_data.iter().enumerate() {
        mt.read_into(pos, &mut read_buffer)
            .expect("[test_read_into] couldn't read data from Merkle Tree");
        assert_eq!(read_buffer, data);
    }

    let temp_dir = tempfile::Builder::new()
        .prefix("test_read_into")
        .tempdir()
        .expect("[test_read_into] couldn't create temp_dir");
    let config = StoreConfig::new(
        temp_dir.path(),
        "test-read-into",
        StoreConfig::default_rows_to_discard(x.len(), BINARY_ARITY),
    );

    let mt2: MerkleTree<[u8; 16], XOR128, DiskStore<_>> =
        MerkleTree::from_data_with_config(&x, config)
            .expect("[test_read_into] failed to create tree");
    for (pos, &data) in target_data.iter().enumerate() {
        mt2.read_into(pos, &mut read_buffer)
            .expect("[test_read_into] couldn't read data from Merkle Tree");
        assert_eq!(read_buffer, data);
    }
}

#[test]
fn test_level_cache_tree_v1() {
    let rows_to_discard = 4;
    let count = SMALL_TREE_BUILD * 2;
    test_levelcache_v1_tree_from_iter::<U2>(
        count,
        get_merkle_tree_len(count, BINARY_ARITY)
            .expect("[test_level_cache_tree_v1] failed to get merkle len"),
        get_merkle_tree_row_count(count, BINARY_ARITY),
        count,
        rows_to_discard,
    );
}

#[test]
fn test_level_cache_tree_v2() {
    let a = XOR128::new();
    let count = SMALL_TREE_BUILD * 2;

    let temp_dir = tempfile::Builder::new()
        .prefix("test_level_cache_tree_v2")
        .tempdir()
        .expect("[test_level_cache_tree_v2] couldn't create temp_dir");
    let temp_path = temp_dir.path();

    // Construct and store an MT using a named DiskStore.
    let config = StoreConfig::new(
        &temp_path,
        String::from("test-cache-v2"),
        StoreConfig::default_rows_to_discard(count, BINARY_ARITY),
    );

    let mut mt_disk: MerkleTree<[u8; 16], XOR128, DiskStore<_>> =
        MerkleTree::from_par_iter_with_config(
            (0..count).into_par_iter().map(|x| {
                let mut xor_128 = a;
                xor_128.reset();
                x.hash(&mut xor_128);
                99.hash(&mut xor_128);
                xor_128.hash()
            }),
            config.clone(),
        )
        .expect("[test_level_cache_tree_v2] Failed to create MT");
    assert_eq!(
        mt_disk.len(),
        get_merkle_tree_len(count, BINARY_ARITY)
            .expect("[test_level_cache_tree_v2] failed to get merkle len")
    );

    // Generate proofs on tree.
    for j in 0..mt_disk.leafs() {
        // First generate and validate the proof using the full
        // range of data we have stored on disk (no partial tree
        // is built or used in this case).
        let p = mt_disk
            .gen_proof(j)
            .expect("[test_level_cache_tree_v2] failed to create Merkle Proof");
        assert!(p
            .validate::<XOR128>()
            .expect("[test_level_cache_tree_v2] failed to validate"));
    }

    // Copy the base data from the store to a separate file that
    // is not managed by the store (for use later with an
    // ExternalReader).
    let reader = OpenOptions::new()
        .read(true)
        .open(StoreConfig::data_path(&config.path, &config.id))
        .expect("[test_level_cache_tree_v2] Failed to open base layer data");
    let mut base_layer = vec![0; count * 16];
    reader
        .read_exact_at(&mut base_layer, 0)
        .expect("[test_level_cache_tree_v2] Failed to read");

    let output_file = temp_path.join("base-data-only");
    std::fs::write(&output_file, &base_layer)
        .expect("[test_level_cache_tree_v2] Failed to write output file");

    // Compact the disk store for use as a LevelCacheStore (v2
    // stores only the cached data and requires the ExternalReader
    // for base data retrieval).
    match mt_disk.compact(config.clone(), StoreConfigDataVersion::Two as u32) {
        Ok(x) => assert!(x),
        Err(_) => panic!("[test_level_cache_tree_v2] Compaction failed"), // Could not do any compaction with this configuration.
    }

    // Then re-create an MT using LevelCacheStore and generate all proofs.
    assert!(LevelCacheStore::<[u8; 16], std::fs::File>::is_consistent(
        get_merkle_tree_len(count, BINARY_ARITY)
            .expect("[test_level_cache_tree_v2] failed to get merkle len"),
        BINARY_ARITY,
        &config
    )
    .expect("[test_level_cache_tree_v2] couldn't check if LevelCacheStore is consistent"));
    let level_cache_store: LevelCacheStore<[u8; 16], _> =
        LevelCacheStore::new_from_disk_with_reader(
            get_merkle_tree_len(count, BINARY_ARITY)
                .expect("[test_level_cache_tree_v2] failed to get merkle len"),
            BINARY_ARITY,
            &config,
            ExternalReader::new_from_path(&output_file)
                .expect("[test_level_cache_tree_v2] couldn't instantiate ExternalReader"),
        )
        .expect("[test_level_cache_tree_v2] couldn't check if LevelCacheStore is consistent");

    let mt_level_cache: MerkleTree<[u8; 16], XOR128, LevelCacheStore<_, _>> =
        MerkleTree::from_data_store(level_cache_store, count)
            .expect("[test_level_cache_tree_v2] Failed to create MT from data store");
    assert_eq!(
        mt_level_cache.len(),
        get_merkle_tree_len(count, BINARY_ARITY)
            .expect("[test_level_cache_tree_v2] failed to get merkle len")
    );

    // Generate proofs on tree.
    for j in 0..mt_level_cache.leafs() {
        let proof = mt_level_cache
            .gen_cached_proof(j, None)
            .expect("[test_level_cache_tree_v2] Failed to generate proof and partial tree");
        assert!(proof
            .validate::<XOR128>()
            .expect("[test_level_cache_tree_v2] failed to validate"));
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
        let row_count = get_merkle_tree_row_count(count, BINARY_ARITY);

        // Test a range of row_counts to cache above the base (for
        // different partial tree sizes).
        //
        // compaction correctly fails at 0 and row_count
        for i in 1..row_count - 1 {
            let temp_dir = tempfile::Builder::new()
                .prefix("test_various_trees_with_partial_cache")
                .tempdir()
                .expect("[test_various_trees_with_partial_cache_v2_only] couldn't create temp_dr");
            let temp_path = temp_dir.path();

            // Construct and store an MT using a named DiskStore.
            let config = StoreConfig::new(
                &temp_path,
                format!("test-partial-cache-{}", i),
                std::cmp::min(i, StoreConfig::default_rows_to_discard(count, BINARY_ARITY)),
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
                .expect("[test_various_trees_with_partial_cache_v2_only] failed to create merkle tree from iter with config");

            // Sanity check loading the store from disk and then
            // re-creating the MT from it.
            assert!(DiskStore::<[u8; 16]>::is_consistent(
                get_merkle_tree_len(count, BINARY_ARITY).expect("[test_various_trees_with_partial_cache_v2_only] failed to get merkle len"),
                BINARY_ARITY,
                &config
            )
            .expect("[test_various_trees_with_partial_cache_v2_only] couldn't check if DiskStore is consistent"));
            let store = DiskStore::new_from_disk(
                get_merkle_tree_len(count, BINARY_ARITY).expect(
                    "[test_various_trees_with_partial_cache_v2_only] failed to get merkle len",
                ),
                BINARY_ARITY,
                &config,
            )
            .expect(
                "[test_various_trees_with_partial_cache_v2_only] couldn't instantiate DiskStore",
            );
            let mt_cache2: MerkleTree<[u8; 16], XOR128, DiskStore<_>> =
                MerkleTree::from_data_store(store, count).expect("[test_various_trees_with_partial_cache_v2_only] couldn't instantiate Merkle Tree");

            assert_eq!(mt_cache.len(), mt_cache2.len());
            assert_eq!(mt_cache.leafs(), mt_cache2.leafs());

            assert_eq!(
                mt_cache.len(),
                get_merkle_tree_len(count, BINARY_ARITY).expect(
                    "[test_various_trees_with_partial_cache_v2_only] failed to get merkle len"
                )
            );
            assert_eq!(mt_cache.leafs(), count);

            for j in 0..mt_cache.leafs() {
                // First generate and validate the proof using the full
                // range of data we have stored on disk (no partial tree
                // is built or used in this case).
                let p = mt_cache.gen_proof(j).expect("[test_various_trees_with_partial_cache_v2_only] couldn't generate Merkle Proof");
                assert!(p
                    .validate::<XOR128>()
                    .expect("[test_various_trees_with_partial_cache_v2_only] failed to validate"));
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
                .expect("[test_various_trees_with_partial_cache_v2_only] Failed to open base layer data");
            let mut base_layer = vec![0; count * 16];
            reader
                .read_exact_at(&mut base_layer, 0)
                .expect("[test_various_trees_with_partial_cache_v2_only] Failed to read");

            let output_file = temp_path.join("base-data-only");
            std::fs::write(&output_file, &base_layer).expect(
                "[test_various_trees_with_partial_cache_v2_only] Failed to write output file",
            );

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
                Ok(x) => assert!(x),
                Err(_) => {
                    panic!("[test_various_trees_with_partial_cache_v2_only] Compaction failed")
                } // Could not do any compaction with this configuration.
            }

            // Then re-create an MT using LevelCacheStore and generate all proofs.
            assert!(LevelCacheStore::<[u8; 16], std::fs::File>::is_consistent(
                get_merkle_tree_len(count, BINARY_ARITY).expect("[test_various_trees_with_partial_cache_v2_only] failed to get merkle len"),
                BINARY_ARITY,
                &config
            )
            .expect("[test_various_trees_with_partial_cache_v2_only] couldn't check if LevelCacheStore is consistent"));
            let level_cache_store: LevelCacheStore<[u8; 16], _> =
                LevelCacheStore::new_from_disk_with_reader(
                    get_merkle_tree_len(count, BINARY_ARITY).expect("[test_various_trees_with_partial_cache_v2_only] failed to get merkle len"),
                    BINARY_ARITY,
                    &config,
                    ExternalReader::new_from_path(&output_file).expect("[test_various_trees_with_partial_cache_v2_only] couldn't instantiate ExternalReader"),
                )
                .expect("[test_various_trees_with_partial_cache_v2_only] couldn't instantiate LevelCacheStore");

            let mt_level_cache: MerkleTree<[u8; 16], XOR128, LevelCacheStore<_, _>> =
                MerkleTree::from_data_store(level_cache_store, count)
                    .expect("[test_various_trees_with_partial_cache_v2_only] Failed to revive LevelCacheStore after compaction");

            // Sanity check that after rebuild, the new MT properties match the original.
            assert_eq!(mt_level_cache.len(), mt_cache_len);
            assert_eq!(mt_level_cache.leafs(), mt_cache.leafs());

            // This is the proper way to generate a single proof using
            // the LevelCacheStore.  The optimal partial tree is
            // built, given the cached parameters and the properly
            // recorded LevelCacheStore.
            for j in 0..mt_level_cache.leafs() {
                let proof = mt_level_cache
                    .gen_cached_proof(j, Some(i))
                    .expect("[test_various_trees_with_partial_cache_v2_only] Failed to generate proof and partial tree");
                assert!(proof
                    .validate::<XOR128>()
                    .expect("[test_various_trees_with_partial_cache_v2_only] failed to validate"));
            }

            // Delete the single store backing this MT (for this test,
            // the DiskStore is compacted and then shared with the
            // LevelCacheStore, so it's still a single store on disk).
            mt_level_cache.delete(config.clone()).expect(
                "[test_various_trees_with_partial_cache_v2_only] Failed to delete test store",
            );

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
    let store: DiskStore<[u8; 16]> = DiskStore::new_from_slice(128, &data)
        .expect("[test_parallel_iter_disk_1] couldn't instantiate DiskStore");

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
        let store: DiskStore<[u8; 16]> = DiskStore::new_from_slice(size, &data)
            .expect("[test_parallel_iter_disk_2] couldn't instantiate DiskStore");

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

pub const SIZE: usize = 0x10;

pub const BINARY_ARITY: usize = 2;
pub const OCT_ARITY: usize = 8;

pub type Item = [u8; SIZE];

#[derive(Debug, Copy, Clone, Default)]
pub struct XOR128 {
    data: Item,
    i: usize,
}

/// Implementation of XOR128 Hasher
///
/// It is used for testing purposes
impl XOR128 {
    pub fn new() -> XOR128 {
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
        unimplemented!(
            "Hasher's contract (finish function is not used) is deliberately broken by design"
        )
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

/// Implementation of SHA-256 Hasher
///
/// It is used for testing purposes
#[derive(Clone)]
pub struct Sha256Hasher {
    engine: Sha256,
}

impl Sha256Hasher {
    pub fn new() -> Sha256Hasher {
        Sha256Hasher {
            engine: Sha256::new(),
        }
    }
}

impl Debug for Sha256Hasher {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.write_str("Sha256Hasher")
    }
}

impl Default for Sha256Hasher {
    fn default() -> Self {
        Sha256Hasher::new()
    }
}

impl Hasher for Sha256Hasher {
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

impl Algorithm<Item> for Sha256Hasher {
    fn hash(&mut self) -> Item {
        let mut result: Item = Item::default();
        let item_size = result.len();
        let hash_output = self.engine.clone().finalize().to_vec();
        self.engine.reset();
        if item_size < hash_output.len() {
            result.copy_from_slice(&hash_output.as_slice()[0..item_size]);
        } else {
            result.copy_from_slice(hash_output.as_slice())
        }
        result
    }
}

pub fn get_vec_tree_from_slice<E: Element, A: Algorithm<E>, U: Unsigned>(
    leafs: usize,
) -> MerkleTree<E, A, VecStore<E>, U> {
    let mut x = Vec::with_capacity(leafs);
    for i in 0..leafs {
        x.push(i * 93);
    }
    MerkleTree::from_data(&x).expect("[get_vec_tree_from_slice] failed to create tree from slice")
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Default, Debug)]
pub struct Item2(pub u64);

impl Element for Item2 {
    fn byte_len() -> usize {
        8
    }

    fn from_slice(bytes: &[u8]) -> Self {
        Item2(NativeEndian::read_u64(bytes))
    }

    fn copy_to_slice(&self, bytes: &mut [u8]) {
        NativeEndian::write_u64(bytes, 1);
    }
}

impl AsRef<[u8]> for Item2 {
    #[allow(unsafe_code)]
    fn as_ref(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(mem::transmute(&self.0), 8) }
    }
}

impl PartialEq<u64> for Item2 {
    fn eq(&self, other: &u64) -> bool {
        self.0 == *other
    }
}

impl From<u64> for Item2 {
    fn from(x: u64) -> Self {
        Item2(x)
    }
}

impl From<Item2> for u64 {
    fn from(x: Item2) -> u64 {
        x.0
    }
}

impl<A: Algorithm<Item2>> Hashable<A> for Item2 {
    fn hash(&self, state: &mut A) {
        state.write_u64(self.0)
    }
}

#[test]
fn test_simple_tree() {
    impl Algorithm<Item2> for DefaultHasher {
        #[inline]
        fn hash(&mut self) -> Item2 {
            Item2(self.finish())
        }

        #[inline]
        fn reset(&mut self) {
            *self = DefaultHasher::default()
        }
    }

    let answer: Vec<Vec<u64>> = vec![
        vec![
            18161131233134742049,
            15963407371316104707,
            8061613778145084206,
        ],
        vec![
            18161131233134742049,
            15963407371316104707,
            2838807777806232157,
            2838807777806232157,
            8061613778145084206,
            8605533607343419251,
            12698627859487956302,
        ],
        vec![
            18161131233134742049,
            15963407371316104707,
            2838807777806232157,
            4356248227606450052,
            8061613778145084206,
            6971098229507888078,
            452397072384919190,
        ],
        vec![
            18161131233134742049,
            15963407371316104707,
            2838807777806232157,
            4356248227606450052,
            5528330654215492654,
            5528330654215492654,
            8061613778145084206,
            6971098229507888078,
            7858164776785041459,
            7858164776785041459,
            452397072384919190,
            13691461346724970593,
            12928874197991182098,
        ],
        vec![
            18161131233134742049,
            15963407371316104707,
            2838807777806232157,
            4356248227606450052,
            5528330654215492654,
            11057097817362835984,
            8061613778145084206,
            6971098229507888078,
            6554444691020019791,
            6554444691020019791,
            452397072384919190,
            2290028692816887453,
            151678167824896071,
        ],
        vec![
            18161131233134742049,
            15963407371316104707,
            2838807777806232157,
            4356248227606450052,
            5528330654215492654,
            11057097817362835984,
            15750323574099240302,
            15750323574099240302,
            8061613778145084206,
            6971098229507888078,
            6554444691020019791,
            13319587930734024288,
            452397072384919190,
            15756788945533226834,
            8300325667420840753,
        ],
    ];
    for items in [2, 4].iter() {
        let mut a = DefaultHasher::new();
        let mt: MerkleTree<Item2, DefaultHasher, VecStore<Item2>> = MerkleTree::try_from_iter(
            [1, 2, 3, 4, 5, 6, 7, 8]
                .iter()
                .map(|x| {
                    a.reset();
                    x.hash(&mut a);
                    Ok(a.hash())
                })
                .take(*items),
        )
        .expect("[test_simple_tree] couldn't instantiate Merkle Tree");

        assert_eq!(mt.leafs(), *items);
        assert_eq!(mt.row_count(), log2_pow2(next_pow2(mt.len())));
        assert_eq!(
            mt.read_range(0, mt.len())
                .expect("[test_simple_tree] couldn't read elements from Merkle Tree"),
            answer[*items - 2].as_slice()
        );

        for i in 0..mt.leafs() {
            let p = mt
                .gen_proof(i)
                .expect("[test_simple_tree] couldn't generate Merkle Proof");
            assert!(p
                .validate::<DefaultHasher>()
                .expect("[test_simple_tree] failed to validate"));
        }
    }
}

/// Custom merkle hash util test
#[derive(Debug, Clone, Default)]
#[allow(clippy::upper_case_acronyms)]
struct CMH(DefaultHasher);

impl CMH {
    pub fn new() -> CMH {
        CMH(DefaultHasher::new())
    }
}

impl Hasher for CMH {
    #[inline]
    fn write(&mut self, msg: &[u8]) {
        <dyn Hasher>::write(&mut self.0, msg)
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.0.finish()
    }
}

impl Algorithm<Item2> for CMH {
    #[inline]
    fn hash(&mut self) -> Item2 {
        Item2(self.finish())
    }

    #[inline]
    fn reset(&mut self) {
        *self = CMH::default()
    }

    #[inline]
    fn leaf(&mut self, leaf: Item2) -> Item2 {
        Item2(leaf.0 & 0xff)
    }

    #[inline]
    fn node(&mut self, left: Item2, right: Item2, _height: usize) -> Item2 {
        self.write(&[1u8]);
        self.write(left.as_ref());
        self.write(&[2u8]);
        self.write(right.as_ref());
        Item2(self.hash().0 & 0xffff)
    }
}

#[test]
fn test_custom_merkle_hasher() {
    let mut a = CMH::new();
    let mt: MerkleTree<Item2, CMH, VecStore<Item2>> =
        MerkleTree::try_from_iter([1, 2, 3, 4, 5, 6, 7, 8].iter().map(|x| {
            a.reset();
            x.hash(&mut a);
            Ok(a.hash())
        }))
        .expect("[test_custom_merkle_hasher] couldn't instantiate Merkle Tree");

    assert_eq!(
        mt.read_range(0, 3)
            .expect("[test_custom_merkle_hasher] couldn't read elements from Merkle Tree")
            .iter()
            .take(mt.leafs())
            .filter(|&&x| x.0 > 255)
            .count(),
        0
    );
    assert_eq!(
        mt.read_range(0, 3)
            .expect("[test_custom_merkle_hasher] couldn't read elements from Merkle Tree")
            .iter()
            .filter(|&&x| x.0 > 65535)
            .count(),
        0
    );
}
