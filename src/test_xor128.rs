#![cfg(test)]
#![cfg(not(tarpaulin_include))]

use std::fs::OpenOptions;
use std::io::Write as ioWrite;
use std::num::ParseIntError;
use std::os::unix::prelude::FileExt;
use std::path::Path;

use crypto::digest::Digest;
use crypto::sha2::Sha256;
use rayon::iter::{plumbing::Producer, IntoParallelIterator, ParallelIterator};
use typenum::marker_traits::Unsigned;
use typenum::{U2, U3, U4, U5, U7, U8};

use crate::hash::{Algorithm, Hashable};
use crate::merkle::{
    get_merkle_tree_len, get_merkle_tree_row_count, is_merkle_tree_size_valid, Element,
    FromIndexedParallelIterator, MerkleTree,
};
use crate::store::{
    DiskStore, DiskStoreProducer, ExternalReader, LevelCacheStore, ReplicaConfig, Store,
    StoreConfig, StoreConfigDataVersion, VecStore, SMALL_TREE_BUILD,
};
use crate::test_common::{Item, Sha256Hasher, BINARY_ARITY, OCT_ARITY, XOR128};

fn build_disk_tree_from_iter<U: Unsigned>(
    leafs: usize,
    len: usize,
    row_count: usize,
    config: &StoreConfig,
) {
    let branches = U::to_usize();
    assert_eq!(
        len,
        get_merkle_tree_len(leafs, branches).expect("failed to get merkle len")
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
    .expect("failed to create tree");

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
        get_merkle_tree_len(leafs, branches).expect("failed to get merkle len")
    );
    assert_eq!(row_count, get_merkle_tree_row_count(leafs, branches));

    let mut x = Vec::with_capacity(leafs);
    for i in 0..leafs {
        x.push(i * 3);
    }

    let mut mt = MerkleTree::from_data_with_config(&x, config.clone())
        .expect("failed to create tree from slice");

    assert_eq!(mt.len(), len);
    assert_eq!(mt.leafs(), leafs);
    assert_eq!(mt.row_count(), row_count);

    mt.set_external_reader_path(replica_path)
        .expect("Failed to set external reader");

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
        get_merkle_tree_len(leafs, branches).expect("failed to get merkle len")
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
        .expect("failed to create tree");

    assert_eq!(mt.len(), len);
    assert_eq!(mt.leafs(), leafs);
    assert_eq!(mt.row_count(), row_count);

    mt.set_external_reader_path(replica_path)
        .expect("Failed to set external reader");

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
    let temp_dir = tempdir::TempDir::new(&name).unwrap();

    // Construct and store an MT using a named DiskStore.
    let config = StoreConfig::new(temp_dir.path(), name, rows_to_discard);
    build_disk_tree_from_iter::<U>(leafs, len, row_count, &config);

    // Sanity check loading the store from disk and then re-creating
    // the MT from it.
    assert!(DiskStore::<[u8; 16]>::is_consistent(len, branches, &config).unwrap());
    let store = DiskStore::new_from_disk(len, branches, &config).unwrap();
    let mut mt_cache: MerkleTree<[u8; 16], XOR128, DiskStore<_>, U> =
        MerkleTree::from_data_store(store, leafs).unwrap();

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
            .unwrap()
    );
    let level_cache_store: LevelCacheStore<[u8; 16], std::fs::File> =
        LevelCacheStore::new_from_disk(len, branches, &config).unwrap();

    let mt_level_cache: MerkleTree<[u8; 16], XOR128, LevelCacheStore<_, _>, U> =
        MerkleTree::from_data_store(level_cache_store, leafs)
            .expect("Failed to create MT from data store");

    assert_eq!(mt_level_cache.len(), len);
    assert_eq!(mt_level_cache.leafs(), leafs);
    assert_eq!(mt_level_cache.row_count(), row_count);

    // Verify all proofs are still working.
    for i in 0..num_challenges {
        let index = i * (leafs / num_challenges);
        let proof = mt_level_cache
            .gen_cached_proof(index, Some(config.rows_to_discard))
            .expect("Failed to generate proof and partial tree");
        assert!(proof.validate::<XOR128>().expect("failed to validate"));
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
    let temp_dir = tempdir::TempDir::new(test_name).unwrap();

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
            .expect("Failed to generate proof and partial tree");
        assert!(proof.validate::<XOR128>().expect("failed to validate"));
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
    let temp_dir = tempdir::TempDir::new(test_name).unwrap();

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
        let proof = lc_tree
            .gen_cached_proof(index, None)
            .expect("Failed to generate proof and partial tree");
        assert!(proof.validate::<XOR128>().expect("failed to validate"));
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
        decode_hex("000102030405060708090a0b0c0d0e0f").unwrap(),
        decode_hex("00000000000000000000000000000000").unwrap(),
    );
    run_test::<Item, XOR128>(
        5,
        decode_hex("000102030405060708090a0b0c0d0e0f").unwrap(),
        decode_hex("000102030405060708090a0b0c0d0e0f").unwrap(),
    );
    run_test::<Item, XOR128>(
        10,
        decode_hex("000102030405060708090a0b0c0d0e0f").unwrap(),
        decode_hex("00000000000000000000000000000000").unwrap(),
    );
    run_test::<Item, XOR128>(
        371,
        decode_hex("000102030405060708090a0b0c0d0e0f").unwrap(),
        decode_hex("000102030405060708090a0b0c0d0e0f").unwrap(),
    );

    let mut sha256 = Sha256::new();
    let number_of_hashing = 951;
    let input = decode_hex("000102030405060708090a0b0c0d0e0f").unwrap();
    let mut expected_output = vec![0u8; sha256.output_bytes()];
    for _ in 0..number_of_hashing {
        sha256.input(input.as_slice());
    }
    sha256.result(expected_output.as_mut_slice());
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
    let temp_dir = tempdir::TempDir::new("test_compound_levelcache_tree").unwrap();
    let len = get_merkle_tree_len(sub_tree_leafs, branches).expect("failed to get merkle len");
    let row_count = get_merkle_tree_row_count(sub_tree_leafs, branches);

    let replica_path = StoreConfig::data_path(
        temp_dir.path(),
        &format!(
            "{}-{}-{}-{}-replica",
            test_name, sub_tree_leafs, len, row_count
        ),
    );
    let mut f_replica =
        std::fs::File::create(&replica_path).expect("failed to create replica file");

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
            .expect("failed to open store");

        // Use that data store as the replica (concat the data to the replica_path)
        let data: Vec<[u8; 16]> = store
            .read_range(std::ops::Range {
                start: 0,
                end: sub_tree_leafs,
            })
            .expect("failed to read store");
        for element in data {
            f_replica
                .write_all(&element)
                .expect("failed to write replica data");
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
            .expect("Failed to build compound levelcache tree");

    assert_eq!(
        tree.len(),
        (get_merkle_tree_len(sub_tree_leafs, branches).expect("failed to get merkle len")
            * sub_tree_count)
            + 1
    );
    assert_eq!(tree.leafs(), sub_tree_count * sub_tree_leafs);

    for i in 0..tree.leafs() {
        // Make sure all elements are accessible.
        let _ = tree.read_at(i).expect("Failed to read tree element");

        // Make sure all proofs validate.
        let p = tree.gen_cached_proof(i, None).unwrap();
        assert!(p.validate::<XOR128>().expect("failed to validate"));
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
        StoreConfig::default_rows_to_discard(x.len(), BINARY_ARITY),
    );

    let mt2: MerkleTree<[u8; 16], XOR128, DiskStore<_>> =
        MerkleTree::from_data_with_config(&x, config).expect("failed to create tree");
    for (pos, &data) in target_data.iter().enumerate() {
        mt2.read_into(pos, &mut read_buffer).unwrap();
        assert_eq!(read_buffer, data);
    }
}

#[test]
fn test_level_cache_tree_v1() {
    let rows_to_discard = 4;
    let count = SMALL_TREE_BUILD * 2;
    test_levelcache_v1_tree_from_iter::<U2>(
        count,
        get_merkle_tree_len(count, BINARY_ARITY).expect("failed to get merkle len"),
        get_merkle_tree_row_count(count, BINARY_ARITY),
        count,
        rows_to_discard,
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
        .expect("Failed to create MT");
    assert_eq!(
        mt_disk.len(),
        get_merkle_tree_len(count, BINARY_ARITY).expect("failed to get merkle len")
    );

    // Generate proofs on tree.
    for j in 0..mt_disk.leafs() {
        // First generate and validate the proof using the full
        // range of data we have stored on disk (no partial tree
        // is built or used in this case).
        let p = mt_disk.gen_proof(j).unwrap();
        assert!(p.validate::<XOR128>().expect("failed to validate"));
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

    // Compact the disk store for use as a LevelCacheStore (v2
    // stores only the cached data and requires the ExternalReader
    // for base data retrieval).
    match mt_disk.compact(config.clone(), StoreConfigDataVersion::Two as u32) {
        Ok(x) => assert!(x),
        Err(_) => panic!("Compaction failed"), // Could not do any compaction with this configuration.
    }

    // Then re-create an MT using LevelCacheStore and generate all proofs.
    assert!(LevelCacheStore::<[u8; 16], std::fs::File>::is_consistent(
        get_merkle_tree_len(count, BINARY_ARITY).expect("failed to get merkle len"),
        BINARY_ARITY,
        &config
    )
    .unwrap());
    let level_cache_store: LevelCacheStore<[u8; 16], _> =
        LevelCacheStore::new_from_disk_with_reader(
            get_merkle_tree_len(count, BINARY_ARITY).expect("failed to get merkle len"),
            BINARY_ARITY,
            &config,
            ExternalReader::new_from_path(&output_file).unwrap(),
        )
        .unwrap();

    let mt_level_cache: MerkleTree<[u8; 16], XOR128, LevelCacheStore<_, _>> =
        MerkleTree::from_data_store(level_cache_store, count)
            .expect("Failed to create MT from data store");
    assert_eq!(
        mt_level_cache.len(),
        get_merkle_tree_len(count, BINARY_ARITY).expect("failed to get merkle len")
    );

    // Generate proofs on tree.
    for j in 0..mt_level_cache.leafs() {
        let proof = mt_level_cache
            .gen_cached_proof(j, None)
            .expect("Failed to generate proof and partial tree");
        assert!(proof.validate::<XOR128>().expect("failed to validate"));
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
            let temp_dir = tempdir::TempDir::new("test_various_trees_with_partial_cache").unwrap();
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
                .expect("failed to create merkle tree from iter with config");

            // Sanity check loading the store from disk and then
            // re-creating the MT from it.
            assert!(DiskStore::<[u8; 16]>::is_consistent(
                get_merkle_tree_len(count, BINARY_ARITY).expect("failed to get merkle len"),
                BINARY_ARITY,
                &config
            )
            .unwrap());
            let store = DiskStore::new_from_disk(
                get_merkle_tree_len(count, BINARY_ARITY).expect("failed to get merkle len"),
                BINARY_ARITY,
                &config,
            )
            .unwrap();
            let mt_cache2: MerkleTree<[u8; 16], XOR128, DiskStore<_>> =
                MerkleTree::from_data_store(store, count).unwrap();

            assert_eq!(mt_cache.len(), mt_cache2.len());
            assert_eq!(mt_cache.leafs(), mt_cache2.leafs());

            assert_eq!(
                mt_cache.len(),
                get_merkle_tree_len(count, BINARY_ARITY).expect("failed to get merkle len")
            );
            assert_eq!(mt_cache.leafs(), count);

            for j in 0..mt_cache.leafs() {
                // First generate and validate the proof using the full
                // range of data we have stored on disk (no partial tree
                // is built or used in this case).
                let p = mt_cache.gen_proof(j).unwrap();
                assert!(p.validate::<XOR128>().expect("failed to validate"));
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
                Err(_) => panic!("Compaction failed"), // Could not do any compaction with this configuration.
            }

            // Then re-create an MT using LevelCacheStore and generate all proofs.
            assert!(LevelCacheStore::<[u8; 16], std::fs::File>::is_consistent(
                get_merkle_tree_len(count, BINARY_ARITY).expect("failed to get merkle len"),
                BINARY_ARITY,
                &config
            )
            .unwrap());
            let level_cache_store: LevelCacheStore<[u8; 16], _> =
                LevelCacheStore::new_from_disk_with_reader(
                    get_merkle_tree_len(count, BINARY_ARITY).expect("failed to get merkle len"),
                    BINARY_ARITY,
                    &config,
                    ExternalReader::new_from_path(&output_file).unwrap(),
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
                let proof = mt_level_cache
                    .gen_cached_proof(j, Some(i))
                    .expect("Failed to generate proof and partial tree");
                assert!(proof.validate::<XOR128>().expect("failed to validate"));
            }

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
