#![cfg(test)]

use hash::*;
use merkle::{log2_pow2, next_pow2};
use merkle::{Element, MerkleTree, SMALL_TREE_BUILD};
use merkle::{FromIndexedParallelIterator, FromIteratorWithConfig};
use rayon::iter::{plumbing::*, IntoParallelIterator, ParallelIterator};
use std::fmt;
use std::hash::Hasher;
use std::iter::FromIterator;
use store::{
    DiskStore, DiskStoreProducer, LevelCacheStore, Store, StoreConfig, VecStore,
    DEFAULT_CACHED_ABOVE_BASE_LAYER,
};

const SIZE: usize = 0x10;

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

#[test]
fn test_from_slice() {
    let x = [String::from("ars"), String::from("zxc")];
    let mt: MerkleTree<[u8; 16], XOR128, VecStore<_>> = MerkleTree::from_data(&x);
    assert_eq!(
        mt.read_range(0, 3),
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
}

#[test]
fn test_read_into() {
    let x = [String::from("ars"), String::from("zxc")];
    let mt: MerkleTree<[u8; 16], XOR128, VecStore<_>> = MerkleTree::from_data(&x);
    let target_data = [
        [0, 97, 114, 115, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 122, 120, 99, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 27, 10, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ];

    let mut read_buffer: [u8; 16] = [0; 16];
    for (pos, &data) in target_data.iter().enumerate() {
        mt.read_into(pos, &mut read_buffer);
        assert_eq!(read_buffer, data);
    }

    let temp_dir = tempdir::TempDir::new("test_read_into").unwrap();
    let config = StoreConfig::new(
        temp_dir.path(),
        "test-read-into",
        DEFAULT_CACHED_ABOVE_BASE_LAYER,
    );

    let mt2: MerkleTree<[u8; 16], XOR128, DiskStore<_>> =
        MerkleTree::from_data_with_config(&x, config);
    for (pos, &data) in target_data.iter().enumerate() {
        mt2.read_into(pos, &mut read_buffer);
        assert_eq!(read_buffer, data);
    }
}

#[test]
fn test_from_iter() {
    let mut a = XOR128::new();
    let mt: MerkleTree<[u8; 16], XOR128, VecStore<_>> =
        MerkleTree::from_iter(["a", "b", "c"].iter().map(|x| {
            a.reset();
            x.hash(&mut a);
            a.hash()
        }));
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

    for items in 2..8 {
        let mut a = XOR128::new();
        let mt_base: MerkleTree<[u8; 16], XOR128, VecStore<_>> = MerkleTree::from_iter(
            [1, 2, 3, 4, 5, 6, 7, 8]
                .iter()
                .map(|x| {
                    a.reset();
                    x.hash(&mut a);
                    a.hash()
                })
                .take(items),
        );

        assert_eq!(mt_base.leafs(), items);
        assert_eq!(mt_base.height(), log2_pow2(next_pow2(mt_base.len())));
        assert_eq!(
            mt_base.read_range(0, mt_base.len()),
            answer[items - 2].as_slice()
        );
        assert_eq!(mt_base.read_at(0), mt_base.read_at(0));

        for i in 0..mt_base.leafs() {
            let p = mt_base.gen_proof(i);
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
            .take(items)
            .map(|item| {
                a2.reset();
                a2.leaf(item).as_ref().to_vec()
            })
            .flatten()
            .collect();
        {
            let mt1: MerkleTree<[u8; 16], XOR128, VecStore<_>> =
                MerkleTree::from_byte_slice(&leafs);
            assert_eq!(mt1.leafs(), items);
            assert_eq!(mt1.height(), log2_pow2(next_pow2(mt1.len())));
            assert_eq!(
                mt_base.read_range(0, mt_base.len()),
                answer[items - 2].as_slice()
            );

            for i in 0..mt1.leafs() {
                let p = mt1.gen_proof(i);
                assert!(p.validate::<XOR128>());
            }
        }

        {
            let mt2: MerkleTree<[u8; 16], XOR128, DiskStore<_>> =
                MerkleTree::from_byte_slice(&leafs);
            assert_eq!(mt2.leafs(), items);
            assert_eq!(mt2.height(), log2_pow2(next_pow2(mt2.len())));
            for i in 0..mt2.leafs() {
                let p = mt2.gen_proof(i);
                assert!(p.validate::<XOR128>());
            }
        }
    }
}

#[test]
fn test_large_tree() {
    let mut a = XOR128::new();
    let count = SMALL_TREE_BUILD * 2;

    // The large `build` algorithm uses a ad hoc parallel solution (instead
    // of the standard `par_iter()` from Rayon) so test these many times
    // to increase the chances of finding a data-parallelism bug. (We're
    // using a size close to the `SMALL_TREE_BUILD` threshold so this
    // shouldn't increase test times considerably.)
    for i in 0..100 {
        let mt_vec: MerkleTree<[u8; 16], XOR128, VecStore<_>> =
            MerkleTree::from_iter((0..count).map(|x| {
                a.reset();
                x.hash(&mut a);
                i.hash(&mut a);
                a.hash()
            }));
        assert_eq!(mt_vec.len(), 2 * count - 1);

        let mt_disk: MerkleTree<[u8; 16], XOR128, DiskStore<_>> =
            MerkleTree::from_par_iter((0..count).into_par_iter().map(|x| {
                let mut xor_128 = a.clone();
                xor_128.reset();
                x.hash(&mut xor_128);
                i.hash(&mut xor_128);
                xor_128.hash()
            }));
        assert_eq!(mt_disk.len(), 2 * count - 1);
    }
}

#[test]
fn test_various_trees_with_partial_cache() {
    env_logger::init();
    let mut a = XOR128::new();

    // Attempt to allow this test to move along relatively quickly.
    let min_count = SMALL_TREE_BUILD / 16;
    let max_count = SMALL_TREE_BUILD * 4;
    let mut count = min_count;

    // Test a range of tree sizes, given a range of leaf elements.
    while count <= max_count {
        let pow = next_pow2(count);
        let height = log2_pow2(2 * pow);

        let cached_above_base_levels = height;

        // Test a range of heights to cache above the base (for
        // different partial tree sizes).
        for i in 0..cached_above_base_levels {
            let temp_dir = tempdir::TempDir::new("test_various_trees_with_partial_cache").unwrap();

            // Construct and store an MT using a named DiskStore.
            let config = StoreConfig::new(
                temp_dir.path(),
                String::from(format!("test-cache-{}", i)),
                i,
            );
            let mut mt_cache: MerkleTree<[u8; 16], XOR128, DiskStore<_>> =
                MerkleTree::from_iter_with_config(
                    (0..count).map(|x| {
                        a.reset();
                        x.hash(&mut a);
                        count.hash(&mut a);
                        a.hash()
                    }),
                    config.clone(),
                );

            // Sanity check loading the store from disk and then
            // re-creating the MT from it.
            let store = DiskStore::new_from_disk(2 * count - 1, &config).unwrap();
            let mt_cache2: MerkleTree<[u8; 16], XOR128, DiskStore<_>> =
                MerkleTree::from_data_store(store, count);

            assert_eq!(mt_cache.len(), mt_cache2.len());
            assert_eq!(mt_cache.leafs(), mt_cache2.leafs());

            assert_eq!(mt_cache.len(), 2 * count - 1);
            assert_eq!(mt_cache.leafs(), count);

            // Generate and validate proof on the first element.
            let p = mt_cache.gen_proof(0);
            assert!(p.validate::<XOR128>());

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

            for j in 1..mt_cache.leafs() {
                // First generate and validate the proof using the full
                // range of data we have stored on disk (no partial tree
                // is built or used in this case).
                let p = mt_cache.gen_proof(j);
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

            // Compact the newly created DiskStore into the
            // LevelCacheStore format.  This uses information from the
            // Config to properly shape the compacted data for later
            // access using the LevelCacheStore interface.
            match mt_cache.compact(config.clone()) {
                Ok(x) => assert_eq!(x, true),
                Err(_) => continue, // Could not do any compaction with this configuration.
            }

            // Then re-create an MT using LevelCacheStore and generate all proofs.
            let level_cache_store: LevelCacheStore<[u8; 16]> =
                Store::new_from_disk(2 * count - 1, &config).unwrap();
            let mt_level_cache: MerkleTree<[u8; 16], XOR128, LevelCacheStore<_>> =
                MerkleTree::from_data_store(level_cache_store, count);

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
