#![cfg(test)]

use hash::*;
use merkle::{log2_pow2, next_pow2};
use merkle::{Element, MerkleTree, SMALL_TREE_BUILD};
use merkle::{FromIndexedParallelIterator, FromIteratorWithConfig};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::iter::FromIterator;
use std::fmt;
use std::hash::Hasher;
use store::{Store, DiskStore, VecStore, LevelCacheStore};
use store::{StoreConfig, DEFAULT_CACHED_ABOVE_BASE_LAYER};

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
    let current_path = temp_dir.path().to_str().unwrap().to_string();
    let config = StoreConfig::new(
        current_path, String::from("test-read-into"),
        DEFAULT_CACHED_ABOVE_BASE_LAYER);

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
                .take(items)
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
            let cached_above_base_levels = 2;
            let mt2: MerkleTree<[u8; 16], XOR128, DiskStore<_>> =
                MerkleTree::from_byte_slice(&leafs);
            assert_eq!(mt2.leafs(), items);
            assert_eq!(mt2.height(), log2_pow2(next_pow2(mt2.len())));
            for i in 0..mt2.leafs() {
                let p = mt2.gen_proof(i);
                assert!(p.validate::<XOR128>());
                if items >= 7 {
                    // When the tree is large enough to have some
                    // cached levels, test the proof generation from a
                    // partial store.
                    let pat = mt2.gen_proof_and_partial_tree(i, cached_above_base_levels)
                        .unwrap();
                    assert!(pat.proof.validate::<XOR128>());
                }
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
    let mut a = XOR128::new();

    let min_count = SMALL_TREE_BUILD / 16;
    let max_count = SMALL_TREE_BUILD * 16;
    let mut count = min_count;

    let pow = next_pow2(min_count);
    let height = log2_pow2(2 * pow);

    let cached_above_base_levels = height - 1;

    // Test a range of tree sizes, given a range of leaf elements.
    while count <= max_count {

        // Test a range of heights to cache above the base (for
        // different partial tree sizes).
        for i in 0..cached_above_base_levels {

            let temp_dir = tempdir::TempDir::new(
                "test_various_trees_with_partial_cache").unwrap();
            let current_path = temp_dir.path().to_str().unwrap().to_string();

            // Construct and store an MT using a named DiskStore.
            let config = StoreConfig::new(
                current_path.clone(), String::from("test-cache"), i);
            let mut mt_cache: MerkleTree<[u8; 16], XOR128, DiskStore<_>> =
                MerkleTree::from_iter_with_config((0..count).map(|x| {
                    a.reset();
                    x.hash(&mut a);
                    count.hash(&mut a);
                    a.hash()
                }), config.clone());

            // Sanity check loading the store from disk and then
            // re-creating the MT from it.
            let store = DiskStore::new_from_disk(mt_cache.len(), config.clone()).unwrap();
            let mt_cache2: MerkleTree<[u8; 16], XOR128, DiskStore<_>> =
                MerkleTree::from_data_store(store, mt_cache.len());

            assert_eq!(mt_cache.len(), mt_cache2.len());
            assert_eq!(mt_cache.leafs(), mt_cache2.leafs());

            assert_eq!(mt_cache.len(), 2 * count - 1);
            assert_eq!(mt_cache.leafs(), count);

            // Generate and validate proof on the first element.
            let p = mt_cache.gen_proof(0);
            assert!(p.validate::<XOR128>());

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
            let pat1 = mt_cache.gen_proof_and_partial_tree(0, i)
                .unwrap();
            assert!(pat1.proof.validate::<XOR128>());

            // Same as above, but generate and validate the proof on the
            // first element of the second data half and retrieve the
            // partial tree needed for future proofs in that range.
            let pat2 = mt_cache.gen_proof_and_partial_tree(mt_cache.leafs() / 2, i)
                .unwrap();
            assert!(pat2.proof.validate::<XOR128>());

            for j in 1..mt_cache.leafs() {
                // First generate and validate the proof using the full
                // range of data we have stored on disk (no partial tree
                // is built or used in this case).
                let p = mt_cache.gen_proof(j);
                assert!(p.validate::<XOR128>());

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
                Store::new_from_disk(count, config.clone()).unwrap();
            let mt_level_cache: MerkleTree<[u8; 16], XOR128, LevelCacheStore<_>> =
                MerkleTree::from_data_store(level_cache_store, count);

            // Sanity check that after rebuild, the new MT properties match the original.
            assert_eq!(mt_level_cache.len(), mt_cache_len);
            assert_eq!(mt_level_cache.leafs(), mt_cache.leafs());

            // This is the proper way to generate a single proof using the
            // LevelCacheStore.  If generating more than 1 proof, it's
            // terribly slow though since the partial tree(s) generated
            // are not re-used across calls.  For that example, see the
            // next test below.
            //
            // This is commented out because it adds a lot of runtime waiting.
            // for j in 0..mt_level_cache.leafs() {
            //     let pat = mt_level_cache.gen_proof_and_partial_tree(j, i);
            //     assert!(pat.proof.validate::<XOR128>());
            // }

            // Optimized proof generation based on simple generation pattern:
            let pat1 = mt_level_cache
                .gen_proof_and_partial_tree(0, i)
                .unwrap();
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
                    let p1 = mt_level_cache
                        .gen_proof_with_partial_tree(j, i, &pat1.merkle_tree);
                    assert!(p1.validate::<XOR128>());
                } else {
                    let p2 = mt_level_cache
                        .gen_proof_with_partial_tree(j, i, &pat2.merkle_tree);
                    assert!(p2.validate::<XOR128>());
                }
            }
        }

        count <<= 1;
    }
}
