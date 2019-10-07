#![cfg(test)]

use hash::*;
use store::{DiskStore, MmapStore, VecStore};
use merkle::FromIndexedParallelIterator;
use merkle::{log2_pow2, next_pow2};
use merkle::{Element, MerkleTree, SMALL_TREE_BUILD};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::fmt;
use std::hash::Hasher;
use std::iter::FromIterator;

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

        let mt0: MerkleTree<[u8; 16], XOR128, MmapStore<_>> = MerkleTree::from_iter(
            [1, 2, 3, 4, 5, 6, 7, 8]
                .iter()
                .map(|x| {
                    a.reset();
                    x.hash(&mut a);
                    a.hash()
                })
                .take(items),
        );

        assert_eq!(mt0.leafs(), items);
        assert_eq!(mt0.height(), log2_pow2(next_pow2(mt0.len())));
        // assert_eq!(mt0.as_slice(), answer[items - 2].as_slice());
        // assert_eq!(mt0[0], mt0[0]);

        for i in 0..mt0.leafs() {
            let p = mt0.gen_proof(i);
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
            let mt2: MerkleTree<[u8; 16], XOR128, MmapStore<_>> =
                MerkleTree::from_byte_slice(&leafs);
            assert_eq!(mt2.leafs(), items);
            assert_eq!(mt2.height(), log2_pow2(next_pow2(mt2.len())));
            for i in 0..mt2.leafs() {
                let p = mt2.gen_proof(i);
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

        let mt_mmap: MerkleTree<[u8; 16], XOR128, MmapStore<_>> =
            MerkleTree::from_iter((0..count).map(|x| {
                a.reset();
                x.hash(&mut a);
                i.hash(&mut a);
                a.hash()
            }));
        assert_eq!(mt_mmap.len(), 2 * count - 1);

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
