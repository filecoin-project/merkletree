#![cfg(test)]
#![cfg(not(tarpaulin_include))]

use std::fmt;
use std::fmt::{Debug, Formatter};
use std::hash::Hasher;

use crypto::digest::Digest;
use crypto::sha2::Sha256;
use typenum::marker_traits::Unsigned;

use crate::hash::Algorithm;
use crate::merkle::{Element, MerkleTree};
use crate::store::VecStore;

pub const SIZE: usize = 0x10;

pub const BINARY_ARITY: usize = 2;
pub const QUAD_ARITY: usize = 4;
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
#[derive(Copy, Clone)]
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
        self.engine.input(bytes)
    }
}

impl Algorithm<Item> for Sha256Hasher {
    fn hash(&mut self) -> Item {
        let mut result: Item = Item::default();
        let item_size = result.len();
        let mut hash_output = vec![0u8; self.engine.output_bytes()];
        self.engine.result(&mut hash_output);
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
    MerkleTree::from_data(&x).expect("failed to create tree from slice")
}
