use crate::hash::*;
use crate::merkle::{Element, MerkleTree};
use crate::store::VecStore;
use std::fmt;
use std::hash::Hasher;
use typenum::marker_traits::Unsigned;

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

pub fn get_vec_tree_from_slice<U: Unsigned>(
    leafs: usize,
) -> MerkleTree<Item, XOR128, VecStore<Item>, U> {
    let mut x = Vec::with_capacity(leafs);
    for i in 0..leafs {
        x.push(i * 93);
    }
    MerkleTree::from_data(&x).expect("failed to create tree from slice")
}
