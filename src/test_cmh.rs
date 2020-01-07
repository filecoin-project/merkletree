#![cfg(test)]

use crate::hash::{Algorithm, Hashable};
use crate::merkle::MerkleTree;
use crate::store::VecStore;
use crate::test_item::Item;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;

/// Custom merkle hash util test
#[derive(Debug, Clone, Default)]
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

impl Algorithm<Item> for CMH {
    #[inline]
    fn hash(&mut self) -> Item {
        Item(self.finish())
    }

    #[inline]
    fn reset(&mut self) {
        *self = CMH::default()
    }

    #[inline]
    fn leaf(&mut self, leaf: Item) -> Item {
        Item(leaf.0 & 0xff)
    }

    #[inline]
    fn node(&mut self, left: Item, right: Item, _height: usize) -> Item {
        self.write(&[1u8]);
        self.write(left.as_ref());
        self.write(&[2u8]);
        self.write(right.as_ref());
        Item(self.hash().0 & 0xffff)
    }
}

#[test]
fn test_custom_merkle_hasher() {
    let mut a = CMH::new();
    let mt: MerkleTree<Item, CMH, VecStore<_>> =
        MerkleTree::try_from_iter([1, 2, 3, 4, 5, 6, 7, 8].iter().map(|x| {
            a.reset();
            x.hash(&mut a);
            Ok(a.hash())
        }))
        .unwrap();

    assert_eq!(
        mt.read_range(0, 3)
            .unwrap()
            .iter()
            .take(mt.leafs())
            .filter(|&&x| x.0 > 255)
            .count(),
        0
    );
    assert_eq!(
        mt.read_range(0, 3)
            .unwrap()
            .iter()
            .filter(|&&x| x.0 > 65535)
            .count(),
        0
    );
}
