#![cfg(test)]
#![cfg(feature = "bitcoin")]

use merkletree::hash::{Algorithm, Hashable};
use merkletree::merkle::MerkleTree;
use merkletree::store::VecStore;
use ring::digest::{Context, SHA256};
use std::fmt;
use std::hash::Hasher;

#[derive(Clone)]
struct RingBitcoinAlgorithm(Context);

impl RingBitcoinAlgorithm {
    fn new() -> RingBitcoinAlgorithm {
        RingBitcoinAlgorithm(Context::new(&SHA256))
    }
}

impl Default for RingBitcoinAlgorithm {
    fn default() -> RingBitcoinAlgorithm {
        RingBitcoinAlgorithm::new()
    }
}

impl Hasher for RingBitcoinAlgorithm {
    #[inline]
    fn write(&mut self, msg: &[u8]) {
        self.0.update(msg)
    }

    #[inline]
    fn finish(&self) -> u64 {
        unimplemented!()
    }
}

type RingSHA256Hash = [u8; 32];

impl Algorithm<RingSHA256Hash> for RingBitcoinAlgorithm {
    /// ring.Context is not reusable after finalization (finish(self)),
    /// and having hash(&self) requires first to clone() the context.
    /// The context can't be moved away out of the struct field to
    /// be created at site due to the semantics of Hasher trait:
    /// Hasher trait has to have a context to write states updates.
    ///
    /// someone should hack it somehow.
    ///
    /// or better change signature in `ring` library of `finish()` to
    /// be `finish(&mut self)` to mark state as finalized or reset it.
    #[inline]
    fn hash(&mut self) -> RingSHA256Hash {
        let h1 = self.0.clone().finish();

        // double sha256
        let mut c = Context::new(&SHA256);
        c.update(h1.as_ref());

        let mut h = [0u8; 32];
        h.copy_from_slice(c.finish().as_ref());
        h
    }

    #[inline]
    fn reset(&mut self) {
        self.0 = Context::new(&SHA256);
    }

    fn leaf(&mut self, leaf: RingSHA256Hash) -> RingSHA256Hash {
        leaf
    }

    fn node(
        &mut self,
        left: RingSHA256Hash,
        right: RingSHA256Hash,
        height: usize,
    ) -> RingSHA256Hash {
        height.hash(self);

        left.hash(self);
        right.hash(self);
        self.hash()
    }
}

struct HexSlice<'a>(&'a [u8]);

impl<'a> HexSlice<'a> {
    fn new<T>(data: &'a T) -> HexSlice<'a>
    where
        T: ?Sized + AsRef<[u8]> + 'a,
    {
        HexSlice(data.as_ref())
    }
}

/// reverse order
impl<'a> fmt::Display for HexSlice<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let len = self.0.len();
        for i in 0..len {
            let byte = self.0[len - 1 - i];
            write!(f, "{:x}{:x}", byte >> 4, byte & 0xf)?;
        }
        Ok(())
    }
}

/// [](https://bitcoin.stackexchange.com/questions/5671/how-do-you-perform-double-sha-256-encoding)
#[test]
fn test_ring_bitcoin_leaf_hash() {
    let mut a = RingBitcoinAlgorithm::new();
    "hello".hash(&mut a);
    let h1 = a.hash();
    assert_eq!(
        format!("{}", HexSlice::new(h1.as_ref())),
        "503d8319a48348cdc610a582f7bf754b5833df65038606eb48510790dfc99595"
    );
}

/*
/// [](http://chimera.labs.oreilly.com/books/1234000001802/ch07.html#merkle_trees)
#[test]
fn test_ring_bitcoin_node() {
    let mut h1 = [0u8; 32];
    let mut h2 = [0u8; 32];
    let mut h3 = [0u8; 32];
    h1[0] = 0x00;
    h2[0] = 0x11;
    h3[0] = 0x22;

    let mut a = RingBitcoinAlgorithm::new();
    let h11 = h1;
    let h12 = h2;
    let h13 = h3;
    let h21 = a.node(h11, h12, 0);
    a.reset();
    let h22 = a.node(h13, h13, 0);
    a.reset();
    let h31 = a.node(h21, h22, 1);
    a.reset();

    assert_eq!(
        format!("{}", HexSlice::new(h21.as_ref())),
        "09545bdfc187478f394589877e39b91aa6d0ea95c2defe6ce8dd3f8bf4e8e8ea"
    );
    assert_eq!(
        format!("{}", HexSlice::new(h22.as_ref())),
        "9300c41775d325ddd2d43daf268b346277edc4f744618ea19770ee310a6e04a1"
    );
    assert_eq!(
        format!("{}", HexSlice::new(h31.as_ref())),
        "5ba580c87c9bae263e6186318d77963846ff7a3e92b45f2ed30495ccf52b4731"
    );

    let t: MerkleTree<RingSHA256Hash, RingBitcoinAlgorithm, VecStore<_>> =
        MerkleTree::try_from_iter(vec![h1, h2, h3].into_iter().map(Ok)).unwrap();
    assert_eq!(
        format!("{}", HexSlice::new(t.root().as_ref())),
        "5ba580c87c9bae263e6186318d77963846ff7a3e92b45f2ed30495ccf52b4731"
    );
}
*/
