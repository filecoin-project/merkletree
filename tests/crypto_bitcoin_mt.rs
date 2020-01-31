#![cfg(test)]
#![cfg(feature = "bitcoin")]

use crypto::digest::Digest;
use crypto::sha2::Sha256;
use merkletree::hash::{Algorithm, Hashable};
use std::fmt;
use std::hash::Hasher;

#[derive(Clone)]
struct CryptoBitcoinAlgorithm(Sha256);

impl CryptoBitcoinAlgorithm {
    fn new() -> CryptoBitcoinAlgorithm {
        CryptoBitcoinAlgorithm(Sha256::new())
    }
}

impl Default for CryptoBitcoinAlgorithm {
    fn default() -> CryptoBitcoinAlgorithm {
        CryptoBitcoinAlgorithm::new()
    }
}

impl Hasher for CryptoBitcoinAlgorithm {
    #[inline]
    fn write(&mut self, msg: &[u8]) {
        self.0.input(msg)
    }

    #[inline]
    fn finish(&self) -> u64 {
        unimplemented!()
    }
}

type CryptoSHA256Hash = [u8; 32];

impl Algorithm<CryptoSHA256Hash> for CryptoBitcoinAlgorithm {
    #[inline]
    fn hash(&mut self) -> CryptoSHA256Hash {
        let mut h = [0u8; 32];
        self.0.result(&mut h);

        // double sha256
        let mut c = Sha256::new();
        c.input(h.as_ref());
        c.result(&mut h);
        h
    }

    #[inline]
    fn reset(&mut self) {
        self.0.reset();
    }

    fn leaf(&mut self, leaf: CryptoSHA256Hash) -> CryptoSHA256Hash {
        leaf
    }

    fn node(
        &mut self,
        left: CryptoSHA256Hash,
        right: CryptoSHA256Hash,
        height: usize,
    ) -> CryptoSHA256Hash {
        height.hash(self);
        self.write(left.as_ref());
        self.write(right.as_ref());
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
fn test_crypto_bitcoin_leaf_hash() {
    let mut a = CryptoBitcoinAlgorithm::new();
    "hello".hash(&mut a);
    let h1 = a.hash();
    assert_eq!(
        format!("{}", HexSlice::new(h1.as_ref())),
        "503d8319a48348cdc610a582f7bf754b5833df65038606eb48510790dfc99595"
    );
}

/*
// Pow2 leafs only supported.
/// [](http://chimera.labs.oreilly.com/books/1234000001802/ch07.html#merkle_trees)
#[test]
fn test_crypto_bitcoin_node() {
    let mut h1 = [0u8; 32];
    let mut h2 = [0u8; 32];
    let mut h3 = [0u8; 32];
    h1[0] = 0x00;
    h2[0] = 0x11;
    h3[0] = 0x22;

    let mut a = CryptoBitcoinAlgorithm::new();
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

    let t: MerkleTree<CryptoSHA256Hash, CryptoBitcoinAlgorithm, VecStore<_>> =
        MerkleTree::try_from_iter(vec![h1, h2, h3].into_iter().map(Ok)).unwrap();
    assert_eq!(
        format!("{}", HexSlice::new(t.root().as_ref())),
        "5ba580c87c9bae263e6186318d77963846ff7a3e92b45f2ed30495ccf52b4731"
    );
}
*/
