#![cfg(test)]
#![cfg(feature = "chaincore")]

use crypto::digest::Digest;
use crypto::sha3::{Sha3, Sha3Mode};
use merkletree::hash::Algorithm;
use merkletree::merkle::MerkleTree;
use merkletree::proof::Proof;
use merkletree::store::VecStore;
use std::fmt;
use std::hash::Hasher;
use typenum::U0;

#[derive(Clone)]
struct CryptoChainCoreAlgorithm(Sha3);

impl CryptoChainCoreAlgorithm {
    fn new() -> CryptoChainCoreAlgorithm {
        CryptoChainCoreAlgorithm(Sha3::new(Sha3Mode::Sha3_256))
    }
}

impl Default for CryptoChainCoreAlgorithm {
    fn default() -> CryptoChainCoreAlgorithm {
        CryptoChainCoreAlgorithm::new()
    }
}

impl Hasher for CryptoChainCoreAlgorithm {
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

impl Algorithm<CryptoSHA256Hash> for CryptoChainCoreAlgorithm {
    #[inline]
    fn hash(&mut self) -> CryptoSHA256Hash {
        let mut h = [0u8; 32];
        self.0.result(&mut h);
        h
    }

    #[inline]
    fn reset(&mut self) {
        self.0.reset();
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

/*
// Pow2 leafs is now required
#[test]
fn test_crypto_chaincore_node() {
    let mut h1 = [0u8; 32];
    let mut h2 = [0u8; 32];
    let mut h3 = [0u8; 32];
    h1[0] = 0x00;
    h2[0] = 0x11;
    h3[0] = 0x22;

    let t: MerkleTree<CryptoSHA256Hash, CryptoChainCoreAlgorithm, VecStore<_>> =
        MerkleTree::try_from_iter(vec![h1, h2, h3].into_iter().map(Ok)).unwrap();
    assert_eq!(
        format!("{}", HexSlice::new(t.root().as_ref())),
        "23704c527ffb21d1b1816938114c2fb0f6e50475d4ab5d07ebff855e7fd20335"
    );
}
 */

#[test]
fn test_merkle_tree_validate_data() {
    let data = vec![1, 2, 3, 4];
    let proof_item = data[0];

    let t: MerkleTree<CryptoSHA256Hash, CryptoChainCoreAlgorithm, VecStore<_>> =
        MerkleTree::from_data(data).unwrap();
    let generated_proof = t.gen_proof(0).unwrap();

    let proof: Proof<CryptoSHA256Hash> = Proof::new::<U0, U0>(
        None,
        generated_proof.lemma().to_owned(),
        generated_proof.path().to_owned(),
    )
    .unwrap();
    assert!(proof
        .validate_with_data::<CryptoChainCoreAlgorithm>(&proof_item)
        .unwrap());
}
