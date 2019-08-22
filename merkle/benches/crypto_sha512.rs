//! cargo bench --features "crypto_bench" --verbose
#![cfg(feature = "crypto_bench")]
#![feature(test)]

mod hash512;

extern crate crypto;
extern crate merkletree;
extern crate rand;
extern crate rayon;
extern crate test;

use crypto::digest::Digest;
use crypto::sha2::Sha512;
use hash512::Hash512;
use merkletree::hash::{Algorithm, Hashable};
use merkletree::merkle::{DiskStore, FromIndexedParallelIterator, MerkleTree, VecStore};
use rand::Rng;
use rayon::prelude::*;
use std::hash::Hasher;
use std::iter::FromIterator;
use test::Bencher;

#[derive(Copy, Clone)]
struct A(Sha512);

impl A {
    fn new() -> A {
        A(Sha512::new())
    }
}

impl Default for A {
    fn default() -> Self {
        A::new()
    }
}

impl Hasher for A {
    #[inline]
    fn write(&mut self, msg: &[u8]) {
        self.0.input(msg)
    }

    #[inline]
    fn finish(&self) -> u64 {
        unimplemented!()
    }
}

impl Algorithm<Hash512> for A {
    #[inline]
    fn hash(&mut self) -> Hash512 {
        let mut h = [0u8; 64];
        self.0.result(&mut h);
        Hash512(h)
    }

    #[inline]
    fn reset(&mut self) {
        self.0.reset();
    }
}

fn tree_5() -> Vec<Hash512> {
    ["one", "two", "three", "four"]
        .iter()
        .map(|x| {
            let mut a = A::new();
            Hashable::hash(x, &mut a);
            a.hash()
        })
        .collect::<Vec<Hash512>>()
}

fn tree_160() -> Vec<Hash512> {
    let mut values = vec![vec![0u8; 256]; 160];
    let mut rng = rand::IsaacRng::new_unseeded();

    for mut v in &mut values {
        rng.fill_bytes(&mut v);
    }

    values
        .par_iter()
        .map(|x| {
            let mut a = A::new();
            a.write(x.as_ref());
            a.hash()
        })
        .collect::<Vec<Hash512>>()
}

fn tree_30000() -> Vec<Hash512> {
    let mut values = vec![vec![0u8; 256]; 30000];
    let mut rng = rand::IsaacRng::new_unseeded();

    for mut v in &mut values {
        rng.fill_bytes(&mut v);
    }

    values
        .par_iter()
        .map(|x| {
            let mut a = A::new();
            a.write(x.as_ref());
            a.hash()
        })
        .collect::<Vec<Hash512>>()
}

#[bench]
fn bench_crypto_sha512(b: &mut Bencher) {
    let mut h = [0u8; 64];
    b.iter(|| {
        let mut x = Sha512::new();
        x.input("12345".as_ref());
        x.result(&mut h);
    });
}

#[bench]
fn bench_crypto_sha512_from_data_5(b: &mut Bencher) {
    let values = tree_5();
    b.iter(|| MerkleTree::<Hash512, A, VecStore<_>>::from_iter(values.clone()));
}

#[bench]
fn bench_crypto_sha512_from_data_5_proof(b: &mut Bencher) {
    let values = tree_5();
    let tree: MerkleTree<Hash512, A, VecStore<_>> = MerkleTree::from_iter(values.clone());

    b.iter(|| {
        for i in 0..values.len() {
            let proof = tree.gen_proof(i);
            test::black_box(proof);
        }
    });
}

#[bench]
fn bench_crypto_sha512_from_data_5_proof_check(b: &mut Bencher) {
    let values = tree_5();
    let tree: MerkleTree<Hash512, A, VecStore<_>> = MerkleTree::from_iter(values.clone());
    let proofs = (0..values.len())
        .map(|i| tree.gen_proof(i))
        .collect::<Vec<_>>();

    b.iter(|| {
        for proof in &proofs {
            test::black_box(proof.validate::<A>());
        }
    });
}

#[bench]
fn bench_crypto_sha512_from_data_160_vec(b: &mut Bencher) {
    let values = tree_160();
    b.iter(|| MerkleTree::<Hash512, A, VecStore<_>>::from_iter(values.clone()));
}

#[bench]
fn bench_crypto_sha512_from_data_160_mmap(b: &mut Bencher) {
    let values = tree_160();
    b.iter(|| MerkleTree::<Hash512, A, DiskStore<_>>::from_iter(values.clone()));
}

#[bench]
fn bench_crypto_sha512_from_data_160_par(b: &mut Bencher) {
    let values = tree_160();
    b.iter(|| MerkleTree::<Hash512, A, VecStore<_>>::from_par_iter(values.clone()));
}

#[bench]
fn bench_crypto_sha512_from_data_30000_vec(b: &mut Bencher) {
    let values = tree_30000();
    b.iter(|| MerkleTree::<Hash512, A, VecStore<_>>::from_iter(values.clone()));
}

#[bench]
fn bench_crypto_sha512_from_data_30000_mmap(b: &mut Bencher) {
    let values = tree_30000();
    b.iter(|| MerkleTree::<Hash512, A, DiskStore<_>>::from_iter(values.clone()));
}

#[bench]
fn bench_crypto_sha512_from_data_30000_par(b: &mut Bencher) {
    let values = tree_30000();
    b.iter(|| MerkleTree::<Hash512, A, VecStore<_>>::from_par_iter(values.clone()));
}

#[bench]
fn bench_crypto_sha512_from_data_160_proof(b: &mut Bencher) {
    let values = tree_160();
    let tree: MerkleTree<Hash512, A, VecStore<_>> = MerkleTree::from_iter(values.clone());

    b.iter(|| {
        for i in 0..values.len() {
            let proof = tree.gen_proof(i);
            test::black_box(proof);
        }
    });
}

#[bench]
fn bench_crypto_sha512_from_data_160_proof_check(b: &mut Bencher) {
    let values = tree_160();
    let tree: MerkleTree<Hash512, A, VecStore<_>> = MerkleTree::from_iter(values.clone());
    let proofs = (0..values.len())
        .map(|i| tree.gen_proof(i))
        .collect::<Vec<_>>();

    b.iter(|| {
        for proof in &proofs {
            test::black_box(proof.validate::<A>());
        }
    });
}
