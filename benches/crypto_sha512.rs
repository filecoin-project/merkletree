//! cargo bench --features "crypto_bench" --verbose
#![cfg(feature = "crypto_bench")]
#![feature(test)]

extern crate test;

use std::hash::Hasher;

mod hash512;

use anyhow::Result;
use crypto::digest::Digest;
use crypto::sha2::Sha512;
use merkletree::hash::{Algorithm, Hashable};
use merkletree::merkle::{FromIndexedParallelIterator, MerkleTree};
use merkletree::store::{DiskStore, VecStore};
use rand::prelude::*;
use rand::thread_rng;
use rayon::prelude::*;
use test::Bencher;

use crate::hash512::Hash512;

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

fn tree_5() -> impl Iterator<Item = Result<Hash512>> {
    ["one", "two", "three", "four"].iter().map(|x| {
        let mut a = A::new();
        Hashable::hash(x, &mut a);
        Ok(a.hash())
    })
}

fn tree_160_par() -> impl IndexedParallelIterator<Item = Hash512> {
    let mut values = vec![[0u8; 256]; 160];

    let mut rng = thread_rng();
    for i in 0..values.len() {
        rng.fill(&mut values[i]);
    }

    values.into_par_iter().map(|x| {
        let mut a = A::new();
        a.write(x.as_ref());
        a.hash()
    })
}

fn tree_160() -> impl Iterator<Item = Result<Hash512>> {
    let mut values = vec![[0u8; 256]; 160];

    let mut rng = thread_rng();
    for i in 0..values.len() {
        rng.fill(&mut values[i]);
    }

    values.into_iter().map(|x| {
        let mut a = A::new();
        a.write(x.as_ref());
        Ok(a.hash())
    })
}

fn tree_30000() -> impl Iterator<Item = Result<Hash512>> {
    let mut values = vec![[0u8; 256]; 30000];

    let mut rng = thread_rng();
    for i in 0..values.len() {
        rng.fill(&mut values[i]);
    }

    values.into_iter().map(|x| {
        let mut a = A::new();
        a.write(x.as_ref());
        Ok(a.hash())
    })
}

fn tree_30000_par() -> impl IndexedParallelIterator<Item = Hash512> {
    let mut values = vec![[0u8; 256]; 30000];

    let mut rng = thread_rng();
    for i in 0..values.len() {
        rng.fill(&mut values[i]);
    }

    values.into_par_iter().map(|x| {
        let mut a = A::new();
        a.write(x.as_ref());
        a.hash()
    })
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
    b.iter(|| MerkleTree::<Hash512, A, VecStore<_>>::try_from_iter(tree_5()).unwrap());
}

#[bench]
fn bench_crypto_sha512_from_data_5_proof(b: &mut Bencher) {
    let tree: MerkleTree<Hash512, A, VecStore<_>> = MerkleTree::try_from_iter(tree_5()).unwrap();

    b.iter(|| {
        for i in 0..tree.len() {
            let proof = tree.gen_proof(i).unwrap();
            test::black_box(proof);
        }
    });
}

#[bench]
fn bench_crypto_sha512_from_data_5_proof_check(b: &mut Bencher) {
    let tree: MerkleTree<Hash512, A, VecStore<_>> = MerkleTree::try_from_iter(tree_5()).unwrap();

    let proofs = (0..tree.len())
        .map(|i| tree.gen_proof(i).unwrap())
        .collect::<Vec<_>>();

    b.iter(|| {
        for proof in &proofs {
            test::black_box(proof.validate::<A>().unwrap());
        }
    });
}

#[bench]
fn bench_crypto_sha512_from_data_160_vec(b: &mut Bencher) {
    b.iter(|| MerkleTree::<Hash512, A, VecStore<_>>::try_from_iter(tree_160()).unwrap());
}

#[bench]
fn bench_crypto_sha512_from_data_160_mmap(b: &mut Bencher) {
    b.iter(|| MerkleTree::<Hash512, A, DiskStore<_>>::try_from_iter(tree_160()).unwrap());
}

#[bench]
fn bench_crypto_sha512_from_data_160_par(b: &mut Bencher) {
    b.iter(|| MerkleTree::<Hash512, A, VecStore<_>>::from_par_iter(tree_160_par()));
}

#[bench]
fn bench_crypto_sha512_from_data_30000_vec(b: &mut Bencher) {
    b.iter(|| MerkleTree::<Hash512, A, VecStore<_>>::try_from_iter(tree_30000()).unwrap());
}

#[bench]
fn bench_crypto_sha512_from_data_30000_mmap(b: &mut Bencher) {
    b.iter(|| MerkleTree::<Hash512, A, DiskStore<_>>::try_from_iter(tree_30000()).unwrap());
}

#[bench]
fn bench_crypto_sha512_from_data_30000_par(b: &mut Bencher) {
    b.iter(|| MerkleTree::<Hash512, A, VecStore<_>>::from_par_iter(tree_30000_par()));
}

#[bench]
fn bench_crypto_sha512_from_data_160_proof(b: &mut Bencher) {
    let tree: MerkleTree<Hash512, A, VecStore<_>> = MerkleTree::try_from_iter(tree_160()).unwrap();

    b.iter(|| {
        for i in 0..tree.len() {
            let proof = tree.gen_proof(i).unwrap();
            test::black_box(proof);
        }
    });
}

#[bench]
fn bench_crypto_sha512_from_data_160_proof_check(b: &mut Bencher) {
    let tree: MerkleTree<Hash512, A, VecStore<_>> = MerkleTree::try_from_iter(tree_160()).unwrap();
    let proofs = (0..tree.len())
        .map(|i| tree.gen_proof(i).unwrap())
        .collect::<Vec<_>>();

    b.iter(|| {
        for proof in &proofs {
            test::black_box(proof.validate::<A>().unwrap());
        }
    });
}
