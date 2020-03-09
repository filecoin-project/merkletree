//! cargo bench --features "crypto_bench" --verbose
#![cfg(feature = "crypto_bench")]
#![feature(test)]

extern crate test;

use std::hash::Hasher;

mod hash512;
mod ringx;

use anyhow::Result;
use merkletree::hash::{Algorithm, Hashable};
use merkletree::merkle::MerkleTree;
use merkletree::store::VecStore;
use rand::prelude::*;
use rand::thread_rng;
use ringx::{Context, SHA512};
use test::Bencher;

use crate::hash512::Hash512;

#[derive(Clone)]
struct B(Context);

impl B {
    fn new() -> B {
        B(Context::new(&SHA512))
    }
}

impl Default for B {
    fn default() -> Self {
        B::new()
    }
}

impl Hasher for B {
    #[inline]
    fn write(&mut self, msg: &[u8]) {
        self.0.update(msg)
    }

    #[inline]
    fn finish(&self) -> u64 {
        unimplemented!()
    }
}

impl Algorithm<Hash512> for B {
    #[inline]
    fn hash(&mut self) -> Hash512 {
        let mut h = [0u8; 64];
        h.copy_from_slice(self.0.finish().as_ref());
        Hash512(h)
    }

    #[inline]
    fn reset(&mut self) {
        self.0.reset();
    }
}

fn tree_5() -> impl Iterator<Item = Result<Hash512>> {
    ["one", "two", "three", "four"].iter().map(|x| {
        let mut a = B::new();
        Hashable::hash(x, &mut a);
        Ok(a.hash())
    })
}

fn tree_160() -> impl Iterator<Item = Result<Hash512>> {
    let mut values = vec![[0u8; 256]; 160];

    let mut rng = thread_rng();
    for i in 0..values.len() {
        rng.fill(&mut values[i]);
    }

    values.into_iter().map(|x| {
        let mut a = B::new();
        a.write(x.as_ref());
        Ok(a.hash())
    })
}

#[bench]
fn bench_ringx_sha512(b: &mut Bencher) {
    b.iter(|| {
        let mut x = Context::new(&SHA512);
        x.update("12345".as_ref());
        x.finish();
    });
}

#[bench]
fn bench_ringx_sha512_from_data_5(b: &mut Bencher) {
    b.iter(|| MerkleTree::<Hash512, B, VecStore<_>>::try_from_iter(tree_5()).unwrap());
}

#[bench]
fn bench_ringx_sha512_from_data_5_proof(b: &mut Bencher) {
    let tree: MerkleTree<Hash512, B, VecStore<_>> = MerkleTree::try_from_iter(tree_5()).unwrap();

    b.iter(|| {
        for i in 0..tree.len() {
            let proof = tree.gen_proof(i).unwrap();
            test::black_box(proof);
        }
    });
}

#[bench]
fn bench_ringx_sha512_from_data_5_proof_check(b: &mut Bencher) {
    let tree: MerkleTree<Hash512, B, VecStore<_>> = MerkleTree::try_from_iter(tree_5()).unwrap();
    let proofs = (0..tree.len())
        .map(|i| tree.gen_proof(i).unwrap())
        .collect::<Vec<_>>();

    b.iter(|| {
        for proof in &proofs {
            test::black_box(proof.validate::<B>().unwrap());
        }
    });
}

#[bench]
fn bench_ringx_sha512_from_data_160(b: &mut Bencher) {
    b.iter(|| MerkleTree::<Hash512, B, VecStore<_>>::try_from_iter(tree_160()).unwrap());
}

#[bench]
fn bench_ringx_sha512_from_data_160_proof(b: &mut Bencher) {
    let tree: MerkleTree<Hash512, B, VecStore<_>> = MerkleTree::try_from_iter(tree_160()).unwrap();

    b.iter(|| {
        for i in 0..tree.len() {
            let proof = tree.gen_proof(i).unwrap();
            test::black_box(proof);
        }
    });
}

#[bench]
fn bench_ringx_sha512_from_data_160_proof_check(b: &mut Bencher) {
    let values = tree_160();
    let tree: MerkleTree<Hash512, B, VecStore<_>> = MerkleTree::try_from_iter(values).unwrap();
    let proofs = (0..tree.len())
        .map(|i| tree.gen_proof(i).unwrap())
        .collect::<Vec<_>>();

    b.iter(|| {
        for proof in &proofs {
            test::black_box(proof.validate::<B>().unwrap());
        }
    });
}
