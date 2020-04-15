#![feature(test)]
extern crate test;

use test::{black_box, Bencher};

use merkletree::merkle::{get_merkle_proof_lemma_len, get_merkle_tree_leafs, get_merkle_tree_len};

const DEFAULT_NUM_BRANCHES: usize = 2;

#[bench]
fn bench_get_merkle_tree_leafs_1mib(b: &mut Bencher) {
    let sector_size = 1024 * 1024;
    let tree_size = 2 * (sector_size / 32) - 1;
    b.iter(|| black_box(get_merkle_tree_leafs(tree_size, DEFAULT_NUM_BRANCHES).unwrap()))
}

#[bench]
fn bench_get_merkle_tree_leafs_256mib(b: &mut Bencher) {
    let sector_size = 1024 * 1024 * 256;
    let tree_size = 2 * (sector_size / 32) - 1;
    b.iter(|| black_box(get_merkle_tree_leafs(tree_size, DEFAULT_NUM_BRANCHES).unwrap()))
}

#[bench]
fn bench_get_merkle_tree_info_1gib(b: &mut Bencher) {
    let branches = 8;
    let sector_size = 1073741824; // 2^30

    b.iter(|| {
        black_box({
            let tree_size = get_merkle_tree_len(sector_size, branches).expect("failed to get len");
            assert_eq!(
                get_merkle_tree_leafs(tree_size, branches).unwrap(),
                sector_size
            );
            get_merkle_proof_lemma_len(tree_size, branches)
        })
    })
}
