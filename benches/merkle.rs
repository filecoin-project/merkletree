use criterion::{black_box, criterion_group, criterion_main, Criterion};

use merkletree::merkle::{get_merkle_proof_lemma_len, get_merkle_tree_leafs, get_merkle_tree_len};

const DEFAULT_NUM_BRANCHES: usize = 2;

fn bench_get_merkle_tree_leafs(c: &mut Criterion) {
    for size in &[1, 256] {
        let sector_size = 1024 * 1024 * size;
        let tree_size = 2 * (sector_size / 32) - 1;
        c.bench_function(&format!("get merkle-tree leafs {size}mib"), |b| {
            b.iter(|| get_merkle_tree_leafs(black_box(tree_size), DEFAULT_NUM_BRANCHES)
                .expect("[bench_get_merkle_tree_leafs] couldn't compute number of leaves in Merkle Tree"));
        });
    }
}

fn bench_get_merkle_tree_info(c: &mut Criterion) {
    let branches = 8;
    let sector_size = 1073741824; // 2^30

    c.bench_function("get merkle-tree info 1gib", |b| {
        b.iter(|| {
            let tree_size = get_merkle_tree_len(black_box(sector_size), branches)
                .expect("[bench_get_merkle_tree_info] failed to get len");
            assert_eq!(
                get_merkle_tree_leafs(tree_size, branches).expect(
                    "[bench_get_merkle_tree_info] couldn't compute number of leaves in Merkle Tree"
                ),
                sector_size
            );
            get_merkle_proof_lemma_len(tree_size, branches)
        });
    });
}

criterion_group!(
    benches,
    bench_get_merkle_tree_leafs,
    bench_get_merkle_tree_info
);
criterion_main!(benches);
