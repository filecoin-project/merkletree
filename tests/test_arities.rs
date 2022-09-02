#![cfg(not(tarpaulin_include))]
pub mod common;

use typenum::{Unsigned, U0, U2, U3, U4, U5, U8};

use merkletree::merkle::{get_merkle_tree_len_generic, Element, MerkleTree};
use merkletree::store::VecStore;

use crate::common::{
    get_vector_of_base_trees, instantiate_new, test_disk_mmap_vec_tree_functionality, TestItemType,
    TestSha256Hasher,
};

#[test]
fn test_base_tree_arities() {
    fn run_test<BaseTreeArity: Unsigned>(leaves: usize, root: TestItemType) {
        let expected_leaves = leaves;
        let len = get_merkle_tree_len_generic::<BaseTreeArity, U0, U0>(leaves)
            .expect("[test_base_tree_arities] couldn't compute Merkle Tree len");

        let tree = instantiate_new::<
            TestItemType,
            TestSha256Hasher,
            VecStore<TestItemType>,
            BaseTreeArity,
        >(leaves, None);
        test_disk_mmap_vec_tree_functionality::<
            TestItemType,
            TestSha256Hasher,
            VecStore<TestItemType>,
            BaseTreeArity,
            U0,
            U0,
        >(tree, expected_leaves, len, root);
    }

    let root = TestItemType::from_slice(&[
        142, 226, 200, 91, 184, 251, 142, 223, 219, 43, 122, 241, 23, 37, 97, 46,
    ]);
    run_test::<U2>(4, root);

    let root = TestItemType::from_slice(&[
        128, 59, 187, 58, 199, 144, 7, 238, 128, 146, 124, 33, 241, 16, 92, 221,
    ]);
    run_test::<U4>(16, root);

    let root = TestItemType::from_slice(&[
        252, 61, 163, 229, 140, 223, 198, 165, 200, 137, 59, 43, 83, 136, 197, 63,
    ]);
    run_test::<U8>(64, root);
}

#[test]
fn test_compound_tree_arities() {
    fn run_test<BaseTreeArity: Unsigned, SubTreeArity: Unsigned>(
        leaves: usize,
        root: TestItemType,
    ) {
        let expected_leaves = leaves * SubTreeArity::to_usize();
        let len = get_merkle_tree_len_generic::<BaseTreeArity, SubTreeArity, U0>(leaves)
            .expect("[test_compound_tree_arities] couldn't compute Merkle Tree len");

        let tree = MerkleTree::<
            TestItemType,
            TestSha256Hasher,
            VecStore<TestItemType>,
            BaseTreeArity,
            SubTreeArity,
        >::from_trees(get_vector_of_base_trees::<
            TestItemType,
            TestSha256Hasher,
            VecStore<TestItemType>,
            BaseTreeArity,
            SubTreeArity,
        >(leaves))
        .expect("can't instantiate compound tree [test_compound_tree_arities]");

        test_disk_mmap_vec_tree_functionality::<
            TestItemType,
            TestSha256Hasher,
            VecStore<TestItemType>,
            BaseTreeArity,
            SubTreeArity,
            U0,
        >(tree, expected_leaves, len, root);
    }

    let root = TestItemType::from_slice(&[
        57, 201, 227, 235, 242, 179, 108, 46, 157, 200, 126, 217, 134, 232, 141, 223,
    ]);
    run_test::<U2, U2>(4, root);

    let root = TestItemType::from_slice(&[
        146, 59, 189, 83, 119, 102, 147, 207, 178, 121, 11, 190, 241, 152, 67, 0,
    ]);
    run_test::<U4, U4>(16, root);

    let root = TestItemType::from_slice(&[
        32, 129, 168, 134, 58, 233, 155, 225, 88, 230, 247, 63, 18, 38, 194, 230,
    ]);
    run_test::<U8, U8>(64, root);

    let root = TestItemType::from_slice(&[
        81, 96, 135, 96, 165, 113, 149, 203, 222, 86, 102, 127, 139, 194, 78, 22,
    ]);
    run_test::<U2, U4>(4, root);

    let root = TestItemType::from_slice(&[
        149, 57, 53, 8, 68, 184, 94, 209, 244, 218, 43, 172, 185, 215, 193, 99,
    ]);
    run_test::<U8, U4>(64, root);

    let root = TestItemType::from_slice(&[
        127, 19, 226, 22, 109, 131, 88, 30, 221, 228, 251, 183, 147, 248, 2, 186,
    ]);
    run_test::<U2, U5>(4, root);

    let root = TestItemType::from_slice(&[
        67, 94, 188, 238, 85, 194, 96, 252, 163, 54, 119, 99, 218, 210, 231, 190,
    ]);
    run_test::<U4, U3>(16, root);
}

#[test]
fn test_compound_compound_tree_arities() {
    fn run_test<BaseTreeArity: Unsigned, SubTreeArity: Unsigned, TopTreeArity: Unsigned>(
        leaves: usize,
        root: TestItemType,
    ) {
        let expected_leaves = leaves * SubTreeArity::to_usize() * TopTreeArity::to_usize();
        let len = get_merkle_tree_len_generic::<BaseTreeArity, SubTreeArity, TopTreeArity>(leaves)
            .expect("[test_compound_compound_tree_arities] couldn't compute Merkle Tree len");

        let base_trees = (0..TopTreeArity::to_usize())
            .flat_map(|_| {
                get_vector_of_base_trees::<
                    TestItemType,
                    TestSha256Hasher,
                    VecStore<TestItemType>,
                    BaseTreeArity,
                    SubTreeArity,
                >(leaves)
            })
            .collect::<Vec<MerkleTree<_, _, _, BaseTreeArity>>>();

        let tree = MerkleTree::from_sub_trees_as_trees(base_trees).expect(
            "can't instantiate compound-compound tree [test_compound_compound_tree_arities]",
        );

        test_disk_mmap_vec_tree_functionality::<
            TestItemType,
            TestSha256Hasher,
            VecStore<TestItemType>,
            BaseTreeArity,
            SubTreeArity,
            TopTreeArity,
        >(tree, expected_leaves, len, root);
    }

    let root = TestItemType::from_slice(&[
        77, 96, 160, 26, 181, 161, 25, 63, 24, 181, 60, 43, 45, 20, 246, 181,
    ]);
    run_test::<U2, U2, U2>(4, root);

    let root = TestItemType::from_slice(&[
        52, 152, 123, 224, 174, 42, 152, 12, 199, 4, 105, 245, 176, 59, 230, 86,
    ]);
    run_test::<U8, U8, U2>(64, root);

    // TODO investigate whether limitations of 'from_sub_trees_as_trees' constructors are reasonable
    // run_test::<U2, U4, U8>(4, root);
    // run_test::<U2, U2, U4>(4, root);
    // run_test::<U8, U2, U8>(64, root);
    // run_test::<U8, U8, U8>(64, root);
    // run_test::<U8, U8, U3>(64, root);
    // run_test::<U2, U5, U3>(4, root);
    // etc...
}
