#![cfg(test)]

use crate::hash::{Algorithm, Hashable};
use crate::merkle::log2_pow2;
use crate::merkle::next_pow2;
use crate::merkle::MerkleTree;
use crate::store::VecStore;
use crate::test_item::Item;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;

impl Algorithm<Item> for DefaultHasher {
    #[inline]
    fn hash(&mut self) -> Item {
        Item(self.finish())
    }

    #[inline]
    fn reset(&mut self) {
        *self = DefaultHasher::default()
    }
}

#[test]
fn test_simple_tree() {
    let answer: Vec<Vec<u64>> = vec![
        vec![
            18161131233134742049,
            15963407371316104707,
            8061613778145084206,
        ],
        vec![
            18161131233134742049,
            15963407371316104707,
            2838807777806232157,
            2838807777806232157,
            8061613778145084206,
            8605533607343419251,
            12698627859487956302,
        ],
        vec![
            18161131233134742049,
            15963407371316104707,
            2838807777806232157,
            4356248227606450052,
            8061613778145084206,
            6971098229507888078,
            452397072384919190,
        ],
        vec![
            18161131233134742049,
            15963407371316104707,
            2838807777806232157,
            4356248227606450052,
            5528330654215492654,
            5528330654215492654,
            8061613778145084206,
            6971098229507888078,
            7858164776785041459,
            7858164776785041459,
            452397072384919190,
            13691461346724970593,
            12928874197991182098,
        ],
        vec![
            18161131233134742049,
            15963407371316104707,
            2838807777806232157,
            4356248227606450052,
            5528330654215492654,
            11057097817362835984,
            8061613778145084206,
            6971098229507888078,
            6554444691020019791,
            6554444691020019791,
            452397072384919190,
            2290028692816887453,
            151678167824896071,
        ],
        vec![
            18161131233134742049,
            15963407371316104707,
            2838807777806232157,
            4356248227606450052,
            5528330654215492654,
            11057097817362835984,
            15750323574099240302,
            15750323574099240302,
            8061613778145084206,
            6971098229507888078,
            6554444691020019791,
            13319587930734024288,
            452397072384919190,
            15756788945533226834,
            8300325667420840753,
        ],
    ];
    for items in [2, 4].iter() {
        let mut a = DefaultHasher::new();
        let mt: MerkleTree<Item, DefaultHasher, VecStore<_>> = MerkleTree::try_from_iter(
            [1, 2, 3, 4, 5, 6, 7, 8]
                .iter()
                .map(|x| {
                    a.reset();
                    x.hash(&mut a);
                    Ok(a.hash())
                })
                .take(*items),
        )
        .unwrap();

        assert_eq!(mt.leafs(), *items);
        assert_eq!(mt.height(), log2_pow2(next_pow2(mt.len())));
        assert_eq!(
            mt.read_range(0, mt.len()).unwrap(),
            answer[*items - 2].as_slice()
        );

        for i in 0..mt.leafs() {
            let p = mt.gen_proof(i).unwrap();
            assert!(p.validate::<DefaultHasher>().expect("failed to validate"));
        }
    }
}
