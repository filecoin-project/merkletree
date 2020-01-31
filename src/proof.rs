use crate::hash::{Algorithm, Hashable};
use std::marker::PhantomData;
use typenum::marker_traits::Unsigned;
use typenum::U2;

/// Merkle tree inclusion proof for data element, for which item = Leaf(Hash(Data Item)).
///
/// Lemma layout:
///
/// ```text
/// [ item h1x h2y h3z ... root ]
/// ```
///
/// Proof validation is positioned hash against lemma path to match root hash.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Proof<T: Eq + Clone + AsRef<[u8]>, U: Unsigned = U2> {
    lemma: Vec<Vec<T>>,
    path: Vec<usize>,   // branch index
    _u: PhantomData<U>, // number of branches per node
}

impl<T: Eq + Clone + AsRef<[u8]>, U: Unsigned> Proof<T, U> {
    /// Creates new MT inclusion proof
    pub fn new(hash: Vec<Vec<T>>, path: Vec<usize>) -> Proof<T, U> {
        assert!(hash.len() > 2);
        assert_eq!(hash.len() - 2, path.len());
        Proof {
            lemma: hash,
            path,
            _u: PhantomData,
        }
    }

    /// Return proof target leaf
    pub fn item(&self) -> Vec<T> {
        self.lemma.first().unwrap().clone()
    }

    /// Return tree root
    pub fn root(&self) -> T {
        self.lemma.last().unwrap()[0].clone()
    }

    /// Verifies MT inclusion proof
    pub fn validate<A: Algorithm<T>>(&self) -> bool {
        let size = self.lemma.len();
        if size < 2 {
            return false;
        }

        let mut a = A::default();
        let mut h = self.item()[0].to_owned();

        let branches = <U as Unsigned>::to_usize();
        for i in 1..size - 1 {
            a.reset();
            h = {
                let mut nodes: Vec<T> = Vec::with_capacity(branches);
                let mut cur_index = 0;
                for j in 0..branches {
                    if j == self.path[i - 1] {
                        nodes.push(h.clone());
                    } else {
                        nodes.push(self.lemma[i][cur_index].clone());
                        cur_index += 1;
                    }
                }
                assert_eq!(cur_index, branches - 1);
                a.multi_node(nodes, i - 1)
            };
        }

        h == self.root()
    }

    /// Verifies MT inclusion proof and that leaf_data is the original leaf data for which proof was generated.
    pub fn validate_with_data<A: Algorithm<T>>(&self, leaf_data: &dyn Hashable<A>) -> bool {
        let mut a = A::default();
        leaf_data.hash(&mut a);
        let item = a.hash();
        a.reset();
        let leaf_hash = a.leaf(item);

        (leaf_hash == self.item()[0]) && self.validate::<A>()
    }

    /// Returns the path of this proof.
    pub fn path(&self) -> &Vec<usize> {
        &self.path
    }

    /// Returns the lemma of this proof.
    pub fn lemma(&self) -> &Vec<Vec<T>> {
        &self.lemma
    }
}
