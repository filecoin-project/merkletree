use hash::{Algorithm, Hashable};

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
pub struct Proof<T: Eq + Clone + AsRef<[u8]>> {
    lemma: Vec<T>,
    path: Vec<bool>,
}

impl<T: Eq + Clone + AsRef<[u8]>> Proof<T> {
    /// Creates new MT inclusion proof
    pub fn new(hash: Vec<T>, path: Vec<bool>) -> Proof<T> {
        assert!(hash.len() > 2);
        assert_eq!(hash.len() - 2, path.len());
        Proof { lemma: hash, path }
    }

    /// Return proof target leaf
    pub fn item(&self) -> T {
        self.lemma.first().unwrap().clone()
    }

    /// Return tree root
    pub fn root(&self) -> T {
        self.lemma.last().unwrap().clone()
    }

    /// Verifies MT inclusion proof
    pub fn validate<A: Algorithm<T>>(&self) -> bool {
        let size = self.lemma.len();
        if size < 2 {
            return false;
        }

        let mut h = self.item();
        let mut a = A::default();

        for i in 1..size - 1 {
            a.reset();
            h = if self.path[i - 1] {
                a.node(h, self.lemma[i].clone(), i - 1)
            } else {
                a.node(self.lemma[i].clone(), h, i - 1)
            };
        }

        h == self.root()
    }

    /// Verifies MT inclusion proof and that leaf_data is the original leaf data for which proof was generated.
    pub fn validate_with_data<A: Algorithm<T>>(&self, leaf_data: &Hashable<A>) -> bool {
        let mut a = A::default();
        leaf_data.hash(&mut a);
        let item = a.hash();
        a.reset();
        let leaf_hash = a.leaf(item);

        (leaf_hash == self.item()) && self.validate::<A>()
    }

    /// Returns the path of this proof.
    pub fn path(&self) -> &Vec<bool> {
        &self.path
    }

    /// Returns the lemma of this proof.
    pub fn lemma(&self) -> &Vec<T> {
        &self.lemma
    }
}
