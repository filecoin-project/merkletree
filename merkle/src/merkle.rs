use hash::{Algorithm, Hashable};
use memmap::MmapMut;
use memmap::MmapOptions;
use proof::Proof;
use rayon::prelude::*;
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::ops::{self, Index};

/// Merkle Tree.
///
/// All leafs and nodes are stored in a linear array (vec).
///
/// A merkle tree is a tree in which every non-leaf node is the hash of its
/// children nodes. A diagram depicting how it works:
///
/// ```text
///         root = h1234 = h(h12 + h34)
///        /                           \
///  h12 = h(h1 + h2)            h34 = h(h3 + h4)
///   /            \              /            \
/// h1 = h(tx1)  h2 = h(tx2)    h3 = h(tx3)  h4 = h(tx4)
/// ```
///
/// In memory layout:
///
/// ```text
///     [h1 h2 h3 h4 h12 h34 root]
/// ```
///
/// Merkle root is always the last element in the array.
///
/// The number of inputs is not always a power of two which results in a
/// balanced tree structure as above.  In that case, parent nodes with no
/// children are also zero and parent nodes with only a single left node
/// are calculated by concatenating the left node with itself before hashing.
/// Since this function uses nodes that are pointers to the hashes, empty nodes
/// will be nil.
///
/// TODO: Ord
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MerkleTree<T, A, K>
where
    T: Element,
    A: Algorithm<T>,
    K: Store<T>,
{
    data: K,
    leafs: usize,
    height: usize,
    _a: PhantomData<A>,
    _t: PhantomData<T>,
}

/// Element stored in the merkle tree.
pub trait Element: Ord + Clone + AsRef<[u8]> + Sync + Send /* Into<Vec<u8>> + From<Vec<u8>>*/ {
    /// Returns the length of an element when serialized as a byte slice.
    fn byte_len() -> usize;
}

/// Backing store of the merkle tree.
pub trait Store<E: Element>: ops::Deref<Target = [E]> {
    fn new(size: usize) -> Self;

    fn write_at(&mut self, el: E, i: usize);
    fn read_at(&self, i: usize) -> &E;
    fn read_range(&self, r: ops::Range<usize>) -> &[E];

    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn push(&mut self, el: E);
}

#[derive(Debug)]
pub struct VecStore<E: Element>(Vec<E>);

impl<E: Element> ops::Deref for VecStore<E> {
    type Target = [E];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<E: Element> Store<E> for VecStore<E> {
    fn new(size: usize) -> Self {
        VecStore(Vec::with_capacity(size))
    }

    fn write_at(&mut self, el: E, i: usize) {
        self.0[i] = el;
    }

    fn read_at(&self, i: usize) -> &E {
        &self.0[i]
    }

    fn read_range(&self, r: ops::Range<usize>) -> &[E] {
        self.0.index(r)
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn push(&mut self, el: E) {
        self.0.push(el);
    }
}

#[derive(Debug)]
pub struct MmapStore<E: Element> {
    store: MmapMut,
    len: usize,
    _e: PhantomData<E>,
}

impl<E: Element> ops::Deref for MmapStore<E> {
    type Target = [E];

    fn deref(&self) -> &Self::Target {
        unimplemented!()
    }
}

impl<E: Element> Store<E> for MmapStore<E> {
    #[allow(unsafe_code)]
    fn new(len: usize) -> Self {
        MmapStore {
            store: MmapOptions::new().len(len).map_anon().unwrap(),
            len: 0,
            _e: Default::default(),
        }
    }

    fn write_at(&mut self, el: E, i: usize) {
        unimplemented!()
    }

    fn read_at(&self, i: usize) -> &E {
        unimplemented!()
    }

    fn read_range(&self, r: ops::Range<usize>) -> &[E] {
        unimplemented!()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn push(&mut self, el: E) {
        let l = self.len;
        assert!(l + 1 <= self.store.len(), "not enough space");

        self.write_at(el, l);
    }
}

impl<T: Element, A: Algorithm<T>, K: Store<T>> MerkleTree<T, A, K> {
    /// Creates new merkle from a sequence of hashes.
    pub fn new<I: IntoIterator<Item = T>>(data: I) -> MerkleTree<T, A, K> {
        Self::from_iter(data)
    }

    /// Creates new merkle tree from a list of hashable objects.
    pub fn from_data<O: Hashable<A>, I: IntoIterator<Item = O>>(data: I) -> MerkleTree<T, A, K> {
        let mut a = A::default();
        Self::from_iter(data.into_iter().map(|x| {
            a.reset();
            x.hash(&mut a);
            a.hash()
        }))
    }

    #[inline]
    fn build(&mut self) {
        // This algorithms assumes that the underlying store has preallocated enough space.
        // TODO: add an assert here to ensure this is the case.

        let mut width = self.leafs;

        // build tree
        let mut i: usize = 0;
        let mut j: usize = width;
        let mut height: usize = 0;

        while width > 1 {
            // if there is odd num of elements, fill in to the even
            if width & 1 == 1 {
                let el = self.data.read_at(self.data.len() - 1).clone();
                self.data.push(el);

                width += 1;
                j += 1;
            }

            // elements are in [i..j] and they are even
            let layer: Vec<_> = self
                .data
                .read_range(i..j)
                .par_chunks(2)
                .map(|v| A::default().node(v[0].clone(), v[1].clone(), height))
                .collect();

            // TODO: avoid collecting into a vec and write the results direclty if possible.
            for el in layer.into_iter() {
                self.data.push(el);
            }

            width >>= 1;
            i = j;
            j = j + width;
            height += 1;
        }
    }

    /// Generate merkle tree inclusion proof for leaf `i`
    #[inline]
    pub fn gen_proof(&self, i: usize) -> Proof<T> {
        assert!(i < self.leafs); // i in [0 .. self.leafs)

        let mut lemma: Vec<T> = Vec::with_capacity(self.height + 1); // path + root
        let mut path: Vec<bool> = Vec::with_capacity(self.height - 1); // path - 1

        let mut base = 0;
        let mut j = i;

        // level 1 width
        let mut width = self.leafs;
        if width & 1 == 1 {
            width += 1;
        }

        lemma.push(self.data.read_at(j).clone());
        while base + 1 < self.len() {
            lemma.push(if j & 1 == 0 {
                // j is left
                self.data.read_at(base + j + 1).clone()
            } else {
                // j is right
                self.data.read_at(base + j - 1).clone()
            });
            path.push(j & 1 == 0);

            base += width;
            width >>= 1;
            if width & 1 == 1 {
                width += 1;
            }
            j >>= 1;
        }

        // root is final
        lemma.push(self.root());
        Proof::new(lemma, path)
    }

    /// Returns merkle root
    #[inline]
    pub fn root(&self) -> T {
        self.data.read_at(self.data.len() - 1).clone()
    }

    /// Returns number of elements in the tree.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the vector contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns height of the tree
    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns original number of elements the tree was built upon.
    #[inline]
    pub fn leafs(&self) -> usize {
        self.leafs
    }

    /// Extracts a slice containing the entire vector.
    ///
    /// Equivalent to `&s[..]`.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self
    }
}

impl<T: Element, A: Algorithm<T>, K: Store<T>> FromParallelIterator<T> for MerkleTree<T, A, K> {
    /// Creates new merkle tree from an iterator over hashable objects.
    fn from_par_iter<I: IntoParallelIterator<Item = T>>(into: I) -> Self {
        let iter = into.into_par_iter();

        let leafs = iter.opt_len().expect("must be sized");
        let pow = next_pow2(leafs);
        let size = 2 * pow - 1;

        let mut data = K::new(size);

        // leafs
        let vs = iter
            .map(|item| {
                let mut a = A::default();
                a.leaf(item)
            })
            .collect::<Vec<_>>();

        for v in vs.into_iter() {
            data.push(v);
        }

        assert!(leafs > 1);
        let mut mt: MerkleTree<T, A, K> = MerkleTree {
            data,
            leafs,
            height: log2_pow2(size + 1),
            _a: PhantomData,
            _t: PhantomData,
        };

        mt.build();

        mt
    }
}

impl<T: Element, A: Algorithm<T>, K: Store<T>> FromIterator<T> for MerkleTree<T, A, K> {
    /// Creates new merkle tree from an iterator over hashable objects.
    fn from_iter<I: IntoIterator<Item = T>>(into: I) -> Self {
        let iter = into.into_iter();

        let leafs = iter.size_hint().1.unwrap();
        assert!(leafs > 1);

        let pow = next_pow2(leafs);
        let size = 2 * pow - 1;

        let mut data = K::new(size);

        // leafs
        let mut a = A::default();
        for item in iter {
            a.reset();
            data.push(a.leaf(item));
        }

        let mut mt: MerkleTree<T, A, K> = MerkleTree {
            data,
            leafs,
            height: log2_pow2(size + 1),
            _a: PhantomData,
            _t: PhantomData,
        };

        mt.build();
        mt
    }
}

impl<T: Element, A: Algorithm<T>, K: Store<T>> ops::Deref for MerkleTree<T, A, K> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.data.deref()
    }
}

/// `next_pow2` returns next highest power of two from a given number if
/// it is not already a power of two.
///
/// [](http://locklessinc.com/articles/next_pow2/)
/// [](https://stackoverflow.com/questions/466204/rounding-up-to-next-power-of-2/466242#466242)
pub fn next_pow2(mut n: usize) -> usize {
    n -= 1;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    n + 1
}

/// find power of 2 of a number which is power of 2
pub fn log2_pow2(n: usize) -> usize {
    n.trailing_zeros() as usize
}
