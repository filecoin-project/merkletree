use hash::{Algorithm, Hashable};
use memmap::MmapMut;
use memmap::MmapOptions;
use proof::Proof;
use rayon::prelude::*;
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::ops;

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
    K: Store,
{
    data: K,
    leafs: usize,
    height: usize,
    _a: PhantomData<A>,
    _t: PhantomData<T>,
}

pub trait Element: Ord + Clone + AsRef<[u8]> + Sync + Send + Into<Vec<u8>> + From<Vec<u8>> {
    /// Returns the length of an element when serialized as a byte slice.
    fn byte_len() -> usize;
}

pub trait Store: AsRef<[u8]> + AsMut<[u8]> + ops::Deref<Target = [u8]> {}

impl<T: Element, A: Algorithm<T>, K: Store> MerkleTree<T, A, K> {
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
        let elem_width = T::byte_len();

        // build tree
        let mut i: usize = 0;
        let mut j: usize = width;
        let mut height: usize = 0;

        while width > 1 {
            // if there is odd num of elements, fill in to the even
            if width & 1 == 1 {
                let data = self.data.as_mut();
                let start = width * elem_width - 2 * elem_width;
                let mid = width * elem_width;
                let end = width * elem_width - elem_width;

                data[mid..end].copy_from_slice(&data[start..mid]);

                width += 1;
                j += 1;
            }

            // elements are in [i..j] and they are even
            // TODO: parallelsim
            let layer: Vec<u8> = self
                .data
                .get(i * elem_width..j * elem_width)
                .expect("out of bounds")
                .chunks(2 * elem_width)
                .flat_map(|v| {
                    let (lhs, rhs) = v.split_at(elem_width);
                    let h = A::default().node(lhs.to_vec().into(), rhs.to_vec().into(), height);
                    h.into()
                })
                .collect();

            width >>= 1;
            let next_i = j;
            let next_j = j + width;
            height += 1;

            {
                let data = self.data.as_mut();
                // TODO: avoid collecting into a vec and write the results direclty if possible.
                data[j * elem_width..next_j * elem_width].copy_from_slice(&layer[..]);
            }

            i = next_i;
            j = next_j;
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

        lemma.push(self.data[j].clone());
        while base + 1 < self.len() {
            lemma.push(if j & 1 == 0 {
                // j is left
                self.data[base + j + 1].clone()
            } else {
                // j is right
                self.data[base + j - 1].clone()
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
        self.data[self.data.len() - 1].clone()
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

#[allow(unsafe_code)]
fn anonymous_mmap(len: usize) -> MmapMut {
    unsafe { MmapOptions::new().len(len).map_anon().unwrap() }
}

impl<T: Element, A: Algorithm<T>, K: Store> FromParallelIterator<T> for MerkleTree<T, A, K> {
    /// Creates new merkle tree from an iterator over hashable objects.
    fn from_par_iter<I: IntoParallelIterator<Item = T>>(into: I) -> Self {
        let iter = into.into_par_iter();
        let mut data: Vec<T> = match iter.opt_len() {
            Some(e) => {
                let pow = next_pow2(e);
                let size = 2 * pow - 1;
                (*anonymous_mmap(size)).to_vec()
            }
            None => Vec::new(),
        };

        // leafs
        data.par_extend(iter.map(|item| {
            let mut a = A::default();
            a.leaf(item)
        }));

        let leafs = data.len();
        let pow = next_pow2(leafs);
        let size = 2 * pow - 1;

        assert!(leafs > 1);
        let mut mt: MerkleTree<T, A> = MerkleTree {
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

impl<T: Element, A: Algorithm<T>, K: Store> FromIterator<T> for MerkleTree<T, A, K> {
    /// Creates new merkle tree from an iterator over hashable objects.
    fn from_iter<I: IntoIterator<Item = T>>(into: I) -> Self {
        let iter = into.into_iter();
        let mut data: Vec<T> = match iter.size_hint().1 {
            Some(e) => {
                let pow = next_pow2(e);
                let size = 2 * pow - 1;
                Vec::with_capacity(size)
            }
            None => Vec::new(),
        };

        // leafs
        let mut a = A::default();
        for item in iter {
            a.reset();
            data.push(a.leaf(item));
        }

        let leafs = data.len();
        let pow = next_pow2(leafs);
        let size = 2 * pow - 1;

        assert!(leafs > 1);

        let mut mt: MerkleTree<T, A> = MerkleTree {
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

impl<T: Element, A: Algorithm<T>, K: Store> ops::Deref for MerkleTree<T, A, K> {
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
