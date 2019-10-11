use hash::{Algorithm, Hashable};
use proof::Proof;
use rayon::prelude::*;
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};
use store::Store;

/// Tree size (number of nodes) used as threshold to decide which build algorithm
/// to use. Small trees (below this value) use the old build algorithm, optimized
/// for speed rather than memory, allocating as much as needed to allow multiple
/// threads to work concurrently without interrupting each other. Large trees (above)
/// use the new build algorithm, optimized for memory rather than speed, allocating
/// as less as possible with multiple threads competing to get the write lock.
pub const SMALL_TREE_BUILD: usize = 1024;

// Number of nodes to process in parallel during the `build` stage.
pub const BUILD_CHUNK_NODES: usize = 1024;

// Number of batched nodes processed and stored together when
// populating from the data leaves.
pub const BUILD_DATA_BLOCK_SIZE: usize = 64 * BUILD_CHUNK_NODES;

// FIXME: Hand-picked constants, some proper benchmarks should be done
// to choose more appropriate values and document the decision.

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
    K: Store<T>
{
    data: K,
    leafs: usize,
    height: usize,

    // Cache with the `root` of the tree built from `data`. This allows to
    // not access the `Store` (e.g., access to disks in `DiskStore`).
    root: T,

    _a: PhantomData<A>,
    _t: PhantomData<T>,
}

/// Element stored in the merkle tree.
pub trait Element: Ord + Clone + AsRef<[u8]> + Sync + Send + Default + std::fmt::Debug {
    /// Returns the length of an element when serialized as a byte slice.
    fn byte_len() -> usize;

    /// Creates the element from its byte form. Panics if the slice is not appropriately sized.
    fn from_slice(bytes: &[u8]) -> Self;

    fn copy_to_slice(&self, bytes: &mut [u8]);
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

    /// Creates new merkle from an already allocated `Store` (used with
    /// `DiskStore::new_with_path` to set its path before instantiating
    /// the MT, which would otherwise just call `DiskStore::new`).
    // FIXME: Taken from `MerkleTree::from_iter` to avoid adding more complexity,
    //  it should receive a `parallel` flag to decide what to do.
    // FIXME: We're repeating too much code here, `from_iter` (and
    //  `from_par_iter`) should be extended to handled a pre-allocated `Store`.
    // FIXME: Remove the `leafs` parameter, that could be obtained from the
    //  store adding a `capacity` method to the trait.
    pub fn from_data_store(data: K, leafs: usize) -> MerkleTree<T, A, K> {
        let pow = next_pow2(leafs);
        Self::build(data, leafs, log2_pow2(2 * pow))
    }

    #[inline]
    fn build(data: K, leafs: usize, height: usize) -> Self {
        if leafs <= SMALL_TREE_BUILD {
            return Self::build_small_tree(data, leafs, height);
        }

        assert!(data.len() == leafs);
        let data_lock = Arc::new(RwLock::new(data));

        // Process one `level` at a time of `width` nodes. Each level has half the nodes
        // as the previous one; the first level, completely stored in `data`, has `leafs`
        // nodes. We guarantee an even number of nodes per `level`, duplicating the last
        // node if necessary.
        let mut level: usize = 0;
        let mut width = leafs;
        let mut level_node_index = 0;
        while width > 1 {
            if width & 1 == 1 {
                // Odd number of nodes, duplicate last.
                let mut active_store = data_lock.write().unwrap();
                let last_node = active_store.read_at(active_store.len() - 1);
                active_store.push(last_node);

                width += 1;
            }

            // Start reading at the beginning of the current level, and writing the next
            // level immediate after.  `level_node_index` keeps track of the current read
            // starts, and width is updated accordingly at each level so that we know where
            // to start writing.
            let (read_start, write_start) = if level == 0 {
                // Note that we previously asserted that data.len() == leafs.
                (0, data_lock.read().unwrap().len())
            } else {
                (level_node_index, level_node_index + width)
            };

            // Allocate `width` indexes during operation (which is a negligible memory bloat
            // compared to the 32-bytes size of the nodes stored in the `Store`s) and hash each
            // pair of nodes to write them to the next level in concurrent threads.
            // Process `BUILD_CHUNK_NODES` nodes in each thread at a time to reduce contention,
            // optimized for big sector sizes (small ones will just have one thread doing all
            // the work).
            debug_assert_eq!(BUILD_CHUNK_NODES % 2, 0);
            Vec::from_iter((read_start..read_start + width).step_by(BUILD_CHUNK_NODES))
                .par_iter()
                .for_each(|&chunk_index| {
                    let chunk_size = std::cmp::min(
                        BUILD_CHUNK_NODES, read_start + width - chunk_index);

                    let chunk_nodes = {
                        // Read everything taking the lock once.
                        data_lock.read().unwrap()
                            .read_range(chunk_index..chunk_index + chunk_size)
                    };

                    // We write the hashed nodes to the next level in the position that
                    // would be "in the middle" of the previous pair (dividing by 2).
                    let write_delta = (chunk_index - read_start) / 2;

                    let nodes_size = (chunk_nodes.len() / 2) * T::byte_len();
                    let hashed_nodes_as_bytes = chunk_nodes.chunks(2).fold(
                        Vec::with_capacity(nodes_size),
                        |mut acc, node_pair| {
                            let h = A::default().node(
                                node_pair[0].clone(),
                                node_pair[1].clone(),
                                level,
                            );
                            acc.extend_from_slice(h.as_ref());
                            acc
                        },
                    );

                    debug_assert_eq!(hashed_nodes_as_bytes.len(), chunk_size / 2 * T::byte_len());
                    // Check that we correctly pre-allocated the space.

                    data_lock
                        .write()
                        .unwrap()
                        .copy_from_slice(&hashed_nodes_as_bytes, write_start + write_delta);
                });

            level_node_index += width;
            level += 1;
            width >>= 1;

            data_lock.write().unwrap().sync();
        }

        assert_eq!(height, level + 1);
        // The root isn't part of the previous loop so `height` is
        // missing one level.

        let root = {
            let data = data_lock.read().unwrap();
            data.read_at(data.len() - 1)
        };

        MerkleTree {
            data: Arc::try_unwrap(data_lock).unwrap().into_inner().unwrap(),
            leafs,
            height,
            root,
            _a: PhantomData,
            _t: PhantomData,
        }
    }

    #[inline]
    fn build_small_tree(mut data: K, leafs: usize, height: usize) -> Self {
        assert!(data.len() == leafs);
        let mut level: usize = 0;
        let mut width = leafs;
        let mut level_node_index = 0;

        while width > 1 {
            if width & 1 == 1 {
                let last_node = data.read_at(data.len() - 1);
                data.push(last_node);

                width += 1;
            }

            // Same indexing logic as `build`.
            let (layer, write_start) = {
                let (read_start, write_start) = if level == 0 {
                    // Note that we previously asserted that data.len() == leafs.
                    (0, data.len())
                } else {
                    (level_node_index, level_node_index + width)
                };

                let layer: Vec<_> = data
                    .read_range(read_start..read_start + width)
                    .par_chunks(2)
                    .map(|v| {
                        let lhs = v[0].to_owned();
                        let rhs = v[1].to_owned();
                        A::default().node(lhs, rhs, level)
                    })
                    .collect();

                (layer, write_start)
            };

            for (i, node) in layer.into_iter().enumerate() {
                data.write_at(node, write_start + i);
            }

            level_node_index += width;
            level += 1;
            width >>= 1;
        }

        assert_eq!(height, level + 1);
        // The root isn't part of the previous loop so `height` is
        // missing one level.

        let root = { data.read_at(data.len() - 1) };

        MerkleTree {
            data,
            leafs,
            height,
            root,
            _a: PhantomData,
            _t: PhantomData,
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

        lemma.push(self.read_at(j));
        while base + 1 < self.len() {
            lemma.push(if j & 1 == 0 {
                // j is left
                self.read_at(base + j + 1)
            } else {
                // j is right
                self.read_at(base + j - 1)
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

        // Sanity check: if the `MerkleTree` lost its integrity and `data` doesn't match the
        // expected values for `leafs` and `height` this can get ugly.
        debug_assert!(lemma.len() == self.height + 1);
        debug_assert!(path.len() == self.height - 1);

        Proof::new(lemma, path)
    }

    /// Returns merkle root
    #[inline]
    pub fn root(&self) -> T {
        self.root.clone()
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

    /// Returns merkle root
    #[inline]
    pub fn read_at(&self, i: usize) -> T {
        self.data.read_at(i)
    }

    pub fn read_range(&self, start: usize, end: usize) -> Vec<T> {
        assert!(start < end);
        self.data.read_range(start..end)
    }

    /// Reads into a pre-allocated slice (for optimization purposes).
    pub fn read_into(&self, pos: usize, buf: &mut [u8]) {
        self.data.read_into(pos, buf);
    }

    /// Build the tree given a slice of all leafs, in bytes form.
    pub fn from_byte_slice(leafs: &[u8]) -> Self {
        assert_eq!(
            leafs.len() % T::byte_len(),
            0,
            "{} not a multiple of {}",
            leafs.len(),
            T::byte_len()
        );

        let leafs_count = leafs.len() / T::byte_len();
        assert!(leafs_count > 1);

        let pow = next_pow2(leafs_count);
        let data = K::new_from_slice(2 * pow - 1, leafs).expect("Failed to create data");

        Self::build(data, leafs_count, log2_pow2(2 * pow))
    }
}

pub trait FromIndexedParallelIterator<T>
where
    T: Send,
{
    fn from_par_iter<I>(par_iter: I) -> Self
    where
        I: IntoParallelIterator<Item = T>,
        I::Iter: IndexedParallelIterator;
}

impl<T: Element, A: Algorithm<T>, K: Store<T>> FromIndexedParallelIterator<T>
    for MerkleTree<T, A, K>
{
    /// Creates new merkle tree from an iterator over hashable objects.
    fn from_par_iter<I>(into: I) -> Self
    where
        I: IntoParallelIterator<Item = T>,
        I::Iter: IndexedParallelIterator,
    {
        let iter = into.into_par_iter();

        let leafs = iter.opt_len().expect("must be sized");
        let pow = next_pow2(leafs);

        let mut data = K::new(2 * pow - 1).expect("Failed to create data");

        populate_data_par::<T, A, K, _>(&mut data, iter);

        Self::build(data, leafs, log2_pow2(2 * pow))
    }
}

impl<T: Element, A: Algorithm<T>, K: Store<T>> FromIterator<T> for MerkleTree<T, A, K> {
    /// Creates new merkle tree from an iterator over hashable objects.
    fn from_iter<I: IntoIterator<Item = T>>(into: I) -> Self {
        let iter = into.into_iter();

        let leafs = iter.size_hint().1.unwrap();
        assert!(leafs > 1);

        let pow = next_pow2(leafs);
        let mut data = K::new(2 * pow - 1).expect("Failed to create data");
        populate_data::<T, A, K, I>(&mut data, iter);

        Self::build(data, leafs, log2_pow2(2 * pow))
    }
}

impl Element for [u8; 32] {
    fn byte_len() -> usize {
        32
    }

    fn from_slice(bytes: &[u8]) -> Self {
        if bytes.len() != 32 {
            panic!("invalid length {}, expected 32", bytes.len());
        }
        *array_ref!(bytes, 0, 32)
    }

    fn copy_to_slice(&self, bytes: &mut [u8]) {
        bytes.copy_from_slice(self);
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

pub fn populate_data<T: Element, A: Algorithm<T>, K: Store<T>, I: IntoIterator<Item = T>>(
    data: &mut K,
    iter: <I as std::iter::IntoIterator>::IntoIter,
) {
    assert!(data.len() == 0);
    let mut buf = Vec::with_capacity(BUILD_DATA_BLOCK_SIZE * T::byte_len());

    let mut a = A::default();
    for item in iter {
        a.reset();
        buf.extend(a.leaf(item).as_ref());
        if buf.len() >= BUILD_DATA_BLOCK_SIZE * T::byte_len() {
            let data_len = data.len();
            // FIXME: Integrate into `len()` call into `copy_from_slice`
            // once we update to `stable` 1.36.
            data.copy_from_slice(&buf, data_len);
            buf.clear();
        }
    }
    let data_len = data.len();
    data.copy_from_slice(&buf, data_len);

    data.sync();
}

fn populate_data_par<T, A, K, I>(data: &mut K, iter: I)
where
    T: Element,
    A: Algorithm<T>,
    K: Store<T>,
    I: ParallelIterator<Item = T> + IndexedParallelIterator,
{
    assert!(data.len() == 0);
    let store = Arc::new(RwLock::new(data));

    iter.chunks(BUILD_DATA_BLOCK_SIZE)
        .enumerate()
        .for_each(|(index, chunk)| {
            let mut a = A::default();
            let mut buf = Vec::with_capacity(BUILD_DATA_BLOCK_SIZE * T::byte_len());

            for item in chunk {
                a.reset();
                buf.extend(a.leaf(item).as_ref());
            }
            store
                .write()
                .unwrap()
                .copy_from_slice(&buf[..], BUILD_DATA_BLOCK_SIZE * index)
        });

    store.write().unwrap().sync();
}
