use hash::{Algorithm, Hashable};
use memmap::MmapMut;
use memmap::MmapOptions;
use positioned_io::{ReadAt, WriteAt};
use proof::Proof;
use rayon::prelude::*;
use std::fs::File;
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::ops::{self, Index};
use std::sync::{Arc, RwLock};
use tempfile::tempfile;

/// Tree size (number of nodes) used as threshold to decide which build algorithm
/// to use. Small trees (below this value) use the old build algorithm, optimized
/// for speed rather than memory, allocating as much as needed to allow multiple
/// threads to work concurrently without interrupting each other. Large trees (above)
/// use the new build algorithm, optimized for memory rather than speed, allocating
/// as less as possible with multiple threads competing to get the write lock.
pub const SMALL_TREE_BUILD: usize = 1024;

// Number of nodes to process in parallel during the `build` stage.
pub const BUILD_CHUNK_NODES: usize = 1024;

// Number of batched nodes processed and stored together in `populate_leaves` to
// avoid single `push`es which degrades performance for `DiskStore`.
pub const BUILD_LEAVES_BLOCK_SIZE: usize = 64 * BUILD_CHUNK_NODES;

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
    K: Store<T>,
{
    leaves: K,
    top_half: K,
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

/// Backing store of the merkle tree.
pub trait Store<E: Element>:
    ops::Deref<Target = [E]> + std::fmt::Debug + Clone + Send + Sync
{
    /// Creates a new store which can store up to `size` elements.
    // FIXME: Return errors on failure instead of panicking
    //  (see https://github.com/filecoin-project/merkle_light/issues/19).
    fn new(size: usize) -> Self;

    fn new_from_slice(size: usize, data: &[u8]) -> Self;

    fn write_at(&mut self, el: E, i: usize);

    // Used to reduce lock contention and do the `E` to `u8`
    // conversion in `build` *outside* the lock.
    // `buf` is a slice of converted `E`s and `start` is its
    // position in `E` sizes (*not* in `u8`).
    fn copy_from_slice(&mut self, buf: &[u8], start: usize);

    fn read_at(&self, i: usize) -> E;
    fn read_range(&self, r: ops::Range<usize>) -> Vec<E>;
    fn read_into(&self, pos: usize, buf: &mut [u8]);

    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn push(&mut self, el: E);

    // Sync contents to disk (if it exists). This function is used to avoid
    // unnecessary flush calls at the cost of added code complexity.
    fn sync(&self) {}
}

#[derive(Debug, Clone)]
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
        if self.0.len() <= i {
            self.0.resize(i + 1, E::default());
        }

        self.0[i] = el;
    }

    // NOTE: Performance regression. To conform with the current API we are
    // unnecessarily converting to and from `&[u8]` in the `VecStore` which
    // already stores `E` (in contrast with the `mmap` versions). We are
    // prioritizing performance for the `mmap` case which will be used in
    // production (`VecStore` is mainly for testing and backwards compatibility).
    fn copy_from_slice(&mut self, buf: &[u8], start: usize) {
        assert_eq!(buf.len() % E::byte_len(), 0);
        let num_elem = buf.len() / E::byte_len();

        if self.0.len() < start + num_elem {
            self.0.resize(start + num_elem, E::default());
        }

        self.0.splice(
            start..start + num_elem,
            buf.chunks_exact(E::byte_len()).map(E::from_slice),
        );
    }

    fn new_from_slice(size: usize, data: &[u8]) -> Self {
        let mut v: Vec<_> = data
            .chunks_exact(E::byte_len())
            .map(E::from_slice)
            .collect();
        let additional = size - v.len();
        v.reserve(additional);

        VecStore(v)
    }

    fn read_at(&self, i: usize) -> E {
        self.0[i].clone()
    }

    fn read_into(&self, i: usize, buf: &mut [u8]) {
        self.0[i].copy_to_slice(buf);
    }

    fn read_range(&self, r: ops::Range<usize>) -> Vec<E> {
        self.0.index(r).to_vec()
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
    fn new(size: usize) -> Self {
        let byte_len = E::byte_len() * size;

        MmapStore {
            store: MmapOptions::new().len(byte_len).map_anon().unwrap(),
            len: 0,
            _e: Default::default(),
        }
    }

    fn new_from_slice(size: usize, data: &[u8]) -> Self {
        assert_eq!(data.len() % E::byte_len(), 0);

        let mut res = Self::new(size);

        let end = data.len();
        res.store[..end].copy_from_slice(data);
        res.len = data.len() / E::byte_len();

        res
    }

    // Writing at positions `i` will mark all other positions as
    // occupied with respect to the `len()` so the new `len()`
    // is `>= i`.
    fn write_at(&mut self, el: E, i: usize) {
        let b = E::byte_len();
        self.store[i * b..(i + 1) * b].copy_from_slice(el.as_ref());
        self.len = std::cmp::max(self.len, i + 1);
    }

    fn copy_from_slice(&mut self, buf: &[u8], start: usize) {
        let b = E::byte_len();
        assert_eq!(buf.len() % b, 0);
        self.store[start * b..start * b + buf.len()].copy_from_slice(buf);
        self.len = std::cmp::max(self.len, start + buf.len() / b);
    }

    fn read_at(&self, i: usize) -> E {
        let b = E::byte_len();
        let start = i * b;
        let end = (i + 1) * b;
        let len = self.len * b;
        assert!(start < len, "start out of range {} >= {}", start, len);
        assert!(end <= len, "end out of range {} > {}", end, len);

        E::from_slice(&self.store[start..end])
    }

    fn read_into(&self, i: usize, buf: &mut [u8]) {
        let b = E::byte_len();
        let start = i * b;
        let end = (i + 1) * b;
        let len = self.len * b;
        assert!(start < len, "start out of range {} >= {}", start, len);
        assert!(end <= len, "end out of range {} > {}", end, len);

        buf.copy_from_slice(&self.store[start..end]);
    }

    fn read_range(&self, r: ops::Range<usize>) -> Vec<E> {
        let b = E::byte_len();
        let start = r.start * b;
        let end = r.end * b;
        let len = self.len * b;
        assert!(start < len, "start out of range {} >= {}", start, len);
        assert!(end <= len, "end out of range {} > {}", end, len);

        self.store[start..end]
            .chunks(b)
            .map(E::from_slice)
            .collect()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn push(&mut self, el: E) {
        let l = self.len;
        assert!(
            (l + 1) * E::byte_len() <= self.store.len(),
            "not enough space"
        );

        self.write_at(el, l);
    }
}

impl<E: Element> Clone for MmapStore<E> {
    fn clone(&self) -> MmapStore<E> {
        MmapStore::new_from_slice(
            self.store.len() / E::byte_len(),
            &self.store[..(self.len() * E::byte_len())],
        )
    }
}

/// Disk-only store use to reduce memory to the minimum at the cost of build
/// time performance. Most of its I/O logic is in the `store_copy_from_slice`
/// and `store_read_range` functions.
#[derive(Debug)]
pub struct DiskStore<E: Element> {
    len: usize,
    _e: PhantomData<E>,
    file: File,

    // We cache the `store.len()` call to avoid accessing disk unnecessarily.
    // Not to be confused with `len`, this saves the total size of the `store`
    // in bytes and the other one keeps track of used `E` slots in the `DiskStore`.
    store_size: usize,
}

impl<E: Element> ops::Deref for DiskStore<E> {
    type Target = [E];

    fn deref(&self) -> &Self::Target {
        unimplemented!()
    }
}

impl<E: Element> Store<E> for DiskStore<E> {
    #[allow(unsafe_code)]
    fn new(size: usize) -> Self {
        let byte_len = E::byte_len() * size;
        let file = tempfile().expect("couldn't create temp file");
        file.set_len(byte_len as u64)
            .unwrap_or_else(|_| panic!("couldn't set len of {}", byte_len));

        DiskStore {
            len: 0,
            _e: Default::default(),
            file,
            store_size: byte_len,
        }
    }

    fn new_from_slice(size: usize, data: &[u8]) -> Self {
        assert_eq!(data.len() % E::byte_len(), 0);

        let mut res = Self::new(size);

        res.store_copy_from_slice(0, data);
        res.len = data.len() / E::byte_len();

        res
    }

    fn write_at(&mut self, el: E, i: usize) {
        let b = E::byte_len();
        self.store_copy_from_slice(i * b, el.as_ref());
        self.len = std::cmp::max(self.len, i + 1);
    }

    fn copy_from_slice(&mut self, buf: &[u8], start: usize) {
        let b = E::byte_len();
        assert_eq!(buf.len() % b, 0);
        self.store_copy_from_slice(start * b, buf);
        self.len = std::cmp::max(self.len, start + buf.len() / b);
    }

    fn read_at(&self, i: usize) -> E {
        let b = E::byte_len();
        let start = i * b;
        let end = (i + 1) * b;
        let len = self.len * b;
        assert!(start < len, "start out of range {} >= {}", start, len);
        assert!(end <= len, "end out of range {} > {}", end, len);

        E::from_slice(&self.store_read_range(start, end))
    }

    fn read_into(&self, i: usize, buf: &mut [u8]) {
        let b = E::byte_len();
        let start = i * b;
        let end = (i + 1) * b;
        let len = self.len * b;
        assert!(start < len, "start out of range {} >= {}", start, len);
        assert!(end <= len, "end out of range {} > {}", end, len);

        self.store_read_into(start, end, buf);
    }

    fn read_range(&self, r: ops::Range<usize>) -> Vec<E> {
        let b = E::byte_len();
        let start = r.start * b;
        let end = r.end * b;
        let len = self.len * b;
        assert!(start < len, "start out of range {} >= {}", start, len);
        assert!(end <= len, "end out of range {} > {}", end, len);

        self.store_read_range(start, end)
            .chunks(b)
            .map(E::from_slice)
            .collect()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn push(&mut self, el: E) {
        let l = self.len;
        assert!(
            (l + 1) * E::byte_len() <= self.store_size(),
            format!(
                "not enough space, l: {}, E size {}, store len {}",
                l,
                E::byte_len(),
                self.store_size()
            )
        );

        self.write_at(el, l);
    }

    fn sync(&self) {
        self.file.sync_all().expect("failed to sync file");
    }
}

impl<E: Element> DiskStore<E> {
    pub fn store_size(&self) -> usize {
        self.store_size
    }

    pub fn store_read_range(&self, start: usize, end: usize) -> Vec<u8> {
        let read_len = end - start;
        let mut read_data = vec![0; read_len];

        assert_eq!(
            self.file
                .read_at(start as u64, &mut read_data)
                .unwrap_or_else(|_| panic!(
                    "failed to read {} bytes from file at offset {}",
                    read_len, start
                )),
            read_len
        );

        read_data
    }

    pub fn store_read_into(&self, start: usize, end: usize, buf: &mut [u8]) {
        buf.copy_from_slice(&self.store_read_range(start, end));
    }

    pub fn store_copy_from_slice(&mut self, start: usize, slice: &[u8]) {
        assert!(start + slice.len() <= self.store_size);
        self.file
            .write_at(start as u64, slice)
            .expect("failed to write file");
    }
}

// FIXME: Fake `Clone` implementation to accomodate the artificial call in
//  `from_data_with_store`, we won't actually duplicate the mmap memory,
//  just recreate the same object (as the original will be dropped).
impl<E: Element> Clone for DiskStore<E> {
    fn clone(&self) -> DiskStore<E> {
        unimplemented!("We can't clone a store with an already associated file");
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
    fn build(leaves: K, top_half: K, leafs: usize, height: usize) -> Self {
        // This algorithms assumes that the underlying store has preallocated enough space.
        // TODO: add an assert here to ensure this is the case.

        if leafs <= SMALL_TREE_BUILD {
            return Self::build_small_tree(leaves, top_half, leafs, height);
        }

        let leaves_lock = Arc::new(RwLock::new(leaves));
        let top_half_lock = Arc::new(RwLock::new(top_half));

        // Process one `level` at a time of `width` nodes. Each level has half the nodes
        // as the previous one; the first level, completely stored in `leaves`, has `leafs`
        // nodes. We guarantee an even number of nodes per `level`, duplicating the last
        // node if necessary.
        // `level_node_index` keeps the "global" index of the first node of the current
        // level: the index we would have if the `leaves` and `top_half` were unified
        // in the same `Store`; it is later converted to the "local" index to access each
        // individual `Store` (according to which `level` we're processing at the moment).
        // We always write to the `top_half` (which contains all the levels but the first
        // one) of the tree and only read from the `leaves` in the first iteration
        // (at `level` 0).
        let mut level: usize = 0;
        let mut width = leafs;
        let mut level_node_index = 0;
        while width > 1 {
            if width & 1 == 1 {
                // Odd number of nodes, duplicate last.
                let mut active_store = if level == 0 {
                    leaves_lock.write().unwrap()
                } else {
                    top_half_lock.write().unwrap()
                };
                let last_node = active_store.read_at(active_store.len() - 1);
                active_store.push(last_node);

                width += 1;
            }

            // We read the `width` nodes of the current `level` from `read_store` and
            // write (half of it) in the `write_store` (which contains the next level).
            // Both `read_start` and `write_start` are "local" indexes with respect to
            // the `read_store` and `write_store` they are accessing.
            let (read_store_lock, write_store_lock, read_start, write_start) = if level == 0 {
                // The first level is in the `leaves`, which is all it contains so the
                // next level to write to will be in the `top_half`. Since we are "jumping"
                // from one `Store` to the other both read/write start indexes start at zero.
                (leaves_lock.clone(), top_half_lock.clone(), 0, 0)
            } else {
                // For all other levels we'll read/write from/to the `top_half` adjusting the
                // "global" index to access this `Store` (offsetting `leaves` length). All levels
                // are contiguous so we read/write `width` nodes apart.
                let read_start = level_node_index - leaves_lock.read().unwrap().len();
                (
                    top_half_lock.clone(),
                    top_half_lock.clone(),
                    read_start,
                    read_start + width,
                )
            };
            // FIXME: Maybe just remove `write_store_lock` and always access `top_half_lock`
            // directly if it makes it more readable.

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
                    let chunk_size =
                        std::cmp::min(BUILD_CHUNK_NODES, read_start + width - chunk_index);

                    let chunk_nodes = {
                        // Read everything taking the lock once.
                        let read_store = read_store_lock.read().unwrap();
                        read_store.read_range(chunk_index..chunk_index + chunk_size)
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

                    write_store_lock
                        .write()
                        .unwrap()
                        .copy_from_slice(&hashed_nodes_as_bytes, write_start + write_delta);
                });

            level_node_index += width;
            level += 1;
            width >>= 1;
            write_store_lock.write().unwrap().sync();
        }

        assert_eq!(height, level + 1);
        // The root isn't part of the previous loop so `height` is
        // missing one level.

        let root = {
            let top_half = top_half_lock.read().unwrap();
            top_half.read_at(top_half.len() - 1)
        };

        MerkleTree {
            leaves: Arc::try_unwrap(leaves_lock).unwrap().into_inner().unwrap(),
            top_half: Arc::try_unwrap(top_half_lock)
                .unwrap()
                .into_inner()
                .unwrap(),
            leafs,
            height,
            root,
            _a: PhantomData,
            _t: PhantomData,
        }
    }

    #[inline]
    fn build_small_tree(mut leaves: K, mut top_half: K, leafs: usize, height: usize) -> Self {
        let mut level: usize = 0;
        let mut width = leafs;
        let mut level_node_index = 0;
        while width > 1 {
            if width & 1 == 1 {
                if level == 0 {
                    let last_node = leaves.read_at(leaves.len() - 1);
                    leaves.push(last_node);
                } else {
                    let last_node = top_half.read_at(top_half.len() - 1);
                    top_half.push(last_node);
                }
                width += 1;
            }

            // Same indexing logic as `build`.
            let (layer, write_start) = {
                let (read_store, read_start, write_start) = if level == 0 {
                    (&leaves, 0, 0)
                } else {
                    let read_start = level_node_index - leaves.len();
                    (&top_half, read_start, read_start + width)
                };

                let layer: Vec<_> = read_store
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
            // FIXME: Just to make the borrow checker happy, ideally the `top_half` borrow
            // should end with `read_store` access.

            for (i, node) in layer.into_iter().enumerate() {
                top_half.write_at(node, write_start + i);
            }

            level_node_index += width;
            level += 1;
            width >>= 1;
        }

        assert_eq!(height, level + 1);
        // The root isn't part of the previous loop so `height` is
        // missing one level.

        let root = { top_half.read_at(top_half.len() - 1) };

        MerkleTree {
            leaves,
            top_half,
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
        self.leaves.len() + self.top_half.len()
    }

    /// Returns `true` if the vector contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.leaves.is_empty() && self.top_half.is_empty()
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
        if i < self.leaves.len() {
            self.leaves.read_at(i)
        } else {
            self.top_half.read_at(i - self.leaves.len())
        }
    }

    // With the leaves decoupled from the rest of the tree we need to split
    // the range if necessary. If the range is covered by a single `Store`
    // we just call its `read_range`, if not, we need to form a new `Vec`
    // to hold both parts.
    // FIXME: The second mechanism can be *very* expensive with big sectors,
    // should the consumer be aware of this to avoid memory bloats?
    pub fn read_range(&self, start: usize, end: usize) -> Vec<T> {
        if start > end {
            panic!("read_range: start > end ({} > {})", start, end);
            // FIXME: Do we need to check this? The implementations of
            // `Store` don't (does `Range` take care of it?).
        }

        let leaves_len = self.leaves.len();
        if end <= self.leaves.len() {
            self.leaves.read_range(start..end)
        } else if start >= self.leaves.len() {
            self.top_half
                .read_range(start - leaves_len..end - leaves_len)
        } else {
            let mut joined = Vec::with_capacity(end - start);
            joined.append(&mut self.leaves.read_range(start..leaves_len));
            joined.append(&mut self.top_half.read_range(0..end - leaves_len));
            joined
        }
    }

    /// Reads into a pre-allocated slice (for optimization purposes).
    pub fn read_into(&self, pos: usize, buf: &mut [u8]) {
        if pos < self.leaves.len() {
            self.leaves.read_into(pos, buf);
        } else {
            self.top_half.read_into(pos - self.leaves.len(), buf);
        }
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
        let pow = next_pow2(leafs_count);

        let leaves = K::new_from_slice(pow, leafs);
        let top_half = K::new(pow);

        assert!(leafs_count > 1);
        Self::build(leaves, top_half, leafs_count, log2_pow2(2 * pow))
    }
}

impl<T: Element, A: Algorithm<T>, K: Store<T>> FromParallelIterator<T> for MerkleTree<T, A, K> {
    /// Creates new merkle tree from an iterator over hashable objects.
    fn from_par_iter<I: IntoParallelIterator<Item = T>>(into: I) -> Self {
        let iter = into.into_par_iter();

        let leafs = iter.opt_len().expect("must be sized");
        let pow = next_pow2(leafs);

        let mut leaves = K::new(pow);
        let top_half = K::new(pow);

        populate_leaves_par::<T, A, K, I>(&mut leaves, iter);

        // // leafs
        // let vs = iter
        //     .map(|item| {
        //         let mut a = A::default();
        //         a.leaf(item)
        //     })
        //     .collect::<Vec<_>>();

        // for v in vs.into_iter() {
        //     leaves.push(v);
        // }
        // leaves.sync();
        // FIXME: Use a similar construction to `populate_leaves`
        // for parallel threads.
        // ^^^

        Self::build(leaves, top_half, leafs, log2_pow2(2 * pow))
    }
}

impl<T: Element, A: Algorithm<T>, K: Store<T>> FromIterator<T> for MerkleTree<T, A, K> {
    /// Creates new merkle tree from an iterator over hashable objects.
    fn from_iter<I: IntoIterator<Item = T>>(into: I) -> Self {
        let iter = into.into_iter();

        let leafs = iter.size_hint().1.unwrap();
        assert!(leafs > 1);

        let pow = next_pow2(leafs);

        let mut leaves = K::new(pow);
        let top_half = K::new(pow);

        populate_leaves::<T, A, K, I>(&mut leaves, iter);

        Self::build(leaves, top_half, leafs, log2_pow2(2 * pow))
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

fn populate_leaves<T: Element, A: Algorithm<T>, K: Store<T>, I: IntoIterator<Item = T>>(
    leaves: &mut K,
    iter: <I as std::iter::IntoIterator>::IntoIter,
) {
    let mut buf = Vec::with_capacity(BUILD_LEAVES_BLOCK_SIZE * T::byte_len());

    let mut a = A::default();
    for item in iter {
        a.reset();
        buf.extend(a.leaf(item).as_ref());
        if buf.len() >= BUILD_LEAVES_BLOCK_SIZE * T::byte_len() {
            let leaves_len = leaves.len();
            // FIXME: Integrate into `len()` call into `copy_from_slice`
            // once we update to `stable` 1.36.
            leaves.copy_from_slice(&buf, leaves_len);
            buf.clear();
        }
    }
    let leaves_len = leaves.len();
    leaves.copy_from_slice(&buf, leaves_len);

    leaves.sync();
}

// FIXME: Copied from `populate_leaves`, can we unify the code?
fn populate_leaves_par<T: Element, A: Algorithm<T>, K: Store<T>, I: IntoParallelIterator<Item = T>>(
    leaves: &mut K,
    iter: <I as rayon::iter::IntoParallelIterator>::Iter,
) {
    let mut buf = Vec::with_capacity(BUILD_LEAVES_BLOCK_SIZE * T::byte_len());

    let mut a = A::default();
    for item in iter {
        a.reset();
        buf.extend(a.leaf(item).as_ref());
        if buf.len() >= BUILD_LEAVES_BLOCK_SIZE * T::byte_len() {
            let leaves_len = leaves.len();
            // FIXME: Integrate into `len()` call into `copy_from_slice`
            // once we update to `stable` 1.36.
            leaves.copy_from_slice(&buf, leaves_len);
            buf.clear();
        }
    }
    let leaves_len = leaves.len();
    leaves.copy_from_slice(&buf, leaves_len);

    leaves.sync();
}
