use hash::{Algorithm, Hashable};
use failure::Error;
use proof::Proof;
use rayon::prelude::*;
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};
use store::{Store, StoreConfig, VecStore};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub struct ProofAndTree<T, A, K>
where
    T: Element,
    A: Algorithm<T>,
    K: Store<T>
{
    pub proof: Proof<T>,
    pub merkle_tree: MerkleTree<T, A, K>
}

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
#[derive(Debug, Clone, Eq, PartialEq, Default)]
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

    /// Creates new merkle from a sequence of hashes.
    pub fn new_with_config<I: IntoIterator<Item = T>>(data: I, config: StoreConfig) -> MerkleTree<T, A, K> {
        Self::from_iter_with_config(data, config)
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

    /// Creates new merkle tree from a list of hashable objects.
    pub fn from_data_with_config<O: Hashable<A>, I: IntoIterator<Item = O>>(data: I, config: StoreConfig) -> MerkleTree<T, A, K> {
        let mut a = A::default();
        Self::from_iter_with_config(data.into_iter().map(|x| {
            a.reset();
            x.hash(&mut a);
            a.hash()
        }), config)
    }

    /// Creates new merkle tree from an already allocated `Store`
    /// (used with `*Store::new_from_disk`).
    pub fn from_data_store(data: K, leafs: usize) -> MerkleTree<T, A, K> {
        let pow = next_pow2(leafs);
        let height = log2_pow2(2 * pow);

        let elements = data.len() / 2 + 1;
        let root = data.read_at(data.len() - 1);

        MerkleTree {
            data,
            leafs: elements,
            height,
            root,
            _a: PhantomData,
            _t: PhantomData,
        }
    }

    fn build(data: K, leafs: usize, height: usize) -> Self {
        // If the incoming is likely already the fully built data, use
        // it instead of rebuilding the data.  This is the normal case
        // for DiskStore::new_from_disk, which can re-use MTs and
        // doesn't need to rebuild anything.
        if data.len() == 2 * leafs - 1 {
            let root = { data.read_at(data.len() - 1) };
            return MerkleTree {
                data,
                leafs,
                height,
                root,
                _a: PhantomData,
                _t: PhantomData,
            };
        }

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

                    // Check that we correctly pre-allocated the space.
                    debug_assert_eq!(hashed_nodes_as_bytes.len(), chunk_size / 2 * T::byte_len());

                    // Write the data into the store.
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

    #[inline]
    fn build_partial_small_tree(mut data: VecStore<T>, leafs: usize, height: usize, stop_index: usize) -> Result<MerkleTree<T, A, VecStore<T>>> {
        let mut level: usize = 0;
        let mut width = leafs;
        let mut level_node_index = 0;

        while width > 1 {
            // NOTE: We assert that the data leaf has already been
            // duplicated if needed from the initial store/data build.
            assert!(leafs % 2 == 0);

            // Same indexing logic as `build_small_tree`.
            let (layer, write_start) = {
                let (read_start, write_start) = if level == 0 {
                    (0, leafs)
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

            // If we get into the already cached region (based on the
            // specified stop_index), stop building the tree since
            // it'll never be accessed by properly behaving callers.
            if level_node_index >= stop_index {
                break;
            }

            level += 1;
            width >>= 1;
        }

        // This root may not be a real root in the case of a partial
        // tree build for the same reason that the height may be
        // incorrect (i.e. we aborted early because we hit the
        // stop_index).
        let root = { data.read_at(data.len() - 1) };

        // Re-claim any memory that was allocated and unused at this
        // point in the data.
        data.compact(Default::default())?;

        Ok(MerkleTree {
            data,
            leafs,
            height,
            root,
            _a: PhantomData,
            _t: PhantomData,
        })
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

    /// Generate merkle tree inclusion proof for leaf `i` by first
    /// building a partial tree (returned) along with the proof.
    pub fn gen_proof_and_partial_tree(&self, i: usize, levels: usize) -> Result<ProofAndTree<T, A, VecStore<T>>> {
        assert!(i < self.leafs); // i in [0 .. self.leafs)

        let mut width = self.leafs;
        if width & 1 == 1 {
            width += 1;
        }

        assert_eq!(width, next_pow2(width));

        let total_size = 2 * width - 1;
        let cache_size = total_size >> levels;
        let partial_height = self.height - levels;
        if cache_size >= total_size {
            return Ok(ProofAndTree {
                proof: self.gen_proof(i),
                merkle_tree: Default::default()
            });
        }

        // Before generating the proof, build the partial tree based
        // on the data side we need it on.
        //
        // FIXME: These perhaps could be smarter about the partial
        // tree width to not overbuild the partial tree.
        let partial_width = width >> 1;
        let offset = if i + 1 > width >> 1 {
            width >> 1
        } else {
            0
        };

        // Copy the proper half of the base data into memory and
        // initialize a VecStore to back a new, smaller MT.
        let mut data_copy = vec![0; partial_width * T::byte_len()];
        self.data.read_range_into(offset, offset + partial_width, &mut data_copy);
        let partial_store = VecStore::new_from_slice(partial_width, &data_copy)
            .expect("Failed to create intermediate Store");
        assert_eq!(partial_store.len(), partial_width);

        // Before building the tree, resize the store where the tree
        // will be built to allow space for the newly constructed layers.
        data_copy.resize(((2 * partial_width) - 1) * T::byte_len(), 0);

        // Build the small tree.  Note that this 'partial_tree' can be
        // trapezoidal in shape, as it stops building upward when it
        // runs into the cached region.
        //
        // FIXME: Eventually that will not be true when we're building
        // only the proper/minimal tree required for the proof.
        let partial_tree: MerkleTree<T, A, VecStore<T>> =
            Self::build_partial_small_tree(
                partial_store, partial_width, partial_height,
                total_size - width - cache_size)?;

        let proof = self.gen_proof_with_partial_tree(i, levels, &partial_tree);

        Ok(ProofAndTree{
            proof,
            merkle_tree: partial_tree
        })
    }

    /// Generate merkle tree inclusion proof for leaf `i` given a
    /// partial tree for lookups where data is otherwise unavailable.
    pub fn gen_proof_with_partial_tree(&self, i: usize, levels: usize, partial_tree: &MerkleTree<T, A, VecStore<T>>) -> Proof<T> {
        assert!(i < self.leafs); // i in [0 .. self.leafs)

        let mut lemma: Vec<T> = Vec::with_capacity(self.height + 1); // path + root
        let mut path: Vec<bool> = Vec::with_capacity(self.height - 1); // path - 1

        let mut base = 0;
        let mut j = i;
        let mut width = self.leafs;
        if width & 1 == 1 {
            width += 1;
        }

        let data_width = width;
        let total_size = 2 * data_width - 1;
        let cache_size = total_size >> levels;

        // Determine the offset where the partial tree provided should
        // have been built from.
        let offset = if i + 1 > width >> 1 {
            width >> 1
        } else {
            0
        };

        let mut offset_level_index = 0;
        let cache_index_start = total_size - cache_size;

        lemma.push(self.read_at(j));
        while base + 1 < self.len() {
            lemma.push(if j & 1 == 0 {
                // j is left
                let left_index = base + j + 1;

                // Check if we can read from either the base data
                // layer, or the cached region (accessed the same way
                // via the store interface).
                if left_index < data_width || left_index >= cache_index_start {
                    self.read_at(left_index)
                } else {
                    // Otherwise, read from the partially built sub-tree.
                    let index = (base >> 1) + j + 1 - (offset >> offset_level_index);
                    partial_tree.read_at(index)
                }
            } else {
                // j is right
                let right_index = base + j - 1;

                // Check if we can read from either the base data
                // layer, or the cached region (accessed the same way
                // via the store interface).
                if right_index < data_width || right_index >= cache_index_start {
                    self.read_at(right_index)
                } else {
                    // Otherwise, read from the partially built sub-tree.
                    let index = (base >> 1) + j - 1 - (offset >> offset_level_index);
                    partial_tree.read_at(index)
                }
            });

            path.push(j & 1 == 0);

            base += width;
            width >>= 1;
            if width & 1 == 1 {
                width += 1;
            }
            j >>= 1;
            if offset != 0 {
                // This keeps track of the sub-tree height, which is
                // required to help us determine the sub-tree offset
                // needed.
                offset_level_index += 1;
            }
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

    /// Truncates the data for later access via LevelCacheStore
    /// interface.
    #[inline]
    pub fn compact(&mut self, config: StoreConfig) -> Result<bool> {
        self.data.compact(config)
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

    /// Returns data reference
    #[inline]
    pub fn data(&self) -> &K {
        &self.data
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

    pub fn read_range_into(&self, start: usize, end: usize, buf: &mut [u8]) {
        assert!(start < end);
        self.data.read_range_into(start, end, buf)
    }

    /// Reads into a pre-allocated slice (for optimization purposes).
    pub fn read_into(&self, pos: usize, buf: &mut [u8]) {
        self.data.read_into(pos, buf);
    }

    /// Build the tree given a slice of all leafs, in bytes form.
    pub fn from_byte_slice_with_config(leafs: &[u8], config: StoreConfig) -> Self {
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
        let data = K::new_from_slice_with_config(2 * pow - 1, leafs, config)
            .expect("Failed to create data store");

        Self::build(data, leafs_count, log2_pow2(2 * pow))
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
        let data = K::new_from_slice(2 * pow - 1, leafs)
            .expect("Failed to create data store");

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

    fn from_par_iter_with_config<I>(par_iter: I, config: StoreConfig) -> Self
    where
        I: IntoParallelIterator<Item = T>,
        I::Iter: IndexedParallelIterator;
}

pub trait FromIteratorWithConfig<T>
where
    T: Send,
{
    fn from_iter_with_config<I>(par_iter: I, config: StoreConfig) -> Self
    where
        I: IntoIterator<Item = T>;
}

// NOTE: This use cannot accept a StoreConfig.
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

        let mut data = K::new(2 * pow - 1)
            .expect("Failed to create data store");
        populate_data_par::<T, A, K, _>(&mut data, iter);

        Self::build(data, leafs, log2_pow2(2 * pow))
    }

    /// Creates new merkle tree from an iterator over hashable objects.
    fn from_par_iter_with_config<I>(into: I, config: StoreConfig) -> Self
    where
        I: IntoParallelIterator<Item = T>,
        I::Iter: IndexedParallelIterator,
    {
        let iter = into.into_par_iter();

        let leafs = iter.opt_len().expect("must be sized");
        let pow = next_pow2(leafs);

        let mut data = K::new_with_config(2 * pow - 1, config)
            .expect("Failed to create data store");

        // If the data store is empty, populate the base layer before
        // building the tree.  If it's not empty, it likely already
        // contains the built tree (as in the case where new_from_disk
        // was invoked).
        if !data.loaded_from_disk() {
            populate_data_par::<T, A, K, _>(&mut data, iter);
        }

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
        let mut data = K::new(2 * pow - 1)
            .expect("Failed to create data store");
        populate_data::<T, A, K, I>(&mut data, iter);

        Self::build(data, leafs, log2_pow2(2 * pow))
    }
}

impl<T: Element, A: Algorithm<T>, K: Store<T>> FromIteratorWithConfig<T> for MerkleTree<T, A, K> {
    /// Creates new merkle tree from an iterator over hashable objects.
    fn from_iter_with_config<I: IntoIterator<Item = T>>(into: I, config: StoreConfig) -> Self {
        let iter = into.into_iter();

        let leafs = iter.size_hint().1.unwrap();
        assert!(leafs > 1);

        let pow = next_pow2(leafs);
        let mut data = K::new_with_config(2 * pow - 1, config)
            .expect("Failed to create data store");

        // If the data store is empty, populate the base layer before
        // building the tree.  If it's not empty, it likely already
        // contains the built tree (as in the case where new_from_disk
        // was invoked).
        if !data.loaded_from_disk() {
            populate_data::<T, A, K, I>(&mut data, iter);
        }

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
    if !data.is_empty() {
        return;
    }

    assert!(data.is_empty());
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
    if !data.is_empty() {
        return;
    }

    assert!(data.is_empty());
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
