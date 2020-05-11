use std::fmt;
use std::fs::OpenOptions;
use std::io::Read;
use std::iter::FromIterator;
use std::ops;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use anyhow::Result;
use positioned_io::ReadAt;
use rayon::iter::plumbing::*;
use rayon::iter::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use typenum::marker_traits::Unsigned;

use crate::hash::Algorithm;
use crate::merkle::{get_merkle_tree_height, log2_pow2, next_pow2, Element};

pub const DEFAULT_CACHED_ABOVE_BASE_LAYER_BINARY: usize = 7;
pub const DEFAULT_CACHED_ABOVE_BASE_LAYER_QUAD: usize = 4;
pub const DEFAULT_CACHED_ABOVE_BASE_LAYER_OCT: usize = 2;

/// Tree size (number of nodes) used as threshold to decide which build algorithm
/// to use. Small trees (below this value) use the old build algorithm, optimized
/// for speed rather than memory, allocating as much as needed to allow multiple
/// threads to work concurrently without interrupting each other. Large trees (above)
/// use the new build algorithm, optimized for memory rather than speed, allocating
/// as less as possible with multiple threads competing to get the write lock.
pub const SMALL_TREE_BUILD: usize = 1024;

// Number of nodes to process in parallel during the `build` stage.
pub const BUILD_CHUNK_NODES: usize = 1024 * 4;

mod disk;
mod level_cache;
mod mmap;
mod vec;

pub use disk::DiskStore;
pub use level_cache::LevelCacheStore;
pub use mmap::MmapStore;
pub use vec::VecStore;

#[derive(Clone)]
pub struct ExternalReader<R: Read + Send + Sync> {
    pub offset: usize,
    pub source: R,
    pub read_fn: fn(start: usize, end: usize, buf: &mut [u8], source: &R) -> Result<usize>,
}

impl<R: Read + Send + Sync> ExternalReader<R> {
    pub fn read(&self, start: usize, end: usize, buf: &mut [u8]) -> Result<usize> {
        (self.read_fn)(start + self.offset, end + self.offset, buf, &self.source)
    }
}

impl ExternalReader<std::fs::File> {
    pub fn new_from_config(replica_config: &ReplicaConfig, index: usize) -> Result<Self> {
        let reader = OpenOptions::new().read(true).open(&replica_config.path)?;

        Ok(ExternalReader {
            offset: replica_config.offsets[index],
            source: reader,
            read_fn: |start, end, buf: &mut [u8], reader: &std::fs::File| {
                reader.read_exact_at(start as u64, &mut buf[0..end - start])?;

                Ok(end - start)
            },
        })
    }

    pub fn new_from_path(path: &PathBuf) -> Result<Self> {
        Self::new_from_config(&ReplicaConfig::from(path), 0)
    }
}

impl<R: Read + Send + Sync> fmt::Debug for ExternalReader<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExternalReader")
            .field("source: Read + Send + Sync", &1)
            .field(
                "read_fn: callback(start: usize, end: usize, buf: &mut [u8])",
                &2,
            )
            .finish()
    }
}

// Version 1 always contained the base layer data (even after 'compact').
// Version 2 no longer contains the base layer data after compact.
#[derive(Clone, Copy, Debug)]
pub enum StoreConfigDataVersion {
    One = 1,
    Two = 2,
}

const DEFAULT_STORE_CONFIG_DATA_VERSION: u32 = StoreConfigDataVersion::Two as u32;

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ReplicaConfig {
    pub path: PathBuf,
    pub offsets: Vec<usize>,
}

impl ReplicaConfig {
    pub fn new<T: Into<PathBuf>>(path: T, offsets: Vec<usize>) -> Self {
        ReplicaConfig {
            path: path.into(),
            offsets,
        }
    }
}

impl From<&PathBuf> for ReplicaConfig {
    fn from(path: &PathBuf) -> Self {
        ReplicaConfig {
            path: path.clone(),
            offsets: vec![0],
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct StoreConfig {
    /// A directory in which data (a merkle tree) can be persisted.
    pub path: PathBuf,

    /// A unique identifier used to help specify the on-disk store
    /// location for this particular data.
    pub id: String,

    /// The number of elements in the DiskStore.  This field is
    /// optional, and unused internally.
    pub size: Option<usize>,

    /// The number of merkle tree levels above the base to cache on disk.
    pub levels: usize,
}

impl StoreConfig {
    pub fn new<T: Into<PathBuf>, S: Into<String>>(path: T, id: S, levels: usize) -> Self {
        StoreConfig {
            path: path.into(),
            id: id.into(),
            size: None,
            levels,
        }
    }

    // If the tree is large enough to use the default, use it.  If
    // it's too small to cache anything, don't cache anything.
    // Otherwise, the tree is 'small' so a fixed value of 2 levels
    // above the base should be sufficient.
    pub fn default_cached_above_base_layer(leafs: usize, branches: usize) -> usize {
        let height = get_merkle_tree_height(leafs, branches);
        if height <= 2 {
            return 0;
        }

        let default_height = match branches {
            2 => DEFAULT_CACHED_ABOVE_BASE_LAYER_BINARY,
            4 => DEFAULT_CACHED_ABOVE_BASE_LAYER_QUAD,
            _ => DEFAULT_CACHED_ABOVE_BASE_LAYER_OCT,
        };

        if height <= default_height {
            2
        } else {
            default_height
        }
    }

    // Deterministically create the data_path on-disk location from a
    // path and specified id.
    pub fn data_path(path: &PathBuf, id: &str) -> PathBuf {
        Path::new(&path).join(format!(
            "sc-{:0>2}-data-{}.dat",
            DEFAULT_STORE_CONFIG_DATA_VERSION, id
        ))
    }

    pub fn from_config<S: Into<String>>(config: &StoreConfig, id: S, size: Option<usize>) -> Self {
        let val = if let Some(size) = size {
            Some(size)
        } else {
            config.size
        };

        StoreConfig {
            path: config.path.clone(),
            id: id.into(),
            size: val,
            levels: config.levels,
        }
    }
}

/// Backing store of the merkle tree.
pub trait Store<E: Element>: std::fmt::Debug + Send + Sync + Sized {
    /// Creates a new store which can store up to `size` elements.
    fn new_with_config(size: usize, branches: usize, config: StoreConfig) -> Result<Self>;
    fn new(size: usize) -> Result<Self>;

    fn new_from_slice_with_config(
        size: usize,
        branches: usize,
        data: &[u8],
        config: StoreConfig,
    ) -> Result<Self>;

    fn new_from_slice(size: usize, data: &[u8]) -> Result<Self>;

    fn new_from_disk(size: usize, branches: usize, config: &StoreConfig) -> Result<Self>;

    fn write_at(&mut self, el: E, index: usize) -> Result<()>;

    // Used to reduce lock contention and do the `E` to `u8`
    // conversion in `build` *outside* the lock.
    // `buf` is a slice of converted `E`s and `start` is its
    // position in `E` sizes (*not* in `u8`).
    fn copy_from_slice(&mut self, buf: &[u8], start: usize) -> Result<()>;

    // compact/shrink resources used where possible.
    fn compact(&mut self, branches: usize, config: StoreConfig, store_version: u32)
        -> Result<bool>;

    // re-instate resource usage where needed.
    fn reinit(&mut self) -> Result<()> {
        Ok(())
    }

    // Removes the store backing (does not require a mutable reference
    // since the config should provide stateless context to what's
    // needed to be removed -- with the exception of in memory stores,
    // where this is arguably not important/needed).
    fn delete(config: StoreConfig) -> Result<()>;

    fn read_at(&self, index: usize) -> Result<E>;
    fn read_range(&self, r: ops::Range<usize>) -> Result<Vec<E>>;
    fn read_into(&self, pos: usize, buf: &mut [u8]) -> Result<()>;
    fn read_range_into(&self, start: usize, end: usize, buf: &mut [u8]) -> Result<()>;

    fn len(&self) -> usize;
    fn loaded_from_disk(&self) -> bool;
    fn is_empty(&self) -> bool;
    fn push(&mut self, el: E) -> Result<()>;
    fn last(&self) -> Result<E> {
        self.read_at(self.len() - 1)
    }

    // Sync contents to disk (if it exists). This function is used to avoid
    // unnecessary flush calls at the cost of added code complexity.
    fn sync(&self) -> Result<()> {
        Ok(())
    }

    #[inline]
    fn build_small_tree<A: Algorithm<E>, U: Unsigned>(
        &mut self,
        leafs: usize,
        height: usize,
    ) -> Result<E> {
        ensure!(leafs % 2 == 0, "Leafs must be a power of two");

        let mut level: usize = 0;
        let mut width = leafs;
        let mut level_node_index = 0;
        let branches = U::to_usize();
        let shift = log2_pow2(branches);

        while width > 1 {
            // Same indexing logic as `build`.
            let (layer, write_start) = {
                let (read_start, write_start) = if level == 0 {
                    // Note that we previously asserted that data.len() == leafs.
                    (0, Store::len(self))
                } else {
                    (level_node_index, level_node_index + width)
                };

                let layer: Vec<_> = self
                    .read_range(read_start..read_start + width)?
                    .par_chunks(branches)
                    .map(|nodes| A::default().multi_node(&nodes, level))
                    .collect();

                (layer, write_start)
            };

            for (i, node) in layer.into_iter().enumerate() {
                self.write_at(node, write_start + i)?;
            }

            level_node_index += width;
            level += 1;
            width >>= shift; // width /= branches;
        }

        ensure!(height == level + 1, "Invalid tree height");
        // The root isn't part of the previous loop so `height` is
        // missing one level.

        self.last()
    }

    fn process_layer<A: Algorithm<E>, U: Unsigned>(
        &mut self,
        width: usize,
        level: usize,
        read_start: usize,
        write_start: usize,
    ) -> Result<()> {
        let branches = U::to_usize();
        let data_lock = Arc::new(RwLock::new(self));

        // Allocate `width` indexes during operation (which is a negligible memory bloat
        // compared to the 32-bytes size of the nodes stored in the `Store`s) and hash each
        // pair of nodes to write them to the next level in concurrent threads.
        // Process `BUILD_CHUNK_NODES` nodes in each thread at a time to reduce contention,
        // optimized for big sector sizes (small ones will just have one thread doing all
        // the work).
        ensure!(BUILD_CHUNK_NODES % branches == 0, "Invalid chunk size");
        Vec::from_iter((read_start..read_start + width).step_by(BUILD_CHUNK_NODES))
            .par_iter()
            .try_for_each(|&chunk_index| -> Result<()> {
                let chunk_size = std::cmp::min(BUILD_CHUNK_NODES, read_start + width - chunk_index);

                let chunk_nodes = {
                    // Read everything taking the lock once.
                    data_lock
                        .read()
                        .unwrap()
                        .read_range(chunk_index..chunk_index + chunk_size)?
                };

                // We write the hashed nodes to the next level in the
                // position that would be "in the middle" of the
                // previous pair (dividing by branches).
                let write_delta = (chunk_index - read_start) / branches;

                let nodes_size = (chunk_nodes.len() / branches) * E::byte_len();
                let hashed_nodes_as_bytes = chunk_nodes.chunks(branches).fold(
                    Vec::with_capacity(nodes_size),
                    |mut acc, nodes| {
                        let h = A::default().multi_node(&nodes, level);
                        acc.extend_from_slice(h.as_ref());
                        acc
                    },
                );

                // Check that we correctly pre-allocated the space.
                ensure!(
                    hashed_nodes_as_bytes.len() == chunk_size / branches * E::byte_len(),
                    "Invalid hashed node length"
                );

                // Write the data into the store.
                data_lock
                    .write()
                    .unwrap()
                    .copy_from_slice(&hashed_nodes_as_bytes, write_start + write_delta)
            })
    }

    // Default merkle-tree build, based on store type.
    fn build<A: Algorithm<E>, U: Unsigned>(
        &mut self,
        leafs: usize,
        height: usize,
        _config: Option<StoreConfig>,
    ) -> Result<E> {
        let branches = U::to_usize();
        ensure!(
            next_pow2(branches) == branches,
            "branches MUST be a power of 2"
        );
        ensure!(Store::len(self) == leafs, "Inconsistent data");
        ensure!(leafs % 2 == 0, "Leafs must be a power of two");

        if leafs <= SMALL_TREE_BUILD {
            return self.build_small_tree::<A, U>(leafs, height);
        }

        let shift = log2_pow2(branches);

        // Process one `level` at a time of `width` nodes. Each level has half the nodes
        // as the previous one; the first level, completely stored in `data`, has `leafs`
        // nodes. We guarantee an even number of nodes per `level`, duplicating the last
        // node if necessary.
        let mut level: usize = 0;
        let mut width = leafs;
        let mut level_node_index = 0;
        while width > 1 {
            // Start reading at the beginning of the current level, and writing the next
            // level immediate after.  `level_node_index` keeps track of the current read
            // starts, and width is updated accordingly at each level so that we know where
            // to start writing.
            let (read_start, write_start) = if level == 0 {
                // Note that we previously asserted that data.len() == leafs.
                //(0, data_lock.read().unwrap().len())
                (0, Store::len(self))
            } else {
                (level_node_index, level_node_index + width)
            };

            self.process_layer::<A, U>(width, level, read_start, write_start)?;

            level_node_index += width;
            level += 1;
            width >>= shift; // width /= branches;
        }

        ensure!(height == level + 1, "Invalid tree height");
        // The root isn't part of the previous loop so `height` is
        // missing one level.

        // Return the root
        self.last()
    }
}

// Using a macro as it is not possible to do a generic implementation for all stores.

macro_rules! impl_parallel_iter {
    ($name:ident, $producer:ident, $iter:ident) => {
        impl<E: Element> ParallelIterator for $name<E> {
            type Item = E;

            fn drive_unindexed<C>(self, consumer: C) -> C::Result
            where
                C: UnindexedConsumer<Self::Item>,
            {
                bridge(self, consumer)
            }

            fn opt_len(&self) -> Option<usize> {
                Some(Store::len(self))
            }
        }
        impl<'a, E: Element> ParallelIterator for &'a $name<E> {
            type Item = E;

            fn drive_unindexed<C>(self, consumer: C) -> C::Result
            where
                C: UnindexedConsumer<Self::Item>,
            {
                bridge(self, consumer)
            }

            fn opt_len(&self) -> Option<usize> {
                Some(Store::len(*self))
            }
        }

        impl<E: Element> IndexedParallelIterator for $name<E> {
            fn drive<C>(self, consumer: C) -> C::Result
            where
                C: Consumer<Self::Item>,
            {
                bridge(self, consumer)
            }

            fn len(&self) -> usize {
                Store::len(self)
            }

            fn with_producer<CB>(self, callback: CB) -> CB::Output
            where
                CB: ProducerCallback<Self::Item>,
            {
                callback.callback(<$producer<E>>::new(0, Store::len(&self), &self))
            }
        }

        impl<'a, E: Element> IndexedParallelIterator for &'a $name<E> {
            fn drive<C>(self, consumer: C) -> C::Result
            where
                C: Consumer<Self::Item>,
            {
                bridge(self, consumer)
            }

            fn len(&self) -> usize {
                Store::len(*self)
            }

            fn with_producer<CB>(self, callback: CB) -> CB::Output
            where
                CB: ProducerCallback<Self::Item>,
            {
                callback.callback(<$producer<E>>::new(0, Store::len(self), self))
            }
        }

        #[derive(Debug, Clone)]
        pub struct $producer<'data, E: 'data + Element> {
            pub(crate) current: usize,
            pub(crate) end: usize,
            pub(crate) store: &'data $name<E>,
        }

        impl<'data, E: 'data + Element> $producer<'data, E> {
            pub fn new(current: usize, end: usize, store: &'data $name<E>) -> Self {
                Self {
                    current,
                    end,
                    store,
                }
            }

            pub fn len(&self) -> usize {
                self.end - self.current
            }

            pub fn is_empty(&self) -> bool {
                self.len() == 0
            }
        }

        impl<'data, E: 'data + Element> Producer for $producer<'data, E> {
            type Item = E;
            type IntoIter = $iter<'data, E>;

            fn into_iter(self) -> Self::IntoIter {
                let $producer {
                    current,
                    end,
                    store,
                } = self;

                $iter {
                    current,
                    end,
                    store,
                    err: false,
                }
            }

            fn split_at(self, index: usize) -> (Self, Self) {
                let len = self.len();

                if len == 0 {
                    return (
                        <$producer<E>>::new(0, 0, &self.store),
                        <$producer<E>>::new(0, 0, &self.store),
                    );
                }

                let current = self.current;
                let first_end = current + std::cmp::min(len, index);

                debug_assert!(first_end >= current);
                debug_assert!(current + len >= first_end);

                (
                    <$producer<E>>::new(current, first_end, &self.store),
                    <$producer<E>>::new(first_end, current + len, &self.store),
                )
            }
        }
        #[derive(Debug)]
        pub struct $iter<'data, E: 'data + Element> {
            current: usize,
            end: usize,
            err: bool,
            store: &'data $name<E>,
        }

        impl<'data, E: 'data + Element> $iter<'data, E> {
            fn is_done(&self) -> bool {
                !self.err && self.len() == 0
            }
        }

        impl<'data, E: 'data + Element> Iterator for $iter<'data, E> {
            type Item = E;

            fn next(&mut self) -> Option<Self::Item> {
                if self.is_done() {
                    return None;
                }

                match self.store.read_at(self.current) {
                    Ok(el) => {
                        self.current += 1;
                        Some(el)
                    }
                    _ => {
                        self.err = true;
                        None
                    }
                }
            }
        }

        impl<'data, E: 'data + Element> ExactSizeIterator for $iter<'data, E> {
            fn len(&self) -> usize {
                debug_assert!(self.current <= self.end);
                self.end - self.current
            }
        }

        impl<'data, E: 'data + Element> DoubleEndedIterator for $iter<'data, E> {
            fn next_back(&mut self) -> Option<Self::Item> {
                if self.is_done() {
                    return None;
                }

                match self.store.read_at(self.end - 1) {
                    Ok(el) => {
                        self.end -= 1;
                        Some(el)
                    }
                    _ => {
                        self.err = true;
                        None
                    }
                }
            }
        }
    };
}

impl_parallel_iter!(VecStore, VecStoreProducer, VecStoreIter);
impl_parallel_iter!(DiskStore, DiskStoreProducer, DiskIter);
//impl_parallel_iter!(LevelCacheStore, LevelCacheStoreProducer, LevelCacheIter);
