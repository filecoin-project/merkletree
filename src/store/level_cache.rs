use std::fmt;
use std::fs::{remove_file, File, OpenOptions};
use std::io::{copy, Read, Seek, SeekFrom};
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::ops;
use std::path::Path;
use std::sync::{Arc, RwLock};

use anyhow::{Context, Result};
use memmap::MmapOptions;
use positioned_io::{ReadAt, WriteAt};
use rayon::iter::*;
use rayon::prelude::*;
use tempfile::tempfile;
use typenum::marker_traits::Unsigned;

use crate::hash::Algorithm;
use crate::merkle::{
    get_merkle_tree_cache_size, get_merkle_tree_leafs, get_merkle_tree_len, log2_pow2, next_pow2,
    Element,
};
use crate::store::{ExternalReader, Store, StoreConfig, BUILD_CHUNK_NODES};

/// The LevelCacheStore is used to reduce the on-disk footprint even
/// further to the minimum at the cost of build time performance.
/// Each LevelCacheStore is created with a StoreConfig object which
/// contains the number of binary tree levels above the base that are
/// 'cached'.  This implementation has hard requirements about the on
/// disk file size based on that number of levels, so on-disk files
/// are tied, structurally to the configuration they were built with
/// and can only be accessed with the same number of levels.
pub struct LevelCacheStore<E: Element, R: Read + Send + Sync> {
    len: usize,
    elem_len: usize,
    file: File,

    // The number of base layer data items.
    data_width: usize,

    // The byte index of where the cached data begins.
    cache_index_start: usize,

    // This flag is useful only immediate after instantiation, which
    // is false if the store was newly initialized and true if the
    // store was loaded from already existing on-disk data.
    loaded_from_disk: bool,

    // We cache the on-disk file size to avoid accessing disk
    // unnecessarily.
    store_size: usize,

    // If provided, the store will use this method to access base
    // layer data.
    reader: Option<ExternalReader<R>>,

    _e: PhantomData<E>,
}

impl<E: Element, R: Read + Send + Sync> fmt::Debug for LevelCacheStore<E, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LevelCacheStore")
            .field("len", &self.len)
            .field("elem_len", &self.len)
            .field("data_width", &self.data_width)
            .field("loaded_from_disk", &self.loaded_from_disk)
            .field("cache_index_start", &self.cache_index_start)
            .field("store_size", &self.store_size)
            .finish()
    }
}

impl<E: Element, R: Read + Send + Sync> LevelCacheStore<E, R> {
    /// Used for opening v2 compacted DiskStores.
    pub fn new_from_disk_with_reader(
        store_range: usize,
        branches: usize,
        config: &StoreConfig,
        reader: ExternalReader<R>,
    ) -> Result<Self> {
        let data_path = StoreConfig::data_path(&config.path, &config.id);

        let file = File::open(data_path)?;
        let metadata = file.metadata()?;
        let store_size = metadata.len() as usize;

        // The LevelCacheStore base data layer must already be a
        // massaged next pow2 (guaranteed if created with
        // DiskStore::compact, which is the only supported method at
        // the moment).
        let size = get_merkle_tree_leafs(store_range, branches)?;
        ensure!(
            size == next_pow2(size),
            "Inconsistent merkle tree height detected"
        );

        // Values below in bytes.
        // Convert store_range from an element count to bytes.
        let store_range = store_range * E::byte_len();

        // LevelCacheStore on disk file is only the cached data, so
        // the file size dictates the cache_size.  Calculate cache
        // start and the updated size with repect to the file size.
        let cache_size = get_merkle_tree_cache_size(size, branches, config.levels)? * E::byte_len();
        let cache_index_start = store_range - cache_size;

        // Sanity checks that the StoreConfig levels matches this
        // particular on-disk file.  Since an external reader *is*
        // set, we check to make sure that the data on disk is *only*
        // the cached element data.
        ensure!(
            store_size == cache_size,
            "Inconsistent store size detected with external reader"
        );

        Ok(LevelCacheStore {
            len: store_range / E::byte_len(),
            elem_len: E::byte_len(),
            file,
            data_width: size,
            cache_index_start,
            store_size,
            loaded_from_disk: false,
            reader: Some(reader),
            _e: Default::default(),
        })
    }

    pub fn set_external_reader(&mut self, reader: ExternalReader<R>) -> Result<()> {
        self.reader = Some(reader);

        Ok(())
    }
}

impl<E: Element, R: Read + Send + Sync> Store<E> for LevelCacheStore<E, R> {
    fn new_with_config(size: usize, branches: usize, config: StoreConfig) -> Result<Self> {
        let data_path = StoreConfig::data_path(&config.path, &config.id);

        // If the specified file exists, load it from disk.  This is
        // the only supported usage of this call for this type of
        // Store.
        if Path::new(&data_path).exists() {
            return Self::new_from_disk(size, branches, &config);
        }

        // Otherwise, create the file and allow it to be the on-disk store.
        let file = OpenOptions::new()
            .write(true)
            .read(true)
            .create_new(true)
            .open(data_path)?;

        let store_size = E::byte_len() * size;
        let leafs = get_merkle_tree_leafs(size, branches)?;

        ensure!(
            leafs == next_pow2(leafs),
            "Inconsistent merkle tree height detected"
        );

        // Calculate cache start and the updated size with repect to
        // the data size.
        let cache_size =
            get_merkle_tree_cache_size(leafs, branches, config.levels)? * E::byte_len();
        let cache_index_start = store_size - cache_size;

        file.set_len(store_size as u64)?;

        Ok(LevelCacheStore {
            len: 0,
            elem_len: E::byte_len(),
            file,
            data_width: leafs,
            cache_index_start,
            store_size,
            loaded_from_disk: false,
            reader: None,
            _e: Default::default(),
        })
    }

    fn new(size: usize) -> Result<Self> {
        let store_size = E::byte_len() * size;
        let file = tempfile()?;
        file.set_len(store_size as u64)?;

        Ok(LevelCacheStore {
            len: 0,
            elem_len: E::byte_len(),
            file,
            data_width: size,
            cache_index_start: 0,
            store_size,
            loaded_from_disk: false,
            reader: None,
            _e: Default::default(),
        })
    }

    fn new_from_slice_with_config(
        size: usize,
        branches: usize,
        data: &[u8],
        config: StoreConfig,
    ) -> Result<Self> {
        ensure!(
            data.len() % E::byte_len() == 0,
            "data size must be a multiple of {}",
            E::byte_len()
        );

        let mut store = Self::new_with_config(size, branches, config)?;

        // If the store was loaded from disk (based on the config
        // information, avoid re-populating the store at this point
        // since it can be assumed by the config that the data is
        // already correct).
        if !store.loaded_from_disk {
            store.store_copy_from_slice(0, data)?;
            store.len = data.len() / store.elem_len;
        }

        Ok(store)
    }

    fn new_from_slice(size: usize, data: &[u8]) -> Result<Self> {
        ensure!(
            data.len() % E::byte_len() == 0,
            "data size must be a multiple of {}",
            E::byte_len()
        );

        let mut store = Self::new(size)?;
        store.store_copy_from_slice(0, data)?;
        store.len = data.len() / store.elem_len;

        Ok(store)
    }

    // Used for opening v1 compacted DiskStores.
    fn new_from_disk(store_range: usize, branches: usize, config: &StoreConfig) -> Result<Self> {
        let data_path = StoreConfig::data_path(&config.path, &config.id);

        let file = File::open(data_path)?;
        let metadata = file.metadata()?;
        let store_size = metadata.len() as usize;

        // The LevelCacheStore base data layer must already be a
        // massaged next pow2 (guaranteed if created with
        // DiskStore::compact, which is the only supported method at
        // the moment).
        let size = get_merkle_tree_leafs(store_range, branches)?;
        ensure!(
            size == next_pow2(size),
            "Inconsistent merkle tree height detected"
        );

        // Values below in bytes.
        // Convert store_range from an element count to bytes.
        let store_range = store_range * E::byte_len();

        // Calculate cache start and the updated size with repect to
        // the data size.
        let cache_size = get_merkle_tree_cache_size(size, branches, config.levels)? * E::byte_len();
        let cache_index_start = store_range - cache_size;

        // For a true v1 compatible store, this check should remain,
        // but since the store structure is identical otherwise this
        // method can be re-used to open v2 stores, so long as an
        // external_reader is set afterward.

        // Sanity checks that the StoreConfig levels matches this
        // particular on-disk file.
        /*
        ensure!(
            store_size == size * E::byte_len() + cache_size,
            "Inconsistent store size detected"
        );
         */

        Ok(LevelCacheStore {
            len: store_range / E::byte_len(),
            elem_len: E::byte_len(),
            file,
            data_width: size,
            cache_index_start,
            loaded_from_disk: true,
            store_size,
            reader: None,
            _e: Default::default(),
        })
    }

    fn write_at(&mut self, el: E, index: usize) -> Result<()> {
        self.store_copy_from_slice(index * self.elem_len, el.as_ref())?;
        self.len = std::cmp::max(self.len, index + 1);

        Ok(())
    }

    fn copy_from_slice(&mut self, buf: &[u8], start: usize) -> Result<()> {
        ensure!(
            buf.len() % self.elem_len == 0,
            "buf size must be a multiple of {}",
            self.elem_len
        );
        self.store_copy_from_slice(start * self.elem_len, buf)?;
        self.len = std::cmp::max(self.len, start + buf.len() / self.elem_len);

        Ok(())
    }

    fn read_at(&self, index: usize) -> Result<E> {
        let start = index * self.elem_len;
        let end = start + self.elem_len;

        let len = self.len * self.elem_len;
        ensure!(start < len, "start out of range {} >= {}", start, len);
        ensure!(end <= len, "end out of range {} > {}", end, len);
        ensure!(
            start <= self.data_width * self.elem_len || start >= self.cache_index_start,
            "out of bounds"
        );

        Ok(E::from_slice(&self.store_read_range(start, end)?))
    }

    fn read_into(&self, index: usize, buf: &mut [u8]) -> Result<()> {
        let start = index * self.elem_len;
        let end = start + self.elem_len;

        let len = self.len * self.elem_len;
        ensure!(start < len, "start out of range {} >= {}", start, len);
        ensure!(end <= len, "end out of range {} > {}", end, len);
        ensure!(
            start <= self.data_width * self.elem_len || start >= self.cache_index_start,
            "out of bounds"
        );

        self.store_read_into(start, end, buf)
    }

    fn read_range_into(&self, start: usize, end: usize, buf: &mut [u8]) -> Result<()> {
        let start = start * self.elem_len;
        let end = end * self.elem_len;

        let len = self.len * self.elem_len;
        ensure!(start < len, "start out of range {} >= {}", start, len);
        ensure!(end <= len, "end out of range {} > {}", end, len);
        ensure!(
            start <= self.data_width * self.elem_len || start >= self.cache_index_start,
            "out of bounds"
        );

        self.store_read_into(start, end, buf)
    }

    fn read_range(&self, r: ops::Range<usize>) -> Result<Vec<E>> {
        let start = r.start * self.elem_len;
        let end = r.end * self.elem_len;

        let len = self.len * self.elem_len;
        ensure!(start < len, "start out of range {} >= {}", start, len);
        ensure!(end <= len, "end out of range {} > {}", end, len);
        ensure!(
            start <= self.data_width * self.elem_len || start >= self.cache_index_start,
            "out of bounds"
        );

        Ok(self
            .store_read_range(start, end)?
            .chunks(self.elem_len)
            .map(E::from_slice)
            .collect())
    }

    fn len(&self) -> usize {
        self.len
    }

    fn loaded_from_disk(&self) -> bool {
        self.loaded_from_disk
    }

    fn compact(
        &mut self,
        _branches: usize,
        _config: StoreConfig,
        _store_version: u32,
    ) -> Result<bool> {
        bail!("Cannot compact this type of Store");
    }

    fn delete(config: StoreConfig) -> Result<()> {
        let path = StoreConfig::data_path(&config.path, &config.id);
        remove_file(&path).with_context(|| format!("Failed to delete {:?}", &path))
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn push(&mut self, el: E) -> Result<()> {
        let len = self.len;
        ensure!(
            (len + 1) * self.elem_len <= self.store_size(),
            "not enough space, len: {}, E size {}, store len {}",
            len,
            self.elem_len,
            self.store_size()
        );

        self.write_at(el, len)
    }

    fn sync(&self) -> Result<()> {
        self.file.sync_all().context("failed to sync file")
    }

    #[allow(unsafe_code)]
    fn process_layer<A: Algorithm<E>, U: Unsigned>(
        &mut self,
        width: usize,
        level: usize,
        read_start: usize,
        write_start: usize,
    ) -> Result<()> {
        // Safety: this operation is safe becase it's a limited
        // writable region on the backing store managed by this type.
        let mut mmap = unsafe {
            let mut mmap_options = MmapOptions::new();
            mmap_options
                .offset((write_start * E::byte_len()) as u64)
                .len(width * E::byte_len())
                .map_mut(&self.file)
        }?;

        let data_lock = Arc::new(RwLock::new(self));
        let branches = U::to_usize();
        let shift = log2_pow2(branches);
        let write_chunk_width = (BUILD_CHUNK_NODES >> shift) * E::byte_len();

        ensure!(BUILD_CHUNK_NODES % branches == 0, "Invalid chunk size");
        Vec::from_iter((read_start..read_start + width).step_by(BUILD_CHUNK_NODES))
            .into_par_iter()
            .zip(mmap.par_chunks_mut(write_chunk_width))
            .try_for_each(|(chunk_index, write_mmap)| -> Result<()> {
                let chunk_size = std::cmp::min(BUILD_CHUNK_NODES, read_start + width - chunk_index);

                let chunk_nodes = {
                    // Read everything taking the lock once.
                    data_lock
                        .read()
                        .unwrap()
                        .read_range_internal(chunk_index..chunk_index + chunk_size)?
                };

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
                let hashed_nodes_as_bytes_len = hashed_nodes_as_bytes.len();
                ensure!(
                    hashed_nodes_as_bytes.len() == chunk_size / branches * E::byte_len(),
                    "Invalid hashed node length"
                );

                write_mmap[0..hashed_nodes_as_bytes_len].copy_from_slice(&hashed_nodes_as_bytes);

                Ok(())
            })
    }

    // LevelCacheStore specific merkle-tree build.
    fn build<A: Algorithm<E>, U: Unsigned>(
        &mut self,
        leafs: usize,
        height: usize,
        config: Option<StoreConfig>,
    ) -> Result<E> {
        let branches = U::to_usize();
        ensure!(
            next_pow2(branches) == branches,
            "branches MUST be a power of 2"
        );
        ensure!(Store::len(self) == leafs, "Inconsistent data");
        ensure!(leafs % 2 == 0, "Leafs must be a power of two");
        ensure!(
            config.is_some(),
            "LevelCacheStore build requires a valid config"
        );

        // Process one `level` at a time of `width` nodes. Each level has half the nodes
        // as the previous one; the first level, completely stored in `data`, has `leafs`
        // nodes. We guarantee an even number of nodes per `level`, duplicating the last
        // node if necessary.
        let mut level: usize = 0;
        let mut width = leafs;
        let mut level_node_index = 0;

        let config = config.unwrap();
        let shift = log2_pow2(branches);

        // Both in terms of elements, not bytes.
        let cache_size = get_merkle_tree_cache_size(leafs, branches, config.levels)?;
        let cache_index_start = (get_merkle_tree_len(leafs, branches)?) - cache_size;

        while width > 1 {
            // Start reading at the beginning of the current level, and writing the next
            // level immediate after.  `level_node_index` keeps track of the current read
            // starts, and width is updated accordingly at each level so that we know where
            // to start writing.
            let (read_start, write_start) = if level == 0 {
                // Note that we previously asserted that data.len() == leafs.
                (0, Store::len(self))
            } else if level_node_index < cache_index_start {
                (0, width)
            } else {
                (
                    level_node_index - cache_index_start,
                    (level_node_index + width) - cache_index_start,
                )
            };

            self.process_layer::<A, U>(width, level, read_start, write_start)?;

            if level_node_index < cache_index_start {
                self.front_truncate(&config, width)?;
            }

            level_node_index += width;
            level += 1;
            width >>= shift; // width /= branches;

            // When the layer is complete, update the store length
            // since we know the backing file was updated outside of
            // the store interface.
            self.set_len(level_node_index);
        }

        // Account for the root element.
        self.set_len(Store::len(self) + 1);
        // Ensure every element is accounted for.
        ensure!(
            Store::len(self) == get_merkle_tree_len(leafs, branches)?,
            "Invalid merkle tree length"
        );

        ensure!(height == level + 1, "Invalid tree height");
        // The root isn't part of the previous loop so `height` is
        // missing one level.

        // Return the root.  Note that the offset is adjusted because
        // we've just built a store that says that it has the full
        // length of elements, when in fact only the cached portion is
        // on disk.
        self.read_at_internal(self.len() - cache_index_start - 1)
    }
}

impl<E: Element, R: Read + Send + Sync> LevelCacheStore<E, R> {
    pub fn set_len(&mut self, len: usize) {
        self.len = len;
    }

    // Remove 'len' elements from the front of the file.
    pub fn front_truncate(&mut self, config: &StoreConfig, len: usize) -> Result<()> {
        let metadata = self.file.metadata()?;
        let store_size = metadata.len();
        let len = (len * E::byte_len()) as u64;

        ensure!(store_size >= len, "Invalid truncation length");

        // Seek the reader past the length we want removed.
        let mut reader = OpenOptions::new()
            .read(true)
            .open(StoreConfig::data_path(&config.path, &config.id))?;
        reader.seek(SeekFrom::Start(len))?;

        // Make sure the store file is opened for read/write.
        self.file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(StoreConfig::data_path(&config.path, &config.id))?;

        // Seek the writer.
        self.file.seek(SeekFrom::Start(0))?;

        let written = copy(&mut reader, &mut self.file)?;
        ensure!(written == store_size - len, "Failed to copy all data");

        self.file.set_len(written)?;

        Ok(())
    }

    pub fn store_size(&self) -> usize {
        self.store_size
    }

    // 'store_range' must be the total number of elements in the store (e.g. tree.len()).
    pub fn is_consistent_v1(
        store_range: usize,
        branches: usize,
        config: &StoreConfig,
    ) -> Result<bool> {
        let data_path = StoreConfig::data_path(&config.path, &config.id);

        let file = File::open(data_path)?;
        let metadata = file.metadata()?;
        let store_size = metadata.len() as usize;

        // The LevelCacheStore base data layer must already be a
        // massaged next pow2 (guaranteed if created with
        // DiskStore::compact, which is the only supported method at
        // the moment).
        let size = get_merkle_tree_leafs(store_range, branches)?;
        ensure!(
            size == next_pow2(size),
            "Inconsistent merkle tree height detected"
        );

        // Calculate cache start and the updated size with repect to
        // the data size.
        let cache_size = get_merkle_tree_cache_size(size, branches, config.levels)? * E::byte_len();

        // Sanity checks that the StoreConfig levels matches this
        // particular on-disk file.
        Ok(store_size == size * E::byte_len() + cache_size)
    }

    // Note that v2 is now the default compaction mode, so this isn't a versioned call.
    // 'store_range' must be the total number of elements in the store (e.g. tree.len()).
    pub fn is_consistent(
        store_range: usize,
        branches: usize,
        config: &StoreConfig,
    ) -> Result<bool> {
        let data_path = StoreConfig::data_path(&config.path, &config.id);

        let file = File::open(data_path)?;
        let metadata = file.metadata()?;
        let store_size = metadata.len() as usize;

        // The LevelCacheStore base data layer must already be a
        // massaged next pow2 (guaranteed if created with
        // DiskStore::compact, which is the only supported method at
        // the moment).
        let size = get_merkle_tree_leafs(store_range, branches)?;
        ensure!(
            size == next_pow2(size),
            "Inconsistent merkle tree height detected"
        );

        // LevelCacheStore on disk file is only the cached data, so
        // the file size dictates the cache_size.  Calculate cache
        // start and the updated size with repect to the file size.
        let cache_size = get_merkle_tree_cache_size(size, branches, config.levels)? * E::byte_len();

        // Sanity checks that the StoreConfig levels matches this
        // particular on-disk file.  Since an external reader *is*
        // set, we check to make sure that the data on disk is *only*
        // the cached element data.
        Ok(store_size == cache_size)
    }

    pub fn store_read_range(&self, start: usize, end: usize) -> Result<Vec<u8>> {
        let read_len = end - start;
        let mut read_data = vec![0; read_len];
        let mut adjusted_start = start;

        ensure!(
            start <= self.data_width * self.elem_len || start >= self.cache_index_start,
            "out of bounds"
        );

        // If an external reader was specified for the base layer, use it.
        if start < self.data_width * self.elem_len && self.reader.is_some() {
            self.reader
                .as_ref()
                .unwrap()
                .read(start, end, &mut read_data)
                .with_context(|| {
                    format!(
                        "failed to read {} bytes from file at offset {}",
                        end - start,
                        start
                    )
                })?;

            return Ok(read_data);
        }

        // Adjust read index if in the cached ranged to be shifted
        // over since the data stored is compacted.
        if start >= self.cache_index_start {
            let v1 = self.reader.is_none();
            adjusted_start = if v1 {
                start - self.cache_index_start + (self.data_width * self.elem_len)
            } else {
                start - self.cache_index_start
            };
        }

        self.file
            .read_exact_at(adjusted_start as u64, &mut read_data)
            .with_context(|| {
                format!(
                    "failed to read {} bytes from file at offset {}",
                    read_len, start
                )
            })?;

        Ok(read_data)
    }

    // This read is for internal use only during the 'build' process.
    fn store_read_range_internal(&self, start: usize, end: usize) -> Result<Vec<u8>> {
        let read_len = end - start;
        let mut read_data = vec![0; read_len];

        ensure!(
            start <= self.data_width * self.elem_len || start >= self.cache_index_start,
            "out of bounds"
        );

        self.file
            .read_exact_at(start as u64, &mut read_data)
            .with_context(|| {
                format!(
                    "failed to read {} bytes from file at offset {}",
                    read_len, start
                )
            })?;

        Ok(read_data)
    }

    fn read_range_internal(&self, r: ops::Range<usize>) -> Result<Vec<E>> {
        let start = r.start * self.elem_len;
        let end = r.end * self.elem_len;

        let len = self.len * self.elem_len;
        ensure!(start < len, "start out of range {} >= {}", start, len);
        ensure!(end <= len, "end out of range {} > {}", end, len);
        ensure!(
            start <= self.data_width * self.elem_len || start >= self.cache_index_start,
            "out of bounds"
        );

        Ok(self
            .store_read_range_internal(start, end)?
            .chunks(self.elem_len)
            .map(E::from_slice)
            .collect())
    }

    fn read_at_internal(&self, index: usize) -> Result<E> {
        let start = index * self.elem_len;
        let end = start + self.elem_len;

        let len = self.len * self.elem_len;
        ensure!(start < len, "start out of range {} >= {}", start, len);
        ensure!(end <= len, "end out of range {} > {}", end, len);
        ensure!(
            start <= self.data_width * self.elem_len || start >= self.cache_index_start,
            "out of bounds"
        );

        Ok(E::from_slice(&self.store_read_range_internal(start, end)?))
    }

    pub fn store_read_into(&self, start: usize, end: usize, buf: &mut [u8]) -> Result<()> {
        ensure!(
            start <= self.data_width * self.elem_len || start >= self.cache_index_start,
            "Invalid read start"
        );

        // If an external reader was specified for the base layer, use it.
        if start < self.data_width * self.elem_len && self.reader.is_some() {
            self.reader
                .as_ref()
                .unwrap()
                .read(start, end, buf)
                .with_context(|| {
                    format!(
                        "failed to read {} bytes from file at offset {}",
                        end - start,
                        start
                    )
                })?;
        } else {
            // Adjust read index if in the cached ranged to be shifted
            // over since the data stored is compacted.
            let adjusted_start = if start >= self.cache_index_start {
                if self.reader.is_none() {
                    // if v1
                    start - self.cache_index_start + (self.data_width * self.elem_len)
                } else {
                    start - self.cache_index_start
                }
            } else {
                start
            };

            self.file
                .read_exact_at(adjusted_start as u64, buf)
                .with_context(|| {
                    format!(
                        "failed to read {} bytes from file at offset {}",
                        end - start,
                        start
                    )
                })?;
        }

        Ok(())
    }

    pub fn store_copy_from_slice(&mut self, start: usize, slice: &[u8]) -> Result<()> {
        ensure!(
            start + slice.len() <= self.store_size,
            "Requested slice too large (max: {})",
            self.store_size
        );
        self.file.write_all_at(start as u64, slice)?;

        Ok(())
    }
}
