use std::fmt;
use std::fs::{remove_file, File, OpenOptions};
use std::io::{copy, Read, Seek, SeekFrom};
use std::marker::PhantomData;
use std::ops::{self, Index};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use positioned_io::{ReadAt, WriteAt};
use rayon::iter::plumbing::*;
use rayon::iter::*;
use serde::{Deserialize, Serialize};
use tempfile::tempfile;

use crate::merkle::{get_merkle_tree_cache_size, get_merkle_tree_leafs, next_pow2, Element};

pub const DEFAULT_CACHED_ABOVE_BASE_LAYER: usize = 7;

// Version 1 always contained the base layer data (even after 'compact').
// Version 2 no longer contains the base layer data after compact.
#[derive(Clone, Copy, Debug)]
pub enum StoreConfigDataVersion {
    One = 1,
    Two = 2,
}

const DEFAULT_STORE_CONFIG_DATA_VERSION: u32 = StoreConfigDataVersion::Two as u32;

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
    pub fn default_cached_above_base_layer(leafs: usize) -> usize {
        if leafs < 5 {
            0
        } else if leafs >> DEFAULT_CACHED_ABOVE_BASE_LAYER == 0 {
            2
        } else {
            DEFAULT_CACHED_ABOVE_BASE_LAYER
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

#[derive(Clone)]
pub struct ExternalReader<R: Read + Send + Sync> {
    pub source: R,
    pub read_fn: fn(start: usize, end: usize, buf: &mut [u8], source: &R) -> Result<usize>,
}

impl<R: Read + Send + Sync> ExternalReader<R> {
    pub fn read(&self, start: usize, end: usize, buf: &mut [u8]) -> Result<usize> {
        (self.read_fn)(start, end, buf, &self.source)
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

/// Backing store of the merkle tree.
pub trait Store<E: Element>:
    ops::Deref<Target = [E]> + std::fmt::Debug + Clone + Send + Sync
{
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
    fn compact(&mut self, branches: usize, config: StoreConfig, store_version: u32)
        -> Result<bool>;

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

    // Sync contents to disk (if it exists). This function is used to avoid
    // unnecessary flush calls at the cost of added code complexity.
    fn sync(&self) -> Result<()> {
        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
pub struct VecStore<E: Element>(Vec<E>);

impl<E: Element> ops::Deref for VecStore<E> {
    type Target = [E];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<E: Element> Store<E> for VecStore<E> {
    fn new_with_config(size: usize, _branches: usize, _config: StoreConfig) -> Result<Self> {
        Self::new(size)
    }

    fn new(size: usize) -> Result<Self> {
        Ok(VecStore(Vec::with_capacity(size)))
    }

    fn write_at(&mut self, el: E, index: usize) -> Result<()> {
        if self.0.len() <= index {
            self.0.resize(index + 1, E::default());
        }

        self.0[index] = el;
        Ok(())
    }

    // NOTE: Performance regression. To conform with the current API we are
    // unnecessarily converting to and from `&[u8]` in the `VecStore` which
    // already stores `E` (in contrast with the `mmap` versions). We are
    // prioritizing performance for the `mmap` case which will be used in
    // production (`VecStore` is mainly for testing and backwards compatibility).
    fn copy_from_slice(&mut self, buf: &[u8], start: usize) -> Result<()> {
        ensure!(
            buf.len() % E::byte_len() == 0,
            "buf size must be a multiple of {}",
            E::byte_len()
        );
        let num_elem = buf.len() / E::byte_len();

        if self.0.len() < start + num_elem {
            self.0.resize(start + num_elem, E::default());
        }

        self.0.splice(
            start..start + num_elem,
            buf.chunks_exact(E::byte_len()).map(E::from_slice),
        );
        Ok(())
    }

    fn new_from_slice_with_config(
        size: usize,
        _branches: usize,
        data: &[u8],
        _config: StoreConfig,
    ) -> Result<Self> {
        Self::new_from_slice(size, &data)
    }

    fn new_from_slice(size: usize, data: &[u8]) -> Result<Self> {
        let mut v: Vec<_> = data
            .chunks_exact(E::byte_len())
            .map(E::from_slice)
            .collect();
        let additional = size - v.len();
        v.reserve(additional);

        Ok(VecStore(v))
    }

    fn new_from_disk(_size: usize, _branches: usize, _config: &StoreConfig) -> Result<Self> {
        unimplemented!("Cannot load a VecStore from disk");
    }

    fn read_at(&self, index: usize) -> Result<E> {
        Ok(self.0[index].clone())
    }

    fn read_into(&self, index: usize, buf: &mut [u8]) -> Result<()> {
        self.0[index].copy_to_slice(buf);
        Ok(())
    }

    fn read_range_into(&self, _start: usize, _end: usize, _buf: &mut [u8]) -> Result<()> {
        unimplemented!("Not required here");
    }

    fn read_range(&self, r: ops::Range<usize>) -> Result<Vec<E>> {
        Ok(self.0.index(r).to_vec())
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn loaded_from_disk(&self) -> bool {
        false
    }

    fn compact(
        &mut self,
        _branches: usize,
        _config: StoreConfig,
        _store_version: u32,
    ) -> Result<bool> {
        self.0.shrink_to_fit();

        Ok(true)
    }

    fn delete(_config: StoreConfig) -> Result<()> {
        Ok(())
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn push(&mut self, el: E) -> Result<()> {
        self.0.push(el);
        Ok(())
    }
}

/// The Disk-only store is used to reduce memory to the minimum at the
/// cost of build time performance. Most of its I/O logic is in the
/// `store_copy_from_slice` and `store_read_range` functions.
#[derive(Debug)]
pub struct DiskStore<E: Element> {
    len: usize,
    elem_len: usize,
    _e: PhantomData<E>,
    file: File,

    // This flag is useful only immediate after instantiation, which
    // is false if the store was newly initialized and true if the
    // store was loaded from already existing on-disk data.
    loaded_from_disk: bool,

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
    fn new_with_config(size: usize, branches: usize, config: StoreConfig) -> Result<Self> {
        let data_path = StoreConfig::data_path(&config.path, &config.id);

        // If the specified file exists, load it from disk.
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
        file.set_len(store_size as u64)?;

        Ok(DiskStore {
            len: 0,
            elem_len: E::byte_len(),
            _e: Default::default(),
            file,
            loaded_from_disk: false,
            store_size,
        })
    }

    #[allow(unsafe_code)]
    fn new(size: usize) -> Result<Self> {
        let store_size = E::byte_len() * size;
        let file = tempfile()?;
        file.set_len(store_size as u64)?;

        Ok(DiskStore {
            len: 0,
            elem_len: E::byte_len(),
            _e: Default::default(),
            file,
            loaded_from_disk: false,
            store_size,
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

    fn new_from_disk(size: usize, _branches: usize, config: &StoreConfig) -> Result<Self> {
        let data_path = StoreConfig::data_path(&config.path, &config.id);

        let file = File::open(&data_path)?;
        let metadata = file.metadata()?;
        let store_size = metadata.len() as usize;

        // Sanity check.
        ensure!(
            store_size == size * E::byte_len(),
            "Invalid formatted file provided. Expected {} bytes, found {} bytes",
            size * E::byte_len(),
            store_size
        );

        Ok(DiskStore {
            len: size,
            elem_len: E::byte_len(),
            _e: Default::default(),
            file,
            loaded_from_disk: true,
            store_size,
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

        Ok(E::from_slice(&self.store_read_range(start, end)?))
    }

    fn read_into(&self, index: usize, buf: &mut [u8]) -> Result<()> {
        let start = index * self.elem_len;
        let end = start + self.elem_len;

        let len = self.len * self.elem_len;
        ensure!(start < len, "start out of range {} >= {}", start, len);
        ensure!(end <= len, "end out of range {} > {}", end, len);

        self.store_read_into(start, end, buf)
    }

    fn read_range_into(&self, start: usize, end: usize, buf: &mut [u8]) -> Result<()> {
        let start = start * self.elem_len;
        let end = end * self.elem_len;

        let len = self.len * self.elem_len;
        ensure!(start < len, "start out of range {} >= {}", start, len);
        ensure!(end <= len, "end out of range {} > {}", end, len);

        self.store_read_into(start, end, buf)
    }

    fn read_range(&self, r: ops::Range<usize>) -> Result<Vec<E>> {
        let start = r.start * self.elem_len;
        let end = r.end * self.elem_len;

        let len = self.len * self.elem_len;
        ensure!(start < len, "start out of range {} >= {}", start, len);
        ensure!(end <= len, "end out of range {} > {}", end, len);

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

    // Specifically, this method truncates an existing DiskStore and
    // formats the data in such a way that is compatible with future
    // access using LevelCacheStore::new_from_disk.
    fn compact(
        &mut self,
        branches: usize,
        config: StoreConfig,
        store_version: u32,
    ) -> Result<bool> {
        // Determine how many base layer leafs there are (and in bytes).
        let leafs = get_merkle_tree_leafs(self.len, branches);
        let data_width = leafs * self.elem_len;

        // Calculate how large the cache should be (based on the
        // config.levels param).
        let cache_size = get_merkle_tree_cache_size(leafs, branches, config.levels) * self.elem_len;

        // The file cannot be compacted if the specified configuration
        // requires either 1) nothing to be cached, or 2) everything
        // to be cached.  For #1, create a data store of leafs and do
        // not use that store as backing for the MT.  For #2, avoid
        // calling this method.  To resolve, provide a sane
        // configuration.
        ensure!(
            cache_size < self.len * self.elem_len && cache_size != 0,
            "Cannot compact with this configuration"
        );

        let v1 = store_version == StoreConfigDataVersion::One as u32;
        let start: u64 = if v1 { data_width as u64 } else { 0 };

        // Calculate cache start and updated size with repect to the
        // data size.
        let cache_start = self.store_size - cache_size;

        // Seek the reader to the start of the cached data.
        let mut reader = OpenOptions::new()
            .read(true)
            .open(StoreConfig::data_path(&config.path, &config.id))?;
        reader.seek(SeekFrom::Start(cache_start as u64))?;

        // Make sure the store file is opened for read/write.
        self.file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(StoreConfig::data_path(&config.path, &config.id))?;

        // Seek the writer.
        self.file.seek(SeekFrom::Start(start))?;

        // Copy the data from the cached region to the writer.
        let written = copy(&mut reader, &mut self.file)?;
        ensure!(written == cache_size as u64, "Failed to copy all data");
        if v1 {
            // Truncate the data on-disk to be the base layer data
            // followed by the cached data.
            self.file.set_len((data_width + cache_size) as u64)?;
            // Adjust our length for internal consistency.
            self.len = (data_width + cache_size) / self.elem_len;
        } else {
            // Truncate the data on-disk to be only the cached data.
            self.file.set_len(cache_size as u64)?;

            // Adjust our length to be the cached elements only for
            // internal consistency.
            self.len = cache_size / self.elem_len;
        }

        // Sync and sanity check that we match on disk (this can be
        // removed if needed).
        self.sync()?;
        let metadata = self.file.metadata()?;
        let store_size = metadata.len() as usize;
        ensure!(
            self.len * self.elem_len == store_size,
            "Inconsistent metadata detected"
        );

        Ok(true)
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
}

impl<E: Element> DiskStore<E> {
    pub fn store_size(&self) -> usize {
        self.store_size
    }

    pub fn store_read_range(&self, start: usize, end: usize) -> Result<Vec<u8>> {
        let read_len = end - start;
        let mut read_data = vec![0; read_len];

        self.file
            .read_exact_at(start as u64, &mut read_data)
            .with_context(|| {
                format!(
                    "failed to read {} bytes from file at offset {}",
                    read_len, start
                )
            })?;

        ensure!(read_data.len() == read_len, "Failed to read the full range");

        Ok(read_data)
    }

    pub fn store_read_into(&self, start: usize, end: usize, buf: &mut [u8]) -> Result<()> {
        self.file
            .read_exact_at(start as u64, buf)
            .with_context(|| {
                format!(
                    "failed to read {} bytes from file at offset {}",
                    end - start,
                    start
                )
            })?;

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

// FIXME: Fake `Clone` implementation to accommodate the artificial call in
//  `from_data_with_store`, we won't actually duplicate the mmap memory,
//  just recreate the same object (as the original will be dropped).
impl<E: Element> Clone for DiskStore<E> {
    fn clone(&self) -> DiskStore<E> {
        unimplemented!("We can't clone a store with an already associated file");
    }
}

/// The LevelCacheStore is used to reduce the on-disk footprint even
/// further to the minimum at the cost of build time performance.
/// Each LevelCacheStore is created with a StoreConfig object which
/// contains the number of binary tree levels above the base that are
/// 'cached'.  This implementation has hard requirements about the on
/// disk file size based on that number of levels, so on-disk files
/// are tied, structurally to the configuration they were built with
/// and can only be accessed with the same number of levels.
///
/// NOTE: Unlike other store types, writes of any kind are not
/// supported (except deletion) since we're accessing specially
/// crafted on-disk data that requires a particular access pattern
/// dictated at the time of creation/compaction.
pub struct LevelCacheStore<E: Element, R: Read + Send + Sync> {
    len: usize,
    elem_len: usize,
    file: File,

    // The number of base layer data items.
    data_width: usize,

    // The byte index of where the cached data begins.
    cache_index_start: usize,

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
        let size = get_merkle_tree_leafs(store_range, branches);
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
        let cache_size = get_merkle_tree_cache_size(size, branches, config.levels) * E::byte_len();
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
            reader: Some(reader),
            _e: Default::default(),
        })
    }

    pub fn set_external_reader(&mut self, reader: ExternalReader<R>) -> Result<bool> {
        // "new_from_disk_with_reader" is the recommended way to set
        // an ExternalReader, as it does a sanity check on the cache
        // size.
        self.reader = Some(reader);

        Ok(true)
    }
}

impl<E: Element, R: Read + Send + Sync> ops::Deref for LevelCacheStore<E, R> {
    type Target = [E];

    fn deref(&self) -> &Self::Target {
        unimplemented!()
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

        bail!("Cannot create a LevelCacheStore in this way. Try DiskStore::compact");
    }

    fn new(_size: usize) -> Result<Self> {
        unimplemented!("LevelCacheStore requires a StoreConfig");
    }

    fn new_from_slice_with_config(
        _size: usize,
        _branches: usize,
        _data: &[u8],
        _config: StoreConfig,
    ) -> Result<Self> {
        unimplemented!("Cannot create a LevelCacheStore in this way. Try 'new_from_disk'.");
    }

    fn new_from_slice(_size: usize, _data: &[u8]) -> Result<Self> {
        unimplemented!("LevelCacheStore requires a StoreConfig");
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
        let size = get_merkle_tree_leafs(store_range, branches);
        ensure!(
            size == next_pow2(size),
            "Inconsistent merkle tree height detected"
        );

        // Values below in bytes.
        // Convert store_range from an element count to bytes.
        let store_range = store_range * E::byte_len();

        // Calculate cache start and the updated size with repect to
        // the data size.
        let cache_size = get_merkle_tree_cache_size(size, branches, config.levels) * E::byte_len();
        let cache_index_start = store_range - cache_size;

        // Sanity checks that the StoreConfig levels matches this
        // particular on-disk file.
        ensure!(
            store_size == size * E::byte_len() + cache_size,
            "Inconsistent store size detected"
        );

        Ok(LevelCacheStore {
            len: store_range / E::byte_len(),
            elem_len: E::byte_len(),
            file,
            data_width: size,
            cache_index_start,
            store_size,
            reader: None,
            _e: Default::default(),
        })
    }

    fn write_at(&mut self, _el: E, _index: usize) -> Result<()> {
        unimplemented!("Not supported by the LevelCacheStore");
    }

    fn copy_from_slice(&mut self, _buf: &[u8], _start: usize) -> Result<()> {
        unimplemented!("Not supported by the LevelCacheStore");
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
        true
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
}

impl<E: Element, R: Read + Send + Sync> LevelCacheStore<E, R> {
    pub fn store_size(&self) -> usize {
        self.store_size
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

    pub fn store_read_into(&self, start: usize, end: usize, buf: &mut [u8]) -> Result<()> {
        assert!(start <= self.data_width * self.elem_len || start >= self.cache_index_start);

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

    pub fn store_copy_from_slice(&mut self, _start: usize, _slice: &[u8]) {
        unimplemented!("Not supported by the LevelCacheStore");
    }
}

// FIXME: Fake `Clone` implementation to accommodate the artificial call in
//  `from_data_with_store`, we won't actually duplicate the mmap memory,
//  just recreate the same object (as the original will be dropped).
impl<E: Element, R: Read + Send + Sync> Clone for LevelCacheStore<E, R> {
    fn clone(&self) -> LevelCacheStore<E, R> {
        unimplemented!("We can't clone a store with an already associated file");
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
