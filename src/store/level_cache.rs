use std::fmt;
use std::fs::{remove_file, File};
use std::io::Read;
use std::marker::PhantomData;
use std::ops;
use std::path::Path;

use anyhow::{Context, Result};
use positioned_io::ReadAt;

use crate::merkle::{get_merkle_tree_cache_size, get_merkle_tree_leafs, next_pow2, Element};
use crate::store::{ExternalReader, Store, StoreConfig};

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
        let cache_size = (2 * self.data_width - 1) * self.elem_len - self.cache_index_start;
        self.reader = Some(reader);

        // If we're using an external reader, check that the data on
        // disk is only the cached elements.
        Ok(self.store_size == cache_size)
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

    pub fn store_copy_from_slice(&mut self, _start: usize, _slice: &[u8]) {
        unimplemented!("Not supported by the LevelCacheStore");
    }
}
