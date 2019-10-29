use failure::{Error, err_msg};
use merkle::{Element, next_pow2};
use positioned_io::{ReadAt, WriteAt};
use serde::{Serialize, Deserialize};
use std::fs::{File, OpenOptions};
use std::io::{copy, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::marker::PhantomData;
use std::ops::{self, Index};
use tempfile::tempfile;

pub type Result<T> = std::result::Result<T, Error>;

const STORE_CONFIG_DATA_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct StoreConfig {
    /// A directory in which data (a merkle tree) can be persisted.
    pub path: PathBuf,

    /// A unique identifier used to help specify the on-disk store
    /// location for this particular data.
    pub id: String,

    /// The number of merkle tree levels above the base to cache on disk.
    pub levels: usize,
}

impl StoreConfig {
    pub fn new(path: String, id: String, levels: usize) -> StoreConfig {
        StoreConfig {
            path: PathBuf::from(path),
            id,
            levels
        }
    }

    // Deterministically create the data_path on-disk location from a
    // path and specified id.
    pub fn data_path(path: &PathBuf, id: &str) -> PathBuf {
        Path::new(&path).join(
            format!("sc-{:0>2}-data-{}.dat", STORE_CONFIG_DATA_VERSION, id))
    }
}


/// Backing store of the merkle tree.
pub trait Store<E: Element>:
    ops::Deref<Target = [E]> + std::fmt::Debug + Clone + Send + Sync
{
    /// Creates a new store which can store up to `size` elements.
    fn new_with_config(size: usize, config: StoreConfig) -> Result<Self>;
    fn new(size: usize) -> Result<Self>;

    fn new_from_slice_with_config(size: usize, data: &[u8], config: StoreConfig) -> Result<Self>;
    fn new_from_slice(size: usize, data: &[u8]) -> Result<Self>;

    fn new_from_disk(size: usize, config: StoreConfig) -> Result<Self>;

    fn write_at(&mut self, el: E, index: usize);

    // Used to reduce lock contention and do the `E` to `u8`
    // conversion in `build` *outside* the lock.
    // `buf` is a slice of converted `E`s and `start` is its
    // position in `E` sizes (*not* in `u8`).
    fn copy_from_slice(&mut self, buf: &[u8], start: usize);
    fn compact(&mut self, config: StoreConfig) -> Result<bool>;

    fn read_at(&self, index: usize) -> E;
    fn read_range(&self, r: ops::Range<usize>) -> Vec<E>;
    fn read_into(&self, pos: usize, buf: &mut [u8]);
    fn read_range_into(&self, start: usize, end: usize, buf: &mut [u8]);

    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn push(&mut self, el: E);

    // Sync contents to disk (if it exists). This function is used to avoid
    // unnecessary flush calls at the cost of added code complexity.
    fn sync(&self) {}
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
    fn new_with_config(size: usize, _config: StoreConfig) -> Result<Self> {
        Self::new(size)
    }

    fn new(size: usize) -> Result<Self> {
        Ok(VecStore(Vec::with_capacity(size)))
    }

    fn write_at(&mut self, el: E, index: usize) {
        if self.0.len() <= index {
            self.0.resize(index + 1, E::default());
        }

        self.0[index] = el;
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

    fn new_from_slice_with_config(size: usize, data: &[u8], _config: StoreConfig) -> Result<Self> {
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

    fn new_from_disk(_size: usize, _config: StoreConfig) -> Result<Self> {
        unimplemented!("Cannot load a VecStore from disk");
    }

    fn read_at(&self, index: usize) -> E {
        self.0[index].clone()
    }

    fn read_into(&self, index: usize, buf: &mut [u8]) {
        self.0[index].copy_to_slice(buf);
    }

    fn read_range_into(&self, _start: usize, _end: usize, _buf: &mut [u8]) {
        unimplemented!("Not required here");
    }

    fn read_range(&self, r: ops::Range<usize>) -> Vec<E> {
        self.0.index(r).to_vec()
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn compact(&mut self, _config: StoreConfig) -> Result<bool> {
        self.0.shrink_to_fit();

        Ok(true)
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn push(&mut self, el: E) {
        self.0.push(el);
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
    fn new_with_config(size: usize, config: StoreConfig) -> Result<Self> {
        let data_path = StoreConfig::data_path(
            &config.path, &config.id);

        // If the specified file exists, load it from disk.
        if Path::new(&data_path).exists() {
            return Self::new_from_disk(size, config);
        }

        // Otherwise, create the file and allow it to be the on-disk store.
        let data = OpenOptions::new()
            .write(true)
            .read(true)
            .create_new(true)
            .open(data_path)?;

        let base_size = E::byte_len() * size;
        data.set_len(base_size as u64)?;

        Ok(DiskStore {
            len: 0,
            elem_len: E::byte_len(),
            _e: Default::default(),
            file: data,
            store_size: base_size,
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
            store_size,
        })
    }

    fn new_from_slice_with_config(size: usize, data: &[u8], config: StoreConfig) -> Result<Self> {
        let mut store = Self::new_with_config(size, config)?;
        store.store_copy_from_slice(0, data);
        store.len = data.len() / store.elem_len;

        Ok(store)
    }

    fn new_from_slice(size: usize, data: &[u8]) -> Result<Self> {
        assert_eq!(data.len() % E::byte_len(), 0);

        let mut store = Self::new(size)?;
        store.store_copy_from_slice(0, data);
        store.len = data.len() / store.elem_len;

        Ok(store)
    }

    fn new_from_disk(size: usize, config: StoreConfig) -> Result<Self> {
        let data_path = StoreConfig::data_path(&config.path, &config.id);

        let data = File::open(data_path)?;
        let metadata = data.metadata()?;
        let store_size = metadata.len() as usize;

        // Sanity check.
        assert_eq!(store_size, size * E::byte_len());

        Ok(DiskStore {
            len: size,
            elem_len: E::byte_len(),
            _e: Default::default(),
            file: data,
            store_size,
        })
    }

    fn write_at(&mut self, el: E, index: usize) {
        self.store_copy_from_slice(index * self.elem_len, el.as_ref());
        self.len = std::cmp::max(self.len, index + 1);
    }

    fn copy_from_slice(&mut self, buf: &[u8], start: usize) {
        assert_eq!(buf.len() % self.elem_len, 0);
        self.store_copy_from_slice(start * self.elem_len, buf);
        self.len = std::cmp::max(self.len, start + buf.len() / self.elem_len);
    }

    fn read_at(&self, index: usize) -> E {
        let start = index * self.elem_len;
        let end = start + self.elem_len;

        let len = self.len * self.elem_len;
        assert!(start < len, "start out of range {} >= {}", start, len);
        assert!(end <= len, "end out of range {} > {}", end, len);

        E::from_slice(&self.store_read_range(start, end))
    }

    fn read_into(&self, index: usize, buf: &mut [u8]) {
        let start = index * self.elem_len;
        let end = start + self.elem_len;

        let len = self.len * self.elem_len;
        assert!(start < len, "start out of range {} >= {}", start, len);
        assert!(end <= len, "end out of range {} > {}", end, len);

        self.store_read_into(start, end, buf);
    }

    fn read_range_into(&self, start: usize, end: usize, buf: &mut [u8]) {
        let start = start * self.elem_len;
        let end = end * self.elem_len;

        let len = self.len * self.elem_len;
        assert!(start < len, "start out of range {} >= {}", start, len);
        assert!(end <= len, "end out of range {} > {}", end, len);

        self.store_read_into(start, end, buf);
    }

    fn read_range(&self, r: ops::Range<usize>) -> Vec<E> {
        let start = r.start * self.elem_len;
        let end = r.end * self.elem_len;

        let len = self.len * self.elem_len;
        assert!(start < len, "start out of range {} >= {}", start, len);
        assert!(end <= len, "end out of range {} > {}", end, len);

        self.store_read_range(start, end)
            .chunks(self.elem_len)
            .map(E::from_slice)
            .collect()
    }

    fn len(&self) -> usize {
        self.len
    }

    // Specifically, this method truncates an existing DiskStore and
    // formats the data in such a way that is compatible with future
    // access using LevelCacheStore::new_from_disk.
    fn compact(&mut self, config: StoreConfig) -> Result<bool> {
        // Determine how many leafs there are (in bytes).
        let data_width = (self.len / 2 + 1) * self.elem_len;

        // Calculate how large the cache should be (based on the
        // config.levels param).
        let mut cache_size = (2 * data_width) >> config.levels;
        if cache_size >= 2 * data_width - 1 {
            // The file cannot be compacted (to fix, provide a sane
            // configuration).
            return Err(err_msg("Cannot compact with this configuration"));
        }

        // Calculate cache start and updated size with repect to the
        // data size.
        let cache_start = std::cmp::max(self.store_size - cache_size, data_width);
        cache_size = self.store_size - cache_start;

        // Seek the reader to the start of the cached data.
        let mut reader = OpenOptions::new()
            .read(true)
            .open(StoreConfig::data_path(&config.path, &config.id))?;
        reader.seek(SeekFrom::Start(cache_start as u64))?;

        // Seek the writer to the end of the base layer data.
        self.file.seek(SeekFrom::Start(data_width as u64))?;

        // Copy the data from the cached region to just after the base
        // layer data.
        let written = copy(&mut reader, &mut self.file)?;
        assert_eq!(written, cache_size as u64);

        // Truncate the data on-disk just after the base layer data
        // and cached data that should be persisted.
        self.file.set_len((data_width + cache_size) as u64)?;

        // Adjust our length to be data_width + cached_layers for
        // internal consistency.
        self.len = (data_width + cache_size) / self.elem_len;

        // Sync and sanity check that we match on disk (this can be
        // removed if needed).
        self.sync();
        let metadata = self.file.metadata()?;
        let store_size = metadata.len() as usize;
        assert_eq!(self.len * self.elem_len, store_size);

        Ok(true)
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn push(&mut self, el: E) {
        let len = self.len;
        assert!(
            (len + 1) * self.elem_len <= self.store_size(),
            format!(
                "not enough space, len: {}, E size {}, store len {}",
                len,
                self.elem_len,
                self.store_size()
            )
        );

        self.write_at(el, len);
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
        assert_eq!(
            self.file
                .read_at(start as u64, buf)
                .unwrap_or_else(|_| panic!(
                    "failed to read {} bytes from file at offset {}",
                    end - start, start
                )),
            end - start
        );
    }

    pub fn store_copy_from_slice(&mut self, start: usize, slice: &[u8]) {
        assert!(start + slice.len() <= self.store_size);
        self.file
            .write_at(start as u64, slice)
            .expect("failed to write file");
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
/// supported since we're accessing specially crafted on-disk data
/// that requires a particular access pattern dictated at the time of
/// creation/compaction.
#[derive(Debug)]
pub struct LevelCacheStore<E: Element> {
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

    _e: PhantomData<E>
}

impl<E: Element> ops::Deref for LevelCacheStore<E> {
    type Target = [E];

    fn deref(&self) -> &Self::Target {
        unimplemented!()
    }
}

impl<E: Element> Store<E> for LevelCacheStore<E> {
    fn new_with_config(size: usize, config: StoreConfig) -> Result<Self> {
        let data_path = StoreConfig::data_path(
            &config.path, &config.id);

        // If the specified file exists, load it from disk.  This is
        // the only supported usage of this call for this type of
        // Store.
        if Path::new(&data_path).exists() {
            return Self::new_from_disk(size, config);
        }

        Err(err_msg("Cannot create a LevelCacheStore in this way. Try DiskStore::compact"))
    }

    fn new(_size: usize) -> Result<Self> {
        unimplemented!("LevelCacheStore requires a StoreConfig");
    }

    fn new_from_slice_with_config(_size: usize, _data: &[u8], _config: StoreConfig) -> Result<Self> {
        unimplemented!("Cannot create a LevelCacheStore in this way. Try 'new_from_disk'.");
    }

    fn new_from_slice(_size: usize, _data: &[u8]) -> Result<Self> {
        unimplemented!("LevelCacheStore requires a StoreConfig");
    }

    fn new_from_disk(size: usize, config: StoreConfig) -> Result<Self> {
        let data_path = StoreConfig::data_path(&config.path, &config.id);

        let data = File::open(data_path)?;
        let metadata = data.metadata()?;
        let store_size = metadata.len() as usize;

        // The LevelCacheStore base data layer must already be a
        // massaged next pow2 (guaranteed if created with
        // DiskStore::compact, which is the only supported method at
        // the moment).
        assert_eq!(size, next_pow2(size));

        // Values below in bytes.
        let data_len = size * E::byte_len();
        let store_range = 2 * data_len - 1;

        // Calculate cache start and the updated size with repect to
        // the data size.
        let mut cache_size = (2 * data_len) >> config.levels;
        let cache_start = std::cmp::max(store_size - cache_size, data_len);
        cache_size = store_size - cache_start;
        let cache_index_start = store_range - cache_size;

        // Sanity checks that the StoreConfig levels matches this
        // particular on-disk file.
        assert_eq!(store_size, data_len + cache_size);

        Ok(LevelCacheStore {
            len: store_range / E::byte_len(),
            elem_len: E::byte_len(),
            file: data,
            data_width: size,
            cache_index_start,
            store_size,
            _e: Default::default()
        })
    }

    fn write_at(&mut self, _el: E, _index: usize) {
        unimplemented!("Not supported by the LevelCacheStore");
    }

    fn copy_from_slice(&mut self, _buf: &[u8], _start: usize) {
        unimplemented!("Not supported by the LevelCacheStore");
    }

    fn read_at(&self, index: usize) -> E {
        let start = index * self.elem_len;
        let end = start + self.elem_len;

        let len = self.len * self.elem_len;
        assert!(start < len, "start out of range {} >= {}", start, len);
        assert!(end <= len, "end out of range {} > {}", end, len);
        assert!(start <= self.data_width * self.elem_len ||
                start >= self.cache_index_start);

        E::from_slice(&self.store_read_range(start, end))
    }

    fn read_into(&self, index: usize, buf: &mut [u8]) {
        let start = index * self.elem_len;
        let end = start + self.elem_len;

        let len = self.len * self.elem_len;
        assert!(start < len, "start out of range {} >= {}", start, len);
        assert!(end <= len, "end out of range {} > {}", end, len);
        assert!(start <= self.data_width * self.elem_len ||
                start >= self.cache_index_start);

        self.store_read_into(start, end, buf);
    }

    fn read_range_into(&self, start: usize, end: usize, buf: &mut [u8]) {
        let start = start * self.elem_len;
        let end = end * self.elem_len;

        let len = self.len * self.elem_len;
        assert!(start < len, "start out of range {} >= {}", start, len);
        assert!(end <= len, "end out of range {} > {}", end, len);
        assert!(start <= self.data_width * self.elem_len ||
                start >= self.cache_index_start);

        self.store_read_into(start, end, buf);
    }

    fn read_range(&self, r: ops::Range<usize>) -> Vec<E> {
        let start = r.start * self.elem_len;
        let end = r.end * self.elem_len;

        let len = self.len * self.elem_len;
        assert!(start < len, "start out of range {} >= {}", start, len);
        assert!(end <= len, "end out of range {} > {}", end, len);
        assert!(start <= self.data_width * self.elem_len ||
                start >= self.cache_index_start);

        self.store_read_range(start, end)
            .chunks(self.elem_len)
            .map(E::from_slice)
            .collect()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn compact(&mut self, _config: StoreConfig) -> Result<bool> {
        Err(err_msg("Cannot compact this type of Store"))
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn push(&mut self, el: E) {
        let len = self.len;
        assert!(
            (len + 1) * self.elem_len <= self.store_size(),
            format!(
                "not enough space, len: {}, E size {}, store len {}",
                len,
                self.elem_len,
                self.store_size()
            )
        );

        self.write_at(el, len);
    }

    fn sync(&self) {
        self.file.sync_all().expect("failed to sync file");
    }
}

impl<E: Element> LevelCacheStore<E> {
    pub fn store_size(&self) -> usize {
        self.store_size
    }

    pub fn store_read_range(&self, start: usize, end: usize) -> Vec<u8> {
        let read_len = end - start;
        let mut read_data = vec![0; read_len];
        let mut adjusted_start = start;

        assert!(start <= self.data_width * self.elem_len ||
                start >= self.cache_index_start);

        // Adjust read index if in the cached ranged to be shifted
        // over since the data stored is compacted.
        if start >= self.cache_index_start {
            adjusted_start = start - self.cache_index_start +
                (self.data_width * self.elem_len) + E::byte_len() - 1;
        }

        assert_eq!(
            self.file
                .read_at(adjusted_start as u64, &mut read_data)
                .unwrap_or_else(|_| panic!(
                    "failed to read {} bytes from file at offset {}",
                    read_len, start
                )),
            read_len
        );

        read_data
    }

    pub fn store_read_into(&self, start: usize, end: usize, buf: &mut [u8]) {
        assert_eq!(
            self.file
                .read_at(start as u64, buf)
                .unwrap_or_else(|_| panic!(
                    "failed to read {} bytes from file at offset {}",
                    end - start, start
                )),
            end - start
        );
    }

    pub fn store_copy_from_slice(&mut self, _start: usize, _slice: &[u8]) {
        unimplemented!("Not supported by the LevelCacheStore");
    }
}

// FIXME: Fake `Clone` implementation to accommodate the artificial call in
//  `from_data_with_store`, we won't actually duplicate the mmap memory,
//  just recreate the same object (as the original will be dropped).
impl<E: Element> Clone for LevelCacheStore<E> {
    fn clone(&self) -> LevelCacheStore<E> {
        unimplemented!("We can't clone a store with an already associated file");
    }
}
