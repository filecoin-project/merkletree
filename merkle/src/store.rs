use failure::Error;
use merkle::{Element, next_pow2, log2_pow2};
use positioned_io::{ReadAt, WriteAt};
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::marker::PhantomData;
use std::ops::{self, Index};
use tempfile::tempfile;

pub type Result<T> = std::result::Result<T, Error>;


#[derive(Debug)]
pub struct StoreConfig {
    /// A directory in which data (a merkle tree) can be persisted.
    pub path: String,

    /// A unique identifier used to help specify the on-disk store
    /// location for this particular data.
    pub id: String,

    /// The number of merkle tree levels to cache in memory, starting
    /// at the root.  A value larger than the number of levels in the
    /// tree will cache the entire merkle tree.
    pub levels: usize,
}

impl StoreConfig {
    pub fn new(path: String, id: String, levels: usize) -> Result<StoreConfig> {
        Ok(StoreConfig {
            path,
            id,
            levels
        })
    }

    // Deterministically create the data_path on-disk location from a
    // path and specified id.
    pub fn data_path(path: &str, id: &str) -> PathBuf {
        Path::new(&path)
            .join(format!("sc-merkle_tree-{}.dat", id))
    }
}

impl Clone for StoreConfig {
    fn clone(&self) -> StoreConfig {
        StoreConfig {
            path: self.path.clone(),
            id: self.id.clone(),
            levels: self.levels
        }
    }
}


/// Backing store of the merkle tree.
pub trait Store<E: Element>:
    ops::Deref<Target = [E]> + std::fmt::Debug + Clone + Send + Sync
{
    /// Creates a new store which can store up to `size` elements.
    fn new_with_config(size: usize, config: Option<StoreConfig>) -> Result<Self>;
    fn new(size: usize) -> Result<Self>;

    fn new_from_slice_with_config(size: usize, data: &[u8], config: Option<StoreConfig>) -> Result<Self>;
    fn new_from_slice(size: usize, data: &[u8]) -> Result<Self>;

    fn new_from_disk(config: Option<StoreConfig>) -> Result<Self>;

    fn write_at(&mut self, el: E, index: usize);

    // Used to reduce lock contention and do the `E` to `u8`
    // conversion in `build` *outside* the lock.
    // `buf` is a slice of converted `E`s and `start` is its
    // position in `E` sizes (*not* in `u8`).
    fn copy_from_slice(&mut self, buf: &[u8], start: usize);

    fn read_at(&self, index: usize) -> E;
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
    fn new_with_config(size: usize, _config: Option<StoreConfig>) -> Result<Self> {
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

    fn new_from_slice_with_config(size: usize, data: &[u8], _config: Option<StoreConfig>) -> Result<Self> {
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

    fn new_from_disk(_config: Option<StoreConfig>) -> Result<Self> {
        unimplemented!("Cannot load a VecStore from disk");
    }

    fn read_at(&self, index: usize) -> E {
        self.0[index].clone()
    }

    fn read_into(&self, index: usize, buf: &mut [u8]) {
        self.0[index].copy_to_slice(buf);
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
    fn new_with_config(size: usize, config: Option<StoreConfig>) -> Result<Self> {
        if config.is_none() {
            return Self::new(size);
        }

        let store_config = config.clone().unwrap();
        let data_path = StoreConfig::data_path(
            &store_config.path, &store_config.id);

        // If the specified file exists, load it from disk.
        if Path::new(&data_path).exists() {
            return Self::new_from_disk(config);
        }

        // Otherwise, create the file and allow it to be the on-disk store.
        let data = OpenOptions::new()
            .write(true)
            .read(true)
            .create_new(true)
            .open(data_path)
            .unwrap();

        let store_size = E::byte_len() * size;
        data.set_len(store_size as u64)
            .unwrap_or_else(|_| panic!("couldn't set len {}", store_size));

        Ok(DiskStore {
            len: 0,
            elem_len: E::byte_len(),
            _e: Default::default(),
            file: data,
            store_size,
        })
    }

    #[allow(unsafe_code)]
    fn new(size: usize) -> Result<Self> {
        let store_size = E::byte_len() * size;
        let file = tempfile().expect("couldn't create temp file");
        file.set_len(store_size as u64)
            .unwrap_or_else(|_| panic!("couldn't set len of {}", store_size));

        Ok(DiskStore {
            len: 0,
            elem_len: E::byte_len(),
            _e: Default::default(),
            file,
            store_size,
        })
    }

    fn new_from_slice_with_config(size: usize, data: &[u8], config: Option<StoreConfig>) -> Result<Self> {
        if config.is_none() {
            return Self::new_from_slice(size, &data);
        }

        let mut store = Self::new_with_config(size, config).expect("Failed to create new store");
        store.store_copy_from_slice(0, data);
        store.len = data.len() / store.elem_len;

        Ok(store)
    }

    fn new_from_slice(size: usize, data: &[u8]) -> Result<Self> {
        assert_eq!(data.len() % E::byte_len(), 0);

        let mut store = Self::new(size)?;

        store.store_copy_from_slice(0, data);
        store.elem_len = E::byte_len();
        store.len = data.len() / store.elem_len;

        Ok(store)
    }

    fn new_from_disk(config: Option<StoreConfig>) -> Result<Self> {
        let config = config.unwrap();
        let data_path = StoreConfig::data_path(&config.path, &config.id);

        let data = File::open(data_path)?;
        let store_size = data.metadata().unwrap().len() as usize;
        let size = store_size / E::byte_len();

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


// Internally uses a DiskStore to manage the on-disk MT, but also has
// some levels of the tree cached in RAM for quicker read accesses.
#[derive(Debug)]
pub struct LevelCacheStore<E: Element> {
    config: StoreConfig,
    data: DiskStore<E>,
    cache: Vec<u8>,
    cache_index_start: usize,
    cache_index_end: usize,

    _e: PhantomData<E>,
}

impl<E: Element> ops::Deref for LevelCacheStore<E> {
    type Target = [E];

    fn deref(&self) -> &Self::Target {
        unimplemented!()
    }
}

// Helper method: Given a store size, the node element size and the
// number of tree levels to cache, return the cached index range
// (start, end).
fn calculate_cache_range(size: usize, elem_len: usize, num_cache_levels: usize) -> (usize, usize) {
    // Calculate the store index map based on the approximate size.
    let pow = next_pow2(size);
    let height = log2_pow2(2 * pow);
    let mut index = pow * elem_len;
    let store_size = pow * elem_len;
    let mut level_index_map = Vec::new();
    for _ in 0..height {
        level_index_map.push(store_size - index);
        index >>= 1;
    }

    // If we're told to cache more tree levels than we have, cache them all.
    let mut levels = num_cache_levels;
    if levels >= level_index_map.len() {
        levels = level_index_map.len() - 1;
    }

    let cache_index_start = level_index_map[level_index_map.len() - levels - 1];
    let cache_index_end = size * elem_len;

    (cache_index_start, cache_index_end)
}

impl<E: Element> Store<E> for LevelCacheStore<E> {
    fn new_with_config(size: usize, config: Option<StoreConfig>) -> Result<Self> {
        let data = DiskStore::new_with_config(size, config.clone())
            .expect("Failed to create merkle tree data store");

        // Calculate the cached range based on specified levels.
        let store_config = config.unwrap();
        let (cache_index_start, cache_index_end) =
            calculate_cache_range(size, E::byte_len(), store_config.levels);
        let cache = vec![0; cache_index_end - cache_index_start];

        Ok(LevelCacheStore{
            config: store_config,
            data,
            cache,
            cache_index_start,
            cache_index_end,
            _e: Default::default()
        })
    }

    fn new(_size: usize) -> Result<Self> {
        unimplemented!("This method requires StoreConfig options in addition to the total size");
    }

    fn new_from_slice_with_config(size: usize, data: &[u8], config: Option<StoreConfig>) -> Result<Self> {
        assert_eq!(data.len() % E::byte_len(), 0);

        let mut store = LevelCacheStore::new_with_config(size, config).unwrap();
        store.store_copy_from_slice(0, data);
        store.data.len = std::cmp::max(
            store.data.len, data.len() / store.data.elem_len);

        Ok(store)
    }

    fn new_from_slice(_size: usize, _data: &[u8]) -> Result<Self> {
        unimplemented!("This method requires StoreConfig options in addition to the total size");
    }

    fn new_from_disk(config: Option<StoreConfig>) -> Result<Self> {
        let store_config = config.clone().unwrap();
        let data = DiskStore::new_from_disk(config)
            .expect("Failed to load merkle tree from disk");

        // Calculate the cached range based on specified levels.
        let (cache_index_start, cache_index_end) =
            calculate_cache_range(
                data.len(), E::byte_len(), store_config.levels);
        let cache = data.store_read_range(
            cache_index_start, cache_index_end);
        assert_eq!(cache.len(), cache_index_end - cache_index_start);

        Ok(LevelCacheStore {
            config: store_config,
            data,
            cache,
            cache_index_start,
            cache_index_end,
            _e: Default::default(),
        })
    }

    fn write_at(&mut self, el: E, index: usize) {
        let start = index * self.data.elem_len;
        let end = start + self.data.elem_len;
        if start >= self.cache_index_start {
            let cache_start = start - self.cache_index_start;
            let cache_end = end - self.cache_index_start;
            assert!(cache_end <= self.cache.len());

            let segment = &mut self.cache[cache_start..cache_end];
            el.copy_to_slice(segment);
        }
        self.data.write_at(el, index)
    }

    fn copy_from_slice(&mut self, buf: &[u8], start: usize) {
        assert_eq!(buf.len() % self.data.elem_len, 0);
        self.store_copy_from_slice(start * self.data.elem_len, buf);
        self.data.len = std::cmp::max(
            self.data.len, start + buf.len() /
                self.data.elem_len);
    }

    fn read_at(&self, index: usize) -> E {
        let start = index * self.data.elem_len;
        let end = start + self.data.elem_len;
        if start >= self.cache_index_start {
            let cache_start = start - self.cache_index_start;
            let cache_end = end - self.cache_index_start;
            assert!(cache_end <= self.cache.len());

            return E::from_slice(&self.cache[cache_start..cache_end])
        }

        self.data.read_at(index)
    }

    fn read_into(&self, index: usize, buf: &mut [u8]) {
        let start = index * self.data.elem_len;
        let end = start + buf.len();
        if start >= self.cache_index_start {
            let cache_start = start - self.cache_index_start;
            let cache_end = end - self.cache_index_start;
            assert!(cache_end <= self.cache.len());

            return buf.copy_from_slice(&self.cache[cache_start..cache_end])
        }

        self.data.read_into(index, buf)
    }

    fn read_range(&self, r: ops::Range<usize>) -> Vec<E> {
        let start = r.start * self.data.elem_len;
        let end = r.end * self.data.elem_len;

        if start >= self.cache_index_start {
            let cache_start = start - self.cache_index_start;
            let cache_end = end - self.cache_index_start;
            assert!(cache_end <= self.cache.len());

            return self.cache[cache_start..cache_end]
                .chunks(E::byte_len())
                .map(E::from_slice)
                .collect();
        }

        self.data.read_range(r)
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn push(&mut self, el: E) {
        self.data.push(el);
    }

    fn sync(&self) {
        self.data.sync()
    }
}

impl<E: Element> LevelCacheStore<E> {
    pub fn store_size(&self) -> usize {
        self.data.store_size()
    }

    pub fn store_read_range(&self, start: usize, end: usize) -> Vec<u8> {
        if start >= self.cache_index_start {
            let cache_start = start - self.cache_index_start;
            let cache_end = end - self.cache_index_start;
            assert!(cache_end <= self.cache.len());

            return self.cache[cache_start..cache_end].to_vec();
        }

        self.data.store_read_range(start, end)
    }

    pub fn store_read_into(&self, start: usize, end: usize, buf: &mut [u8]) {
        if start >= self.cache_index_start {
            let cache_start = start - self.cache_index_start;
            let cache_end = end - self.cache_index_start;
            assert!(cache_end <= self.cache.len());

            return buf.copy_from_slice(&self.cache[cache_start..cache_end]);
        }

        self.data.store_read_into(start, end, buf);
    }

    pub fn store_copy_from_slice(&mut self, start: usize, slice: &[u8]) {
        if start >= self.cache_index_start {
            let end = std::cmp::min(self.cache_index_end, start + slice.len());
            let cache_start = start - self.cache_index_start;
            let cache_end = end - self.cache_index_start;
            assert!(cache_end <= self.cache.len());

            let segment = &mut self.cache[cache_start..cache_end];
            segment.copy_from_slice(&slice[0..]);
        }

        self.data.store_copy_from_slice(start, slice);
    }
}

impl<E: Element> Clone for LevelCacheStore<E> {
    fn clone(&self) -> LevelCacheStore<E> {
        unimplemented!("LevelCacheStore cloning is unsupported");
    }
}
