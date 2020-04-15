use std::fs::{remove_file, File, OpenOptions};
use std::io::{copy, Seek, SeekFrom};
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
use crate::store::{Store, StoreConfig, StoreConfigDataVersion, BUILD_CHUNK_NODES};

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
        let leafs = get_merkle_tree_leafs(self.len, branches)?;
        let data_width = leafs * self.elem_len;

        // Calculate how large the cache should be (based on the
        // config.levels param).
        let cache_size =
            get_merkle_tree_cache_size(leafs, branches, config.levels)? * self.elem_len;

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
                        .read_range(chunk_index..chunk_index + chunk_size)?
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

    // DiskStore specific merkle-tree build.
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

        // Process one `level` at a time of `width` nodes. Each level has half the nodes
        // as the previous one; the first level, completely stored in `data`, has `leafs`
        // nodes. We guarantee an even number of nodes per `level`, duplicating the last
        // node if necessary.
        let mut level: usize = 0;
        let mut width = leafs;
        let mut level_node_index = 0;

        let shift = log2_pow2(branches);

        while width > 1 {
            // Start reading at the beginning of the current level, and writing the next
            // level immediate after.  `level_node_index` keeps track of the current read
            // starts, and width is updated accordingly at each level so that we know where
            // to start writing.
            let (read_start, write_start) = if level == 0 {
                // Note that we previously asserted that data.len() == leafs.
                (0, Store::len(self))
            } else {
                (level_node_index, level_node_index + width)
            };

            self.process_layer::<A, U>(width, level, read_start, write_start)?;

            level_node_index += width;
            level += 1;
            width >>= shift; // width /= branches;

            // When the layer is complete, update the store length
            // since we know the backing file was updated outside of
            // the store interface.
            self.set_len(Store::len(self) + width);
        }

        // Ensure every element is accounted for.
        ensure!(
            Store::len(self) == get_merkle_tree_len(leafs, branches)?,
            "Invalid merkle tree length"
        );

        ensure!(height == level + 1, "Invalid tree height");
        // The root isn't part of the previous loop so `height` is
        // missing one level.

        // Return the root
        self.last()
    }
}

impl<E: Element> DiskStore<E> {
    fn set_len(&mut self, len: usize) {
        self.len = len;
    }

    // 'store_range' must be the total number of elements in the store
    // (e.g. tree.len()).  Arity/branches is ignored since a
    // DiskStore's size is related only to the number of elements in
    // the tree.
    pub fn is_consistent(
        store_range: usize,
        _branches: usize,
        config: &StoreConfig,
    ) -> Result<bool> {
        let data_path = StoreConfig::data_path(&config.path, &config.id);

        let file = File::open(&data_path)?;
        let metadata = file.metadata()?;
        let store_size = metadata.len() as usize;

        Ok(store_size == store_range * E::byte_len())
    }

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
