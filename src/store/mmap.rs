use std::fs::{File, OpenOptions};
use std::marker::PhantomData;
use std::ops;
use std::path::{Path, PathBuf};

use anyhow::Result;
use memmap::MmapMut;

use crate::merkle::Element;
use crate::store::{Store, StoreConfig};

/// Store that saves the data on disk, and accesses it using memmap.
#[derive(Debug)]
pub struct MmapStore<E: Element> {
    path: PathBuf,
    map: Option<MmapMut>,
    file: File,
    len: usize,
    store_size: usize,
    _e: PhantomData<E>,
}

impl<E: Element> ops::Deref for MmapStore<E> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.map.as_ref().unwrap()[..]
    }
}

impl<E: Element> Store<E> for MmapStore<E> {
    #[allow(unsafe_code)]
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
            .open(&data_path)?;

        let store_size = E::byte_len() * size;
        file.set_len(store_size as u64)?;

        let map = unsafe { MmapMut::map_mut(&file)? };

        Ok(MmapStore {
            path: data_path,
            map: Some(map),
            file,
            len: 0,
            store_size,
            _e: Default::default(),
        })
    }

    #[allow(unsafe_code)]
    fn new(size: usize) -> Result<Self> {
        let store_size = E::byte_len() * size;

        let file = tempfile::NamedTempFile::new()?;
        file.as_file().set_len(store_size as u64)?;
        let (file, path) = file.into_parts();
        let map = unsafe { MmapMut::map_mut(&file)? };

        Ok(MmapStore {
            path: path.keep()?,
            map: Some(map),
            file,
            len: 0,
            store_size,
            _e: Default::default(),
        })
    }

    #[allow(unsafe_code)]
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

        let map = unsafe { MmapMut::map_mut(&file)? };

        Ok(MmapStore {
            path: data_path,
            map: Some(map),
            file,
            len: size,
            store_size,
            _e: Default::default(),
        })
    }

    fn write_at(&mut self, el: E, index: usize) -> Result<()> {
        let start = index * E::byte_len();
        let end = start + E::byte_len();

        if self.map.is_none() {
            self.reinit()?;
        }

        self.map.as_mut().unwrap()[start..end].copy_from_slice(el.as_ref());
        self.len = std::cmp::max(self.len, index + 1);

        Ok(())
    }

    fn copy_from_slice(&mut self, buf: &[u8], start: usize) -> Result<()> {
        ensure!(
            buf.len() % E::byte_len() == 0,
            "buf size must be a multiple of {}",
            E::byte_len()
        );

        let map_start = start * E::byte_len();
        let map_end = map_start + buf.len();

        if self.map.is_none() {
            self.reinit()?;
        }

        self.map.as_mut().unwrap()[map_start..map_end].copy_from_slice(buf);
        self.len = std::cmp::max(self.len, start + (buf.len() / E::byte_len()));

        Ok(())
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
        if !store.loaded_from_disk() {
            if store.map.is_none() {
                store.reinit()?;
            }

            let len = data.len();

            store.map.as_mut().unwrap()[0..len].copy_from_slice(data);
            store.len = len / E::byte_len();
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
        ensure!(store.map.is_some(), "Internal map needs to be initialized");

        let len = data.len();
        store.map.as_mut().unwrap()[0..len].copy_from_slice(data);
        store.len = len / E::byte_len();

        Ok(store)
    }

    fn read_at(&self, index: usize) -> Result<E> {
        ensure!(self.map.is_some(), "Internal map needs to be initialized");

        let start = index * E::byte_len();
        let end = start + E::byte_len();
        let len = self.len * E::byte_len();

        ensure!(start < len, "start out of range {} >= {}", start, len);
        ensure!(end <= len, "end out of range {} > {}", end, len);

        Ok(E::from_slice(&self.map.as_ref().unwrap()[start..end]))
    }

    fn read_into(&self, index: usize, buf: &mut [u8]) -> Result<()> {
        ensure!(self.map.is_some(), "Internal map needs to be initialized");

        let start = index * E::byte_len();
        let end = start + E::byte_len();
        let len = self.len * E::byte_len();

        ensure!(start < len, "start out of range {} >= {}", start, len);
        ensure!(end <= len, "end out of range {} > {}", end, len);

        buf.copy_from_slice(&self.map.as_ref().unwrap()[start..end]);

        Ok(())
    }

    fn read_range_into(&self, _start: usize, _end: usize, _buf: &mut [u8]) -> Result<()> {
        unimplemented!("Not required here");
    }

    fn read_range(&self, r: ops::Range<usize>) -> Result<Vec<E>> {
        ensure!(self.map.is_some(), "Internal map needs to be initialized");

        let start = r.start * E::byte_len();
        let end = r.end * E::byte_len();
        let len = self.len * E::byte_len();

        ensure!(start < len, "start out of range {} >= {}", start, len);
        ensure!(end <= len, "end out of range {} > {}", end, len);

        Ok(self.map.as_ref().unwrap()[start..end]
            .chunks(E::byte_len())
            .map(E::from_slice)
            .collect())
    }

    fn len(&self) -> usize {
        self.len
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
        let map = self.map.take();

        Ok(map.is_some())
    }

    #[allow(unsafe_code)]
    fn reinit(&mut self) -> Result<()> {
        self.map = unsafe { Some(MmapMut::map_mut(&self.file)?) };
        ensure!(self.map.is_some(), "Re-init mapping failed");

        Ok(())
    }

    fn delete(_config: StoreConfig) -> Result<()> {
        Ok(())
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn push(&mut self, el: E) -> Result<()> {
        let l = self.len;

        if self.map.is_none() {
            self.reinit()?;
        }

        ensure!(
            (l + 1) * E::byte_len() <= self.map.as_ref().unwrap().len(),
            "not enough space"
        );

        self.write_at(el, l)
    }
}
