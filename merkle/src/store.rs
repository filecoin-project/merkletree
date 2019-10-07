use merkle::Element;
use memmap::MmapMut;
use memmap::MmapOptions;
use positioned_io::{ReadAt, WriteAt};
use std::fs::File;
use std::marker::PhantomData;
use std::ops::{self, Index};
use tempfile::tempfile;


/// Backing store of the merkle tree.
pub trait Store<E: Element>:
    ops::Deref<Target = [E]> + std::fmt::Debug + Clone + Send + Sync
{
    /// Creates a new store which can store up to `size` elements.
    fn new(size: usize) -> Result<Self, &'static str>;

    fn new_from_slice(size: usize, data: &[u8]) -> Result<Self, &'static str>;

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
    fn new(size: usize) -> Result<Self, &'static str> {
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

    fn new_from_slice(size: usize, data: &[u8]) -> Result<Self, &'static str> {
        let mut v: Vec<_> = data
            .chunks_exact(E::byte_len())
            .map(E::from_slice)
            .collect();
        let additional = size - v.len();
        v.reserve(additional);

        Ok(VecStore(v))
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

#[derive(Debug)]
pub struct MmapStore<E: Element> {
    store: MmapMut,
    len: usize,
    elem_len: usize,
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
    fn new(size: usize) -> Result<Self, &'static str> {
        let byte_len = E::byte_len();

        let mapped = MmapOptions::new()
            .len(byte_len * size)
            .map_anon()
            .unwrap();

        Ok(MmapStore {
            store: mapped,
            len: 0,
            elem_len: byte_len,
            _e: Default::default(),
        })
    }

    fn new_from_slice(size: usize, data: &[u8]) -> Result<Self, &'static str> {
        assert_eq!(data.len() % E::byte_len(), 0);

        let mut res = Self::new(size)?;
        assert_eq!(E::byte_len(), res.elem_len);

        let end = data.len();
        res.store[..end].copy_from_slice(data);
        res.len = data.len() / res.elem_len;

        Ok(res)
    }

    // Writing at positions `i` will mark all other positions as
    // occupied with respect to the `len()` so the new `len()`
    // is `>= i`.
    fn write_at(&mut self, el: E, index: usize) {
        self.store[index * self.elem_len..(index + 1) * self.elem_len].copy_from_slice(el.as_ref());
        self.len = std::cmp::max(self.len, index + 1);
    }

    fn copy_from_slice(&mut self, buf: &[u8], start: usize) {
        assert_eq!(buf.len() % self.elem_len, 0);
        self.store[start * self.elem_len..start * self.elem_len + buf.len()].copy_from_slice(buf);
        self.len = std::cmp::max(self.len, start + buf.len() / self.elem_len);
    }

    fn read_at(&self, index: usize) -> E {
        let start = index * self.elem_len;
        let end = start + self.elem_len;
        let len = self.len * self.elem_len;
        assert!(start < len, "start out of range {} >= {}", start, len);
        assert!(end <= len, "end out of range {} > {}", end, len);

        E::from_slice(&self.store[start..end])
    }

    fn read_into(&self, index: usize, buf: &mut [u8]) {
        let start = index * self.elem_len;
        let end = start + self.elem_len;
        let len = self.len * self.elem_len;
        assert!(start < len, "start out of range {} >= {}", start, len);
        assert!(end <= len, "end out of range {} > {}", end, len);

        buf.copy_from_slice(&self.store[start..end]);
    }

    fn read_range(&self, r: ops::Range<usize>) -> Vec<E> {
        let start = r.start * self.elem_len;
        let end = r.end * self.elem_len;
        let len = self.len * self.elem_len;
        assert!(start < len, "start out of range {} >= {}", start, len);
        assert!(end <= len, "end out of range {} > {}", end, len);

        self.store[start..end]
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
        let l = self.len;
        assert!(
            (l + 1) * self.elem_len <= self.store.len(),
            "not enough space"
        );

        self.write_at(el, l);
    }
}

impl<E: Element> Clone for MmapStore<E> {
    fn clone(&self) -> MmapStore<E> {
        MmapStore::new_from_slice(
            self.store.len() / self.elem_len,
            &self.store[..(self.len() * self.elem_len)],
        ).expect("new_from_slice failed")
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
    #[allow(unsafe_code)]
    fn new(size: usize) -> Result<Self, &'static str> {
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

    fn new_from_slice(size: usize, data: &[u8]) -> Result<Self, &'static str> {
        assert_eq!(data.len() % E::byte_len(), 0);

        let mut res = Self::new(size)?;

        res.store_copy_from_slice(0, data);
        res.elem_len = E::byte_len();
        res.len = data.len() / res.elem_len;

        Ok(res)
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
        buf.copy_from_slice(&self.store_read_range(start, end));
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
