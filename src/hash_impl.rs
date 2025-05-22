use std::hash::Hasher;
use std::mem;
use std::slice;

use crate::hash::Hashable;

macro_rules! impl_write {
    ($(($ty:ident, $meth:ident),)*) => {$(
        impl<H: Hasher> Hashable<H> for $ty {
            fn hash(&self, state: &mut H) {
                state.$meth(*self)
            }

            #[allow(trivial_casts, unsafe_code)]
            fn hash_slice(data: &[$ty], state: &mut H) {
                let newlen = data.len() * mem::size_of::<$ty>();
                let ptr = data.as_ptr() as *const u8;
                state.write(unsafe { slice::from_raw_parts(ptr, newlen) })
            }
        }
    )*}
}

impl_write! {
    (u8, write_u8),
    (u16, write_u16),
    (u32, write_u32),
    (u64, write_u64),
    (usize, write_usize),
    (i8, write_i8),
    (i16, write_i16),
    (i32, write_i32),
    (i64, write_i64),
    (isize, write_isize),
    (u128, write_u128),
    (i128, write_i128),
}

impl<H: Hasher, const N: usize> Hashable<H> for [u8; N] {
    fn hash(&self, state: &mut H) {
        state.write(self.as_ref())
    }

    #[allow(trivial_casts, unsafe_code)]
    fn hash_slice(data: &[[u8; N]], state: &mut H) {
        let newlen = mem::size_of_val(data);
        let ptr = data.as_ptr() as *const u8;
        state.write(unsafe { slice::from_raw_parts(ptr, newlen) })
    }
}

impl<H: Hasher> Hashable<H> for Vec<u8> {
    fn hash(&self, state: &mut H) {
        state.write(self.as_ref());
    }
}

impl<H: Hasher> Hashable<H> for bool {
    fn hash(&self, state: &mut H) {
        state.write_u8(*self as u8)
    }
}

impl<H: Hasher> Hashable<H> for char {
    fn hash(&self, state: &mut H) {
        state.write_u32(*self as u32)
    }
}

impl<H: Hasher> Hashable<H> for str {
    fn hash(&self, state: &mut H) {
        state.write(self.as_bytes());
        // empty str nope: state.write_u8(0xff)
    }
}

impl<H: Hasher> Hashable<H> for String {
    fn hash(&self, state: &mut H) {
        state.write(self.as_bytes());
        // empty str nope: state.write_u8(0xff)
    }
}

macro_rules! impl_hash_tuple {
    () => (
        impl<H: Hasher> Hashable<H> for () {
            fn hash(&self, _: &mut H) {}
        }
    );

    ( $($name:ident)+) => (
        impl<Z: Hasher, $($name: Hashable<Z>),*> Hashable<Z> for ($($name,)*)
        where
            last_type!($($name,)+): ?Sized
        {
            #[allow(non_snake_case)]
            fn hash(&self, state: &mut Z) {
                let ($(ref $name,)*) = *self;
                $($name.hash(state);)*
            }
        }
    );
}

macro_rules! last_type {
    ($a:ident,) => { $a };
    ($a:ident, $($rest_a:ident,)+) => { last_type!($($rest_a,)+) };
}

impl_hash_tuple! {}
impl_hash_tuple! { A }
impl_hash_tuple! { A B }
impl_hash_tuple! { A B C }
impl_hash_tuple! { A B C D }
impl_hash_tuple! { A B C D E }
impl_hash_tuple! { A B C D E F }
impl_hash_tuple! { A B C D E F G }
impl_hash_tuple! { A B C D E F G H }
impl_hash_tuple! { A B C D E F G H I }
impl_hash_tuple! { A B C D E F G H I J }
impl_hash_tuple! { A B C D E F G H I J K }
impl_hash_tuple! { A B C D E F G H I J K L }

impl<H: Hasher, T: Hashable<H>> Hashable<H> for [T] {
    fn hash(&self, state: &mut H) {
        Hashable::hash_slice(self, state)
    }
}

impl<'a, H: Hasher, T: ?Sized + Hashable<H>> Hashable<H> for &'a T {
    fn hash(&self, state: &mut H) {
        (**self).hash(state);
    }
}

impl<'a, H: Hasher, T: ?Sized + Hashable<H>> Hashable<H> for &'a mut T {
    fn hash(&self, state: &mut H) {
        (**self).hash(state);
    }
}

impl<H: Hasher, T: ?Sized> Hashable<H> for *const T {
    #[allow(trivial_casts, unsafe_code)]
    fn hash(&self, state: &mut H) {
        if mem::size_of::<Self>() == mem::size_of::<usize>() {
            // Thin pointer
            state.write_usize(*self as *const () as usize);
        } else {
            // Fat pointer
            let (a, b) = unsafe { *(self as *const Self as *const (usize, usize)) };
            state.write_usize(a);
            state.write_usize(b);
        }
    }
}

impl<H: Hasher, T: ?Sized> Hashable<H> for *mut T {
    #[allow(trivial_casts, unsafe_code)]
    fn hash(&self, state: &mut H) {
        if mem::size_of::<Self>() == mem::size_of::<usize>() {
            // Thin pointer
            state.write_usize(*self as *const () as usize);
        } else {
            // Fat pointer
            let (a, b) = unsafe { *(self as *const Self as *const (usize, usize)) };
            state.write_usize(a);
            state.write_usize(b);
        }
    }
}
