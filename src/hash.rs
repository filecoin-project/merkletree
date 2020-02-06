//! Hash infrastructure for items in Merkle Tree.

use std::hash::Hasher;

/// A hashable type.
///
/// Types implementing `Hashable` are able to be [`hash`]ed with an instance of
/// [`Hasher`].
///
/// ## Implementing `Hashable`
///
/// You can derive `Hashable` with `#[derive(Hashable)]` if all fields implement `Hashable`.
/// The resulting hash will be the combination of the values from calling
/// [`hash`] on each field.
///
/// ```text
/// #[macro_use]
/// extern crate merkletree_derive;
/// extern crate merkletree;
///
/// use merkletree::hash::Hashable;
///
/// fn main() {
///     #[derive(Hashable)]
///     struct Foo {
///         name: String,
///         country: String,
///     }
/// }
/// ```
///
/// If you need more control over how a value is hashed, you can of course
/// implement the `Hashable` trait yourself:
///
/// ```
/// extern crate merkletree;
///
/// use merkletree::hash::Hashable;
/// use std::hash::Hasher;
/// use std::collections::hash_map::DefaultHasher;
///
/// fn main() {
///    struct Person {
///        id: u32,
///        name: String,
///        phone: u64,
///    }
///
///    impl<H: Hasher> Hashable<H> for Person {
///        fn hash(&self, state: &mut H) {
///            self.id.hash(state);
///            self.name.hash(state);
///            self.phone.hash(state);
///        }
///    }
///
///    let foo = Person{
///        id: 1,
///        name: String::from("blah"),
///        phone: 2,
///    };
///
///    let mut hr = DefaultHasher::new();
///    foo.hash(&mut hr);
///    assert_eq!(hr.finish(), 7101638158313343130)
/// }
/// ```
///
/// ## `Hashable` and `Eq`
///
/// When implementing both `Hashable` and [`Eq`], it is important that the following
/// property holds:
///
/// ```text
/// k1 == k2 -> hash(k1) == hash(k2)
/// ```
///
/// In other words, if two keys are equal, their hashes must also be equal.
pub trait Hashable<H: Hasher> {
    /// Feeds this value into the given [`Hasher`].
    ///
    /// [`Hasher`]: trait.Hasher.html
    fn hash(&self, state: &mut H);

    /// Feeds a slice of this type into the given [`Hasher`].
    ///
    /// [`Hasher`]: trait.Hasher.html
    fn hash_slice(data: &[Self], state: &mut H)
    where
        Self: Sized,
    {
        for piece in data {
            piece.hash(state);
        }
    }
}

/// MT leaf hash prefix
const LEAF: u8 = 0x00;

/// MT interior node hash prefix
const INTERIOR: u8 = 0x01;

/// A trait for hashing an arbitrary stream of bytes for calculating merkle tree
/// nodes.
///
/// T is a hash item must be of known size at compile time, globally ordered, with
/// default value as a neutral element of the hash space. Neutral element is
/// interpreted as 0 or nil and required for evaluation of merkle tree.
///
/// [`Algorithm`] breaks the [`Hasher`] contract at `finish()`, but that is intended.
/// This trait extends [`Hasher`] with `hash -> T` and `reset` state methods,
/// plus implements default behavior of evaluation of MT interior nodes.
pub trait Algorithm<T>: Hasher + Default
where
    T: Clone + AsRef<[u8]>,
{
    /// Returns the hash value for the data stream written so far.
    fn hash(&mut self) -> T;

    /// Reset Hasher state.
    #[inline]
    fn reset(&mut self) {
        *self = Self::default();
    }

    /// Returns hash value for MT leaf (prefix 0x00).
    #[inline]
    fn leaf(&mut self, leaf: T) -> T {
        self.write(&[LEAF]);
        self.write(leaf.as_ref());
        self.hash()
    }

    /// Returns hash value for MT interior node (prefix 0x01).
    #[inline]
    fn node(&mut self, left: T, right: T, _height: usize) -> T {
        self.write(&[INTERIOR]);
        self.write(left.as_ref());
        self.write(right.as_ref());
        self.hash()
    }

    /// Returns hash value for MT interior node (prefix 0x01).
    #[inline]
    fn multi_node(&mut self, nodes: &[T], _height: usize) -> T {
        self.write(&[INTERIOR]);
        for node in nodes {
            self.write(node.as_ref());
        }
        self.hash()
    }
}
