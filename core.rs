// Native Rust implementations for the root typeclasses and list operations.
//
// Argument-passing follows the morloc Rust FFI convention: a Copy scalar is
// taken by value; a non-Copy value (String/Vec/VecDeque) and every generic
// (type-variable) parameter is taken by shared reference. Functions that build
// a new owned collection clone only the elements they must own -- there are no
// copies of whole containers that are merely read. Higher-order functions call
// their closure argument with every element/accumuland BY REFERENCE.
//
// The numeric surface (neg/abs/pow/inv/ln/to_real/into/tryInto) is expressed
// through small traits macro-implemented over the concrete numeric types, so a
// single generic morloc method dispatches to the right per-type operation.

use std::collections::VecDeque;

// ===========================================================================
// Booleans, identity, comparison
// ===========================================================================

pub fn morloc_not(x: bool) -> bool { !x }
pub fn morloc_and(x: bool, y: bool) -> bool { x && y }
pub fn morloc_or(x: bool, y: bool) -> bool { x || y }

// idrust: force a value through the Rust pool (mostly for testing).
pub fn morloc_id<A: Clone>(x: &A) -> A { x.clone() }

pub fn morloc_eq<A: PartialEq>(x: &A, y: &A) -> bool { x == y }
pub fn morloc_le<A: PartialOrd>(x: &A, y: &A) -> bool { x <= y }

// ===========================================================================
// Arithmetic (Integral / Numeric)
// ===========================================================================

pub fn morloc_add<A: Copy + std::ops::Add<Output = A>>(x: &A, y: &A) -> A { *x + *y }
pub fn morloc_sub<A: Copy + std::ops::Sub<Output = A>>(x: &A, y: &A) -> A { *x - *y }
pub fn morloc_mul<A: Copy + std::ops::Mul<Output = A>>(x: &A, y: &A) -> A { *x * *y }
pub fn morloc_div<A: Copy + std::ops::Div<Output = A>>(x: &A, y: &A) -> A { *x / *y }

// Integer division and remainder truncate toward zero (Rust's `/` and `%` on
// integers), matching the C++ member's integer `(/)`/`(%)`.
pub fn morloc_int_div<A: Copy + std::ops::Div<Output = A>>(x: &A, y: &A) -> A { *x / *y }
pub fn morloc_int_mod<A: Copy + std::ops::Rem<Output = A>>(x: &A, y: &A) -> A { *x % *y }

// -- neg / abs over every numeric type (unary `-` is undefined for unsigned in
//    Rust, so unsigned neg wraps, matching the C++ template's `(-1)*x`) --------
pub trait MorlocNum: Copy {
    fn m_neg(self) -> Self;
    fn m_abs(self) -> Self;
}
macro_rules! impl_num_signed {
    ($($t:ty),*) => {$(impl MorlocNum for $t {
        fn m_neg(self) -> Self { -self }
        fn m_abs(self) -> Self { self.abs() }
    })*};
}
macro_rules! impl_num_unsigned {
    ($($t:ty),*) => {$(impl MorlocNum for $t {
        fn m_neg(self) -> Self { self.wrapping_neg() }
        fn m_abs(self) -> Self { self }
    })*};
}
impl_num_signed!(i8, i16, i32, i64, f32, f64);
impl_num_unsigned!(u8, u16, u32, u64);

pub fn morloc_neg<A: MorlocNum>(x: &A) -> A { x.m_neg() }
pub fn morloc_abs<A: MorlocNum>(x: &A) -> A { x.m_abs() }

// -- pow: float exponentiation for reals, integer power for integers ---------
pub trait MorlocPow: Copy {
    fn m_pow(self, e: Self) -> Self;
}
macro_rules! impl_pow_int {
    ($($t:ty),*) => {$(impl MorlocPow for $t {
        fn m_pow(self, e: Self) -> Self { self.pow(e as u32) }
    })*};
}
macro_rules! impl_pow_float {
    ($($t:ty),*) => {$(impl MorlocPow for $t {
        fn m_pow(self, e: Self) -> Self { self.powf(e) }
    })*};
}
impl_pow_int!(i8, i16, i32, i64, u8, u16, u32, u64);
impl_pow_float!(f32, f64);

pub fn morloc_pow<A: MorlocPow>(x: &A, y: &A) -> A { x.m_pow(*y) }

// -- float-only Numeric operations -------------------------------------------
pub trait MorlocFloat: Copy {
    fn m_inv(self) -> Self;
    fn m_ln(self) -> Self;
    fn m_floor_div(self, o: Self) -> Self;
    fn m_float_mod(self, o: Self) -> Self;
}
macro_rules! impl_float {
    ($($t:ty),*) => {$(impl MorlocFloat for $t {
        fn m_inv(self) -> Self { (1.0 as $t) / self }
        fn m_ln(self) -> Self { self.ln() }
        fn m_floor_div(self, o: Self) -> Self { (self / o).floor() }
        fn m_float_mod(self, o: Self) -> Self { self - o * (self / o).floor() }
    })*};
}
impl_float!(f32, f64);

pub fn morloc_inv<A: MorlocFloat>(x: &A) -> A { x.m_inv() }
pub fn morloc_ln<A: MorlocFloat>(x: &A) -> A { x.m_ln() }
pub fn morloc_floor_div<A: MorlocFloat>(x: &A, y: &A) -> A { x.m_floor_div(*y) }
pub fn morloc_float_mod<A: MorlocFloat>(x: &A, y: &A) -> A { x.m_float_mod(*y) }

// ===========================================================================
// Numeric conversions (RealLike / TotalInto / PartialInto)
// ===========================================================================

pub trait MorlocReal: Copy {
    fn m_to_real(self) -> f64;
}
macro_rules! impl_real {
    ($($t:ty),*) => {$(impl MorlocReal for $t {
        fn m_to_real(self) -> f64 { self as f64 }
    })*};
}
impl_real!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

pub fn morloc_to_real<A: MorlocReal>(x: &A) -> f64 { x.m_to_real() }

// TotalInto: lossless widening. The set of instances guarantees the `as` cast
// is exact, so a plain cast is correct.
pub trait MorlocInto: Copy {
    fn as_u16(self) -> u16;
    fn as_u32(self) -> u32;
    fn as_u64(self) -> u64;
    fn as_i16(self) -> i16;
    fn as_i32(self) -> i32;
    fn as_i64(self) -> i64;
    fn as_f32(self) -> f32;
    fn as_f64(self) -> f64;
}
macro_rules! impl_into {
    ($($t:ty),*) => {$(impl MorlocInto for $t {
        fn as_u16(self) -> u16 { self as u16 }
        fn as_u32(self) -> u32 { self as u32 }
        fn as_u64(self) -> u64 { self as u64 }
        fn as_i16(self) -> i16 { self as i16 }
        fn as_i32(self) -> i32 { self as i32 }
        fn as_i64(self) -> i64 { self as i64 }
        fn as_f32(self) -> f32 { self as f32 }
        fn as_f64(self) -> f64 { self as f64 }
    })*};
}
impl_into!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

pub fn morloc_into_u16<F: MorlocInto>(x: &F) -> u16 { x.as_u16() }
pub fn morloc_into_u32<F: MorlocInto>(x: &F) -> u32 { x.as_u32() }
pub fn morloc_into_u64<F: MorlocInto>(x: &F) -> u64 { x.as_u64() }
pub fn morloc_into_i16<F: MorlocInto>(x: &F) -> i16 { x.as_i16() }
pub fn morloc_into_i32<F: MorlocInto>(x: &F) -> i32 { x.as_i32() }
pub fn morloc_into_i64<F: MorlocInto>(x: &F) -> i64 { x.as_i64() }
pub fn morloc_into_int<F: MorlocInto>(x: &F) -> i64 { x.as_i64() }
pub fn morloc_into_f32<F: MorlocInto>(x: &F) -> f32 { x.as_f32() }
pub fn morloc_into_f64<F: MorlocInto>(x: &F) -> f64 { x.as_f64() }

// PartialInto: bounds-checked. Integer sources use `TryFrom` (range check);
// float sources reject non-finite and non-integer values, then range-check
// through a wide `i128` intermediate. A failure raises a catchable morloc error.
pub trait MorlocTryTo: Copy {
    fn m_try_u8(self) -> Option<u8>;
    fn m_try_u16(self) -> Option<u16>;
    fn m_try_u32(self) -> Option<u32>;
    fn m_try_u64(self) -> Option<u64>;
    fn m_try_i8(self) -> Option<i8>;
    fn m_try_i16(self) -> Option<i16>;
    fn m_try_i32(self) -> Option<i32>;
    fn m_try_i64(self) -> Option<i64>;
}
macro_rules! impl_try_int {
    ($($t:ty),*) => {$(impl MorlocTryTo for $t {
        fn m_try_u8(self)  -> Option<u8>  { u8::try_from(self).ok() }
        fn m_try_u16(self) -> Option<u16> { u16::try_from(self).ok() }
        fn m_try_u32(self) -> Option<u32> { u32::try_from(self).ok() }
        fn m_try_u64(self) -> Option<u64> { u64::try_from(self).ok() }
        fn m_try_i8(self)  -> Option<i8>  { i8::try_from(self).ok() }
        fn m_try_i16(self) -> Option<i16> { i16::try_from(self).ok() }
        fn m_try_i32(self) -> Option<i32> { i32::try_from(self).ok() }
        fn m_try_i64(self) -> Option<i64> { i64::try_from(self).ok() }
    })*};
}
macro_rules! impl_try_float {
    ($($t:ty),*) => {$(impl MorlocTryTo for $t {
        fn m_try_u8(self)  -> Option<u8>  { float_to_int(self as f64).and_then(|v| u8::try_from(v).ok()) }
        fn m_try_u16(self) -> Option<u16> { float_to_int(self as f64).and_then(|v| u16::try_from(v).ok()) }
        fn m_try_u32(self) -> Option<u32> { float_to_int(self as f64).and_then(|v| u32::try_from(v).ok()) }
        fn m_try_u64(self) -> Option<u64> { float_to_int(self as f64).and_then(|v| u64::try_from(v).ok()) }
        fn m_try_i8(self)  -> Option<i8>  { float_to_int(self as f64).and_then(|v| i8::try_from(v).ok()) }
        fn m_try_i16(self) -> Option<i16> { float_to_int(self as f64).and_then(|v| i16::try_from(v).ok()) }
        fn m_try_i32(self) -> Option<i32> { float_to_int(self as f64).and_then(|v| i32::try_from(v).ok()) }
        fn m_try_i64(self) -> Option<i64> { float_to_int(self as f64).and_then(|v| i64::try_from(v).ok()) }
    })*};
}
impl_try_int!(i8, i16, i32, i64, u8, u16, u32, u64);
impl_try_float!(f32, f64);

// A finite, integer-valued float as i128 (a wide intermediate that covers every
// integer target's range; out-of-range floats saturate and fail the TryFrom).
fn float_to_int(x: f64) -> Option<i128> {
    if x.is_finite() && x.trunc() == x {
        Some(x as i128)
    } else {
        None
    }
}

macro_rules! try_into_fn {
    ($name:ident, $method:ident, $to:ty, $msg:literal) => {
        pub fn $name<F: MorlocTryTo>(x: &F) -> $to {
            x.$method()
                .unwrap_or_else(|| rustmorloc::morloc_throw($msg))
        }
    };
}
try_into_fn!(morloc_try_u8, m_try_u8, u8, "value out of range for U8");
try_into_fn!(morloc_try_u16, m_try_u16, u16, "value out of range for U16");
try_into_fn!(morloc_try_u32, m_try_u32, u32, "value out of range for U32");
try_into_fn!(morloc_try_u64, m_try_u64, u64, "value out of range for U64");
try_into_fn!(morloc_try_i8, m_try_i8, i8, "value out of range for I8");
try_into_fn!(morloc_try_i16, m_try_i16, i16, "value out of range for I16");
try_into_fn!(morloc_try_i32, m_try_i32, i32, "value out of range for I32");
try_into_fn!(morloc_try_i64, m_try_i64, i64, "value out of range for I64");
try_into_fn!(morloc_try_int, m_try_i64, i64, "value out of range for Int");

// ===========================================================================
// Sizeable: a single generic size over the sizeable types
// ===========================================================================

pub trait MorlocLen {
    fn m_len(&self) -> u64;
}
impl<A> MorlocLen for Vec<A> {
    fn m_len(&self) -> u64 { self.len() as u64 }
}
impl MorlocLen for String {
    fn m_len(&self) -> u64 { self.len() as u64 }
}
pub fn morloc_size<F: MorlocLen>(x: &F) -> u64 { x.m_len() }

// ===========================================================================
// Semigroup
// ===========================================================================

pub fn morloc_str_add(x: &str, y: &str) -> String {
    let mut s = String::with_capacity(x.len() + y.len());
    s.push_str(x);
    s.push_str(y);
    s
}

pub fn morloc_list_add<A: Clone>(xs: &[A], ys: &[A]) -> Vec<A> {
    let mut v = Vec::with_capacity(xs.len() + ys.len());
    v.extend_from_slice(xs);
    v.extend_from_slice(ys);
    v
}

pub fn morloc_deque_add<A: Clone>(xs: &VecDeque<A>, ys: &VecDeque<A>) -> VecDeque<A> {
    let mut v = VecDeque::with_capacity(xs.len() + ys.len());
    v.extend(xs.iter().cloned());
    v.extend(ys.iter().cloned());
    v
}

// ===========================================================================
// Functor
// ===========================================================================

pub fn morloc_map_vec<A, B, F: Fn(&A) -> B>(f: F, xs: &[A]) -> Vec<B> {
    xs.iter().map(|x| f(x)).collect()
}

pub fn morloc_map_deque<A, B, F: Fn(&A) -> B>(f: F, xs: &VecDeque<A>) -> VecDeque<B> {
    xs.iter().map(|x| f(x)).collect()
}

// ===========================================================================
// Foldable (fold threads an owned accumulator; the closure reads it by ref)
// ===========================================================================

pub fn morloc_fold_vec<B: Clone, A, F: Fn(&B, &A) -> B>(f: F, init: &B, xs: &[A]) -> B {
    xs.iter().fold(init.clone(), |acc, x| f(&acc, x))
}

pub fn morloc_fold_deque<B: Clone, A, F: Fn(&B, &A) -> B>(f: F, init: &B, xs: &VecDeque<A>) -> B {
    xs.iter().fold(init.clone(), |acc, x| f(&acc, x))
}

pub fn morloc_fold1_vec<A: Clone, F: Fn(&A, &A) -> A>(f: F, xs: &[A]) -> A {
    let mut it = xs.iter();
    let mut acc = it
        .next()
        .unwrap_or_else(|| rustmorloc::morloc_throw("fold1: empty collection"))
        .clone();
    for x in it {
        acc = f(&acc, x);
    }
    acc
}

pub fn morloc_fold1_deque<A: Clone, F: Fn(&A, &A) -> A>(f: F, xs: &VecDeque<A>) -> A {
    let mut it = xs.iter();
    let mut acc = it
        .next()
        .unwrap_or_else(|| rustmorloc::morloc_throw("fold1: empty collection"))
        .clone();
    for x in it {
        acc = f(&acc, x);
    }
    acc
}

pub fn morloc_safe_fold1_vec<A: Clone, F: Fn(&A, &A) -> A>(f: F, xs: &[A]) -> Option<A> {
    let mut it = xs.iter();
    let mut acc = it.next()?.clone();
    for x in it {
        acc = f(&acc, x);
    }
    Some(acc)
}

pub fn morloc_safe_fold1_deque<A: Clone, F: Fn(&A, &A) -> A>(f: F, xs: &VecDeque<A>) -> Option<A> {
    let mut it = xs.iter();
    let mut acc = it.next()?.clone();
    for x in it {
        acc = f(&acc, x);
    }
    Some(acc)
}

// ===========================================================================
// Stack / Queue
// ===========================================================================

pub fn morloc_cons_vec<A: Clone>(x: &A, xs: &[A]) -> Vec<A> {
    let mut v = Vec::with_capacity(xs.len() + 1);
    v.push(x.clone());
    v.extend_from_slice(xs);
    v
}

pub fn morloc_uncons_vec<A: Clone>(xs: &[A]) -> (A, Vec<A>) {
    match xs.split_first() {
        Some((h, t)) => (h.clone(), t.to_vec()),
        None => rustmorloc::morloc_throw("uncons: empty list"),
    }
}

pub fn morloc_snoc_vec<A: Clone>(xs: &[A], x: &A) -> Vec<A> {
    let mut v = Vec::with_capacity(xs.len() + 1);
    v.extend_from_slice(xs);
    v.push(x.clone());
    v
}

pub fn morloc_unsnoc_vec<A: Clone>(xs: &[A]) -> (Vec<A>, A) {
    match xs.split_last() {
        Some((l, init)) => (init.to_vec(), l.clone()),
        None => rustmorloc::morloc_throw("unsnoc: empty list"),
    }
}

pub fn morloc_cons_deque<A: Clone>(x: &A, xs: &VecDeque<A>) -> VecDeque<A> {
    let mut v = VecDeque::with_capacity(xs.len() + 1);
    v.push_back(x.clone());
    v.extend(xs.iter().cloned());
    v
}

pub fn morloc_uncons_deque<A: Clone>(xs: &VecDeque<A>) -> (A, VecDeque<A>) {
    let h = xs
        .front()
        .unwrap_or_else(|| rustmorloc::morloc_throw("uncons: empty deque"))
        .clone();
    (h, xs.iter().skip(1).cloned().collect())
}

pub fn morloc_snoc_deque<A: Clone>(xs: &VecDeque<A>, x: &A) -> VecDeque<A> {
    let mut v = VecDeque::with_capacity(xs.len() + 1);
    v.extend(xs.iter().cloned());
    v.push_back(x.clone());
    v
}

pub fn morloc_unsnoc_deque<A: Clone>(xs: &VecDeque<A>) -> (VecDeque<A>, A) {
    let l = xs
        .back()
        .unwrap_or_else(|| rustmorloc::morloc_throw("unsnoc: empty deque"))
        .clone();
    (xs.iter().take(xs.len() - 1).cloned().collect(), l)
}

// ===========================================================================
// Indexable / Sliceable / IndexLike
// ===========================================================================

// IndexLike: turn an index-like value (or an optional one) into ?I64.
pub trait MorlocToIndex {
    fn m_to_index(&self) -> Option<i64>;
}
macro_rules! impl_to_index {
    ($($t:ty),*) => {$(
        impl MorlocToIndex for $t {
            fn m_to_index(&self) -> Option<i64> { Some(*self as i64) }
        }
        impl MorlocToIndex for Option<$t> {
            fn m_to_index(&self) -> Option<i64> { self.map(|v| v as i64) }
        }
    )*};
}
impl_to_index!(i8, i16, i32, i64, u8, u16, u32, u64);

pub fn morloc_to_index<I: MorlocToIndex>(x: &I) -> Option<i64> { x.m_to_index() }

// Negative indices count from the end (Python semantics).
pub fn morloc_at<A: Clone>(oi: Option<i64>, xs: &[A]) -> A {
    let i = oi.unwrap_or_else(|| rustmorloc::morloc_throw("index is null"));
    let n = xs.len() as i64;
    let j = if i < 0 { i + n } else { i };
    if j < 0 || j >= n {
        rustmorloc::morloc_throw("index out of bounds");
    }
    xs[j as usize].clone()
}

// Python-style slice with an optional start/stop/step; negative bounds and a
// negative step are supported. Bounds arrive as ?I64.
pub fn morloc_slice<A: Clone>(
    start: Option<i64>,
    stop: Option<i64>,
    step: Option<i64>,
    xs: &[A],
) -> Vec<A> {
    let n = xs.len() as i64;
    let k = step.unwrap_or(1);
    if k == 0 {
        rustmorloc::morloc_throw("slice step cannot be zero");
    }
    let (mut i, mut j) = if k > 0 {
        (start.unwrap_or(0), stop.unwrap_or(n))
    } else {
        (start.unwrap_or(n - 1), stop.unwrap_or(-1))
    };
    // A supplied negative bound wraps from the end; an omitted reverse-stop
    // default of -1 means "one before index 0" and must not wrap.
    if i < 0 && start.is_some() {
        i += n;
    }
    if j < 0 && stop.is_some() {
        j += n;
    }
    if k > 0 {
        i = i.clamp(0, n);
        j = j.clamp(0, n);
    } else {
        i = i.clamp(-1, n - 1);
        j = j.clamp(-1, n - 1);
    }
    let mut out = Vec::new();
    let mut p = i;
    while (k > 0 && p < j) || (k < 0 && p > j) {
        out.push(xs[p as usize].clone());
        p += k;
    }
    out
}

// ===========================================================================
// Other list operations
// ===========================================================================

pub fn morloc_zipWith<A, B, C, F: Fn(&A, &B) -> C>(f: F, xs: &[A], ys: &[B]) -> Vec<C> {
    xs.iter().zip(ys.iter()).map(|(a, b)| f(a, b)).collect()
}

// Stable bottom-up merge sort driven by a non-strict "a is not after b"
// predicate: `first(a, b)` must hold when a and b are equivalent.
//
// The standard library's sort_by needs a comparator that is a total order and
// panics when it detects one that is not. Both callers below are handed an
// order that need not be total: morloc's `Ord` class admits Real, whose Rust
// form f64 is only PartialOrd (every comparison against NaN is false), and
// sortBy takes an arbitrary user predicate. A merge sort has no such
// requirement -- any predicate yields a permutation of the input -- and it is
// stable, so equal elements keep their input order.
fn morloc_merge_sort_by<A: Clone, F: Fn(&A, &A) -> bool>(xs: &[A], first: F) -> Vec<A> {
    let mut src = xs.to_vec();
    let n = src.len();
    let mut dst = src.clone();
    let mut width = 1;
    while width < n {
        let mut lo = 0;
        while lo < n {
            let mid = std::cmp::min(lo + width, n);
            let hi = std::cmp::min(lo + 2 * width, n);
            let (mut i, mut j) = (lo, mid);
            for k in lo..hi {
                // Take from the left run unless it is exhausted. `first` is
                // non-strict, so on a tie it holds and the left element wins --
                // which is what makes the sort stable.
                let take_left = if i >= mid {
                    false
                } else if j >= hi {
                    true
                } else {
                    first(&src[i], &src[j])
                };
                if take_left {
                    dst[k] = src[i].clone();
                    i += 1;
                } else {
                    dst[k] = src[j].clone();
                    j += 1;
                }
            }
            lo = hi;
        }
        std::mem::swap(&mut src, &mut dst);
        width *= 2;
    }
    src
}

// PartialOrd, not Ord: morloc's `Ord` class has a universal instance and Real
// maps to f64, which does not implement Ord. NaN compares false against
// everything, so it sorts to a well-defined position rather than panicking.
pub fn morloc_sort<A: PartialOrd + Clone>(xs: &[A]) -> Vec<A> {
    morloc_merge_sort_by(xs, |a, b| a <= b)
}

// sortBy takes a "less-than" predicate: x precedes y when cmp(x, y). The
// element that "goes first" is therefore the one the predicate does not place
// after the other.
pub fn morloc_sortBy<A: Clone, F: Fn(&A, &A) -> bool>(cmp: F, xs: &[A]) -> Vec<A> {
    morloc_merge_sort_by(xs, |a, b| !cmp(b, a))
}

pub fn morloc_filter<A: Clone, F: Fn(&A) -> bool>(f: F, xs: &[A]) -> Vec<A> {
    xs.iter().filter(|x| f(x)).cloned().collect()
}

pub fn morloc_unzip<A: Clone, B: Clone>(xs: &[(A, B)]) -> (Vec<A>, Vec<B>) {
    xs.iter().map(|(a, b)| (a.clone(), b.clone())).unzip()
}

pub fn morloc_replicate<A: Clone>(n: i64, x: &A) -> Vec<A> {
    vec![x.clone(); n.max(0) as usize]
}

pub fn morloc_takeWhile<A: Clone, F: Fn(&A) -> bool>(f: F, xs: &[A]) -> Vec<A> {
    xs.iter().take_while(|x| f(x)).cloned().collect()
}

pub fn morloc_dropWhile<A: Clone, F: Fn(&A) -> bool>(f: F, xs: &[A]) -> Vec<A> {
    xs.iter().skip_while(|x| f(x)).cloned().collect()
}

pub fn morloc_partition<A: Clone, F: Fn(&A) -> bool>(f: F, xs: &[A]) -> (Vec<A>, Vec<A>) {
    xs.iter().cloned().partition(|x| f(x))
}

pub fn morloc_scanl<B: Clone, A, F: Fn(&B, &A) -> B>(f: F, init: &B, xs: &[A]) -> Vec<B> {
    let mut acc = init.clone();
    let mut out = Vec::with_capacity(xs.len() + 1);
    out.push(acc.clone());
    for x in xs {
        acc = f(&acc, x);
        out.push(acc.clone());
    }
    out
}

pub fn morloc_enumerate<A: Clone>(xs: &[A]) -> Vec<(i64, A)> {
    xs.iter()
        .enumerate()
        .map(|(i, x)| (i as i64, x.clone()))
        .collect()
}

pub fn morloc_intersperse<A: Clone>(sep: &A, xs: &[A]) -> Vec<A> {
    let mut out = Vec::with_capacity(xs.len().saturating_mul(2).saturating_sub(1));
    for (i, x) in xs.iter().enumerate() {
        if i > 0 {
            out.push(sep.clone());
        }
        out.push(x.clone());
    }
    out
}

pub fn morloc_iterate<A: Clone, F: Fn(&A) -> A>(n: i64, f: F, x: &A) -> Vec<A> {
    let n = n.max(0) as usize;
    let mut out = Vec::with_capacity(n);
    let mut cur = x.clone();
    for _ in 0..n {
        let next = f(&cur);
        out.push(cur);
        cur = next;
    }
    out
}

// Group maximal runs of adjacent elements the predicate deems equal.
pub fn morloc_groupBy<A: Clone, F: Fn(&A, &A) -> bool>(eq: F, xs: &[A]) -> Vec<Vec<A>> {
    xs.chunk_by(|a, b| eq(a, b)).map(<[A]>::to_vec).collect()
}

pub fn morloc_find<A: Clone, F: Fn(&A) -> bool>(f: F, xs: &[A]) -> Option<A> {
    xs.iter().find(|x| f(x)).cloned()
}

// unique needs only Eq, so keep insertion order with a linear-scan membership
// test (quadratic, but clones only the elements it keeps).
pub fn morloc_unique<A: PartialEq + Clone>(xs: &[A]) -> Vec<A> {
    let mut out: Vec<A> = Vec::new();
    for x in xs {
        if !out.iter().any(|y| y == x) {
            out.push(x.clone());
        }
    }
    out
}

// Group values by key, keys ascending, values in input order. Equal keys are
// merged wherever they occur, not only in runs: the stable sort brings every
// occurrence of a key together first. PartialOrd for the same reason as
// morloc_sort -- a Real key must work.
pub fn morloc_groupSort<A: PartialOrd + Clone, B: Clone>(xs: &[(A, B)]) -> Vec<(A, Vec<B>)> {
    let sorted = morloc_merge_sort_by(xs, |x, y| x.0 <= y.0);
    let mut out: Vec<(A, Vec<B>)> = Vec::new();
    for (a, b) in sorted {
        match out.last_mut() {
            Some((key, vals)) if *key == a => vals.push(b),
            _ => out.push((a, vec![b])),
        }
    }
    out
}

pub fn morloc_range(a: i64, b: i64) -> Vec<i64> {
    (a..b).collect()
}

pub fn morloc_rangeStep(a: i64, b: i64, step: i64) -> Vec<i64> {
    if step <= 0 {
        Vec::new()
    } else {
        (a..b).step_by(step as usize).collect()
    }
}

// ===========================================================================
// Packable (List a) (Deque a): explicit Vec <-> VecDeque conversion
// ===========================================================================

pub fn morloc_pack_deque<A: Clone>(xs: &[A]) -> VecDeque<A> {
    xs.iter().cloned().collect()
}

pub fn morloc_unpack_deque<A: Clone>(xs: &VecDeque<A>) -> Vec<A> {
    xs.iter().cloned().collect()
}
