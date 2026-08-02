// Serialization helpers for types that are not directly representable on the
// wire. Packable (Int) Unit: a Unit is carried as an Int.

// pack :: Int -> Unit
pub fn morloc_pack_unit(_u: i64) {}

// unpack :: Unit -> Int
pub fn morloc_unpack_unit(_x: ()) -> i64 {
    0
}
