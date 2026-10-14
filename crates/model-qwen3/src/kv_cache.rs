use crate::{ModelError, Result};

/// Checks that `rows` more tokens fit in a cache holding `len` of `capacity` tokens.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn check_room(len: usize, rows: usize, capacity: usize) -> Result<()> {
    if len > capacity || rows > capacity - len {
        return Err(ModelError::CacheFull {
            len,
            rows,
            capacity,
        });
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub(crate) use gpu::KvCache;

#[cfg(feature = "cuda")]
mod gpu {
    use cuda_backend::{Backend, Tensor};

    use super::check_room;
    use crate::{Qwen3Config, Result};

    /// The keys and values of every token one layer has seen.
    ///
    /// Both tensors are `[kv_heads, capacity, head_dim]` and are allocated once,
    /// with room for the whole sequence. The first `len` rows of every head are
    /// filled, and each new token writes only its own row. Keys are stored
    /// after their norm and rotation, so a later token can use them as they are.
    pub(crate) struct KvCache {
        pub(crate) k: Tensor<f32>,
        pub(crate) v: Tensor<f32>,
        pub(crate) len: usize,
    }

    impl KvCache {
        /// An empty cache with room for `capacity` tokens.
        pub(crate) fn new(
            backend: &Backend,
            config: &Qwen3Config,
            capacity: usize,
        ) -> Result<Self> {
            let shape = [config.num_key_value_heads, capacity, config.head_dim];
            Ok(Self {
                k: backend.zeros(&shape)?,
                v: backend.zeros(&shape)?,
                len: 0,
            })
        }

        /// An empty cache filled with NaN, used by the checks. Reading a row
        /// nothing wrote then shows up as NaN instead of quietly reading zeros
        pub(crate) fn filled_with_nan(
            backend: &Backend,
            config: &Qwen3Config,
            capacity: usize,
        ) -> Result<Self> {
            let shape = [config.num_key_value_heads, capacity, config.head_dim];
            let nan = vec![f32::NAN; shape.iter().product()];
            Ok(Self {
                k: backend.upload(&nan, &shape)?,
                v: backend.upload(&nan, &shape)?,
                len: 0,
            })
        }

        pub(crate) fn capacity(&self) -> usize {
            usize::try_from(self.k.shape()[1]).expect("tensor dimensions are never negative")
        }

        /// Checks that `rows` more tokens fit before any of them are written.
        pub(crate) fn check_room(&self, rows: usize) -> Result<()> {
            check_room(self.len, rows, self.capacity())
        }

        /// Writes the keys and values of the next `rows` tokens after the
        /// ones already cached. Both are `[kv_heads, rows, head_dim]`.
        pub(crate) fn append(
            &mut self,
            backend: &Backend,
            k: &Tensor<f32>,
            v: &Tensor<f32>,
            rows: usize,
        ) -> Result<()> {
            self.check_room(rows)?;
            backend.write_rows(&mut self.k, self.len, k)?;
            backend.write_rows(&mut self.v, self.len, v)?;
            self.len += rows;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_rows_that_fill_the_cache() {
        assert!(check_room(5, 1, 6).is_ok());
        assert!(check_room(0, 6, 6).is_ok());
    }

    #[test]
    fn rejects_rows_past_the_end() {
        let error = check_room(5, 2, 6).unwrap_err();

        assert_eq!(
            error.to_string(),
            "KV cache holds 5 of 6 tokens, so 2 more do not fit"
        );
    }

    #[test]
    fn rejects_ranges_that_would_overflow() {
        assert!(check_room(1, usize::MAX, 6).is_err());
        assert!(check_room(usize::MAX, 1, 6).is_err());
    }
}
