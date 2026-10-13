use std::fmt;
use std::path::PathBuf;

use safetensors::Dtype as SafeDType;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    Float4,
    Float6E2M3,
    Float6E3M2,
    Float8E4M3,
    Float8E4M3Fnuz,
    Float8E5M2,
    Float8E5M2Fnuz,
    Float8E8M0,
    BFloat16,
    Float16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Complex64,
    Other(String),
}

impl DType {
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Bool => "bool",
            Self::Float4 => "float4",
            Self::Float6E2M3 => "float6_e2m3",
            Self::Float6E3M2 => "float6_e3m2",
            Self::Float8E4M3 => "float8_e4m3",
            Self::Float8E4M3Fnuz => "float8_e4m3fnuz",
            Self::Float8E5M2 => "float8_e5m2",
            Self::Float8E5M2Fnuz => "float8_e5m2fnuz",
            Self::Float8E8M0 => "float8_e8m0",
            Self::BFloat16 => "bfloat16",
            Self::Float16 => "float16",
            Self::Float32 => "float32",
            Self::Float64 => "float64",
            Self::Int8 => "int8",
            Self::Int16 => "int16",
            Self::Int32 => "int32",
            Self::Int64 => "int64",
            Self::UInt8 => "uint8",
            Self::UInt16 => "uint16",
            Self::UInt32 => "uint32",
            Self::UInt64 => "uint64",
            Self::Complex64 => "complex64",
            Self::Other(name) => name,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl From<SafeDType> for DType {
    fn from(dtype: SafeDType) -> Self {
        match dtype {
            SafeDType::BOOL => Self::Bool,
            SafeDType::F4 => Self::Float4,
            SafeDType::F6_E2M3 => Self::Float6E2M3,
            SafeDType::F6_E3M2 => Self::Float6E3M2,
            SafeDType::F8_E4M3 => Self::Float8E4M3,
            SafeDType::F8_E4M3FNUZ => Self::Float8E4M3Fnuz,
            SafeDType::F8_E5M2 => Self::Float8E5M2,
            SafeDType::F8_E5M2FNUZ => Self::Float8E5M2Fnuz,
            SafeDType::F8_E8M0 => Self::Float8E8M0,
            SafeDType::BF16 => Self::BFloat16,
            SafeDType::F16 => Self::Float16,
            SafeDType::F32 => Self::Float32,
            SafeDType::F64 => Self::Float64,
            SafeDType::I8 => Self::Int8,
            SafeDType::I16 => Self::Int16,
            SafeDType::I32 => Self::Int32,
            SafeDType::I64 => Self::Int64,
            SafeDType::U8 => Self::UInt8,
            SafeDType::U16 => Self::UInt16,
            SafeDType::U32 => Self::UInt32,
            SafeDType::U64 => Self::UInt64,
            SafeDType::C64 => Self::Complex64,
            other => Self::Other(format!("{other:?}")),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Cuda { ordinal: u32 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BufferOwner {
    ModelFile(PathBuf),
    OwnedHost,
    DeviceAllocation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorLayout {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl TensorLayout {
    #[must_use]
    pub fn contiguous(dtype: DType, shape: Vec<usize>) -> Self {
        let mut stride = 1;
        let mut strides = vec![0; shape.len()];
        for (index, dimension) in shape.iter().enumerate().rev() {
            strides[index] = stride;
            stride *= dimension;
        }
        Self {
            dtype,
            shape,
            strides,
        }
    }

    #[must_use]
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDescriptor {
    pub name: String,
    pub layout: TensorLayout,
    pub device: Device,
    pub owner: BufferOwner,
    pub byte_len: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedTensor {
    pub descriptor: TensorDescriptor,
    pub bytes: Vec<u8>,
}

impl OwnedTensor {
    /// Decodes a float32 tensor from its little-endian `SafeTensors` bytes.
    ///
    /// # Errors
    ///
    /// Returns an error when the tensor is not float32 or its byte length does
    /// not match its shape.
    pub fn to_f32_vec(&self) -> crate::Result<Vec<f32>> {
        if self.descriptor.layout.dtype != DType::Float32 {
            return Err(crate::EngineError::TensorDtype {
                name: self.descriptor.name.clone(),
                expected: "float32",
                actual: self.descriptor.layout.dtype.to_string(),
            });
        }

        let expected = self.descriptor.layout.element_count() * size_of::<f32>();
        if self.bytes.len() != expected {
            return Err(crate::EngineError::TensorDataLength {
                name: self.descriptor.name.clone(),
                expected,
                actual: self.bytes.len(),
            });
        }

        let (values, _) = self.bytes.as_chunks::<{ size_of::<f32>() }>();
        Ok(values
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect())
    }

    /// Decodes a float32 or bfloat16 tensor as float32 values.
    ///
    /// Every bfloat16 value is also a float32 value, so widening is exact.
    ///
    /// # Errors
    ///
    /// Returns an error when the tensor has another dtype or its byte length
    /// does not match its shape.
    pub fn widen_to_f32_vec(&self) -> crate::Result<Vec<f32>> {
        if self.descriptor.layout.dtype != DType::BFloat16 {
            return self.to_f32_vec();
        }

        // bfloat16 is the top half of the float32 with the same value
        Ok(self
            .to_bf16_bits()?
            .into_iter()
            .map(|bits| f32::from_bits(u32::from(bits) << 16))
            .collect())
    }

    /// Decodes the raw bits of a bfloat16 tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the tensor is not bfloat16 or its byte length does
    /// not match its shape.
    pub fn to_bf16_bits(&self) -> crate::Result<Vec<u16>> {
        if self.descriptor.layout.dtype != DType::BFloat16 {
            return Err(crate::EngineError::TensorDtype {
                name: self.descriptor.name.clone(),
                expected: "bfloat16",
                actual: self.descriptor.layout.dtype.to_string(),
            });
        }

        let expected = self.descriptor.layout.element_count() * size_of::<u16>();
        if self.bytes.len() != expected {
            return Err(crate::EngineError::TensorDataLength {
                name: self.descriptor.name.clone(),
                expected,
                actual: self.bytes.len(),
            });
        }

        let (values, _) = self.bytes.as_chunks::<{ size_of::<u16>() }>();
        Ok(values
            .iter()
            .map(|bytes| u16::from_le_bytes(*bytes))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{BufferOwner, DType, Device, OwnedTensor, TensorDescriptor, TensorLayout};

    #[test]
    fn creates_row_major_strides() {
        let layout = TensorLayout::contiguous(DType::Float32, vec![2, 3, 4]);
        assert_eq!(layout.strides, vec![12, 4, 1]);
        assert_eq!(layout.element_count(), 24);
    }

    #[test]
    fn handles_scalar_layouts() {
        let layout = TensorLayout::contiguous(DType::Float32, vec![]);
        assert_eq!(layout.strides, Vec::<usize>::new());
        assert_eq!(layout.element_count(), 1);
    }

    #[test]
    fn decodes_float32_tensor_bytes() {
        let tensor = OwnedTensor {
            descriptor: TensorDescriptor {
                name: "values".to_owned(),
                layout: TensorLayout::contiguous(DType::Float32, vec![2]),
                device: Device::Cpu,
                owner: BufferOwner::ModelFile(PathBuf::from("fixture.safetensors")),
                byte_len: 8,
            },
            bytes: [1.5_f32.to_le_bytes(), (-2.25_f32).to_le_bytes()].concat(),
        };

        assert_eq!(tensor.to_f32_vec().unwrap(), vec![1.5, -2.25]);
        assert_eq!(tensor.widen_to_f32_vec().unwrap(), vec![1.5, -2.25]);
    }

    #[test]
    fn decodes_bfloat16_tensor_bits() {
        let tensor = OwnedTensor {
            descriptor: TensorDescriptor {
                name: "values".to_owned(),
                layout: TensorLayout::contiguous(DType::BFloat16, vec![2]),
                device: Device::Cpu,
                owner: BufferOwner::ModelFile(PathBuf::from("fixture.safetensors")),
                byte_len: 4,
            },
            bytes: [0x3f80_u16.to_le_bytes(), 0xc010_u16.to_le_bytes()].concat(),
        };

        assert_eq!(tensor.to_bf16_bits().unwrap(), vec![0x3f80, 0xc010]);
        assert_eq!(tensor.widen_to_f32_vec().unwrap(), vec![1.0, -2.25]);
    }
}
