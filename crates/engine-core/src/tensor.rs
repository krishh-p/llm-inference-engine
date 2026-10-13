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

#[cfg(test)]
mod tests {
    use super::{DType, TensorLayout};

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
}
