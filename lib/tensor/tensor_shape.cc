// Copyright 2020 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file implements TensorShape.

#include "tfrt/tensor/tensor_shape.h"

#include <optional>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {
namespace {

llvm::SmallVector<Index, 4> ComputeInnerDims(
    const llvm::SmallVector<Index, 4>& orig_dims, int64_t num_out_dims) {
  llvm::SmallVector<Index, 4> out_dims(num_out_dims, 0);
  int64_t offset = orig_dims.size() - num_out_dims;
  for (int64_t out_dim = num_out_dims - 1; out_dim >= 0; --out_dim) {
    const int64_t in_dim = out_dim + offset;
    assert(orig_dims[in_dim] >= 0 && "Unknown dimension");
    out_dims[out_dim] = orig_dims[in_dim];
  }
  for (int64_t in_dim = 0; in_dim < offset; ++in_dim) {
    out_dims[0] *= orig_dims[in_dim];
  }
  return out_dims;
}

}  // namespace

raw_ostream& operator<<(raw_ostream& os, const TensorShape& value) {
  os << '[';
  llvm::SmallVector<Index, 8> dims;
  value.GetDimensions(&dims);
  if (!dims.empty()) {
    os << dims[0];
    for (size_t i = 1, e = dims.size(); i != e; ++i) os << ", " << dims[i];
  }
  return os << ']';
}

bool TensorShape::operator==(const TensorShape& other) const {
  // We assume that two identical shapes have the same representation kind.
  if (GetRepresentationKind() != other.GetRepresentationKind()) return false;
  if (!IsRepresentationExternal()) {
    // Both rep16 and rep32 have the same size and share the same memory, so
    // either representation is sufficient when comparing the block of memory.
    return memcmp(&representation_, &other.representation_,
                  sizeof(representation_)) == 0;
  }
  if (GetRank() != other.GetRank()) return false;
  return std::equal(representation_.rep_external.dims,
                    representation_.rep_external.dims + GetRank(),
                    other.representation_.rep_external.dims);
  return true;
}

bool TensorShape::operator!=(const TensorShape& other) const {
  return !(*this == other);
}

// Return the total number of elements in this TensorShape.  This is all of
// the dimensions multiplied together.
Index TensorShape::GetNumElements() const {
  Index result = 1;
  switch (GetRepresentationKind()) {
    case RepKind::kRep16:
      for (size_t i = 0, e = GetRank(); i != e; ++i)
        result *= representation_.rep16.dims[i];
      return result;

    case RepKind::kRep32:
      switch (GetRank()) {
        case 4:
          result = representation_.rep32.dim3;
          LLVM_FALLTHROUGH;
        case 3:
          result *= representation_.rep32.dims[2];
          LLVM_FALLTHROUGH;
        case 2:
          result *= representation_.rep32.dims[1];
          LLVM_FALLTHROUGH;
        case 1:
          result *= representation_.rep32.dims[0];
          return result;
        default:
          assert(0 && "unreachable");
          return result;
      }

    case RepKind::kRepExternal:
      for (size_t i = 0, e = GetRank(); i != e; ++i)
        result *= representation_.rep_external.dims[i];
      return result;
  }
}

void TensorShape::GetDimensions(MutableArrayRef<Index> result) const {
  auto rank = GetRank();
  assert(rank == result.size() && "Incorrect rank");
  switch (GetRepresentationKind()) {
    case RepKind::kRep16:
      for (int i = 0, e = rank; i != e; ++i)
        result[i] = representation_.rep16.dims[i];
      return;

    case RepKind::kRep32:
      switch (rank) {
        case 4:
          result[3] = representation_.rep32.dim3;
          LLVM_FALLTHROUGH;
        case 3:
          result[2] = representation_.rep32.dims[2];
          LLVM_FALLTHROUGH;
        case 2:
          result[1] = representation_.rep32.dims[1];
          LLVM_FALLTHROUGH;
        case 1:
          result[0] = representation_.rep32.dims[0];
          return;
        default:
          assert(0 && "unreachable");
          return;
      }

    case RepKind::kRepExternal:
      memcpy(result.data(), representation_.rep_external.dims,
             sizeof(size_t) * rank);
      return;
  }
}

void TensorShape::GetStrides(MutableArrayRef<Index> result) const {
  GetDimensions(result);
  Index multiplier = 1;
  for (int i = GetRank() - 1; i >= 0; --i) {
    Index dim_size = result[i];
    result[i] = multiplier;
    multiplier *= dim_size;
  }
};

// Return all of the dimensions in this TensorShape in a way that is easy to
// process.
void TensorShape::GetDimensions(llvm::SmallVectorImpl<Index>* result) const {
  result->resize(GetRank());
  GetDimensions(*result);
}

// Return strides of this TensorShape in a way that is easy to process.
void TensorShape::GetStrides(llvm::SmallVectorImpl<Index>* result) const {
  result->resize(GetRank());
  GetStrides(*result);
}

Index TensorShape::GetDimensionSize(int dim_idx) const {
  assert(dim_idx < GetRank());
  switch (GetRepresentationKind()) {
    case RepKind::kRep16:
      return representation_.rep16.dims[dim_idx];

    case RepKind::kRep32:
      switch (dim_idx) {
        case 3:
          return representation_.rep32.dim3;
        case 2:
          return representation_.rep32.dims[2];
        case 1:
          return representation_.rep32.dims[1];
        case 0:
          return representation_.rep32.dims[0];
        default:
          assert(0 && "unreachable");
          return 0;
      }

    case RepKind::kRepExternal:
      return representation_.rep_external.dims[dim_idx];
  }
}

raw_ostream& operator<<(raw_ostream& os, const PartialTensorShape& value) {
  if (value.IsUnranked()) {
    return os << "Unknown rank";
  }

  os << '[';
  if (!value.GetShape()->empty()) {
    llvm::interleaveComma(value.GetShape().value(), os);
  }
  return os << ']';
}

PartialTensorShape::PartialTensorShape(std::optional<ArrayRef<Index>> dims) {
  if (dims.has_value()) {
    llvm::SmallVector<Index, 4> dims_vec{dims.value().begin(),
                                         dims.value().end()};
    dims_ = std::move(dims_vec);
  }
}

bool PartialTensorShape::IsUnranked() const {
  if (dims_.has_value()) {
    return false;
  }
  return true;
}

std::optional<ArrayRef<Index>> PartialTensorShape::GetShape() const {
  if (IsUnranked()) {
    return std::nullopt;
  }
  return llvm::ArrayRef(dims_.value());
}

bool PartialTensorShape::IsShapeKnown() const {
  if (IsUnranked()) {
    return false;
  }
  // TODO(ashwinm): This can be precomputed.
  return std::find_if(dims_->begin(), dims_->end(),
                      PartialTensorShape::IsUnknownDim) == dims_->end();
}

int PartialTensorShape::GetRank() const {
  if (IsUnranked()) {
    return PartialTensorShape::kUnknownDimSize;
  }
  return GetShape().value().size();
}

int64_t PartialTensorShape::GetDimensionSize(int dim_idx) const {
  assert(!IsUnranked() && "GetDim must be called on a ranked tensor shape");
  assert(dim_idx >= 0 && dim_idx < GetRank() &&
         "Index i must be in the range of [0, rank)");

  return (*dims_)[dim_idx];
}

Expected<TensorShape> PartialTensorShape::ToTensorShape() const {
  if (IsShapeKnown()) {
    return TensorShape(dims_.value());
  }

  if (IsUnranked()) {
    return MakeStringError("Unknown rank");
  }

  llvm::SmallVector<Index, 4> unknown_dims;
  for (int i = 0; i < dims_->size(); i++) {
    if (IsUnknownDim(dims_.value()[i])) {
      unknown_dims.push_back(i);
    }
  }
  std::string str;
  llvm::raw_string_ostream os(str);
  os << "[";
  llvm::interleaveComma(unknown_dims, os);
  os << "]";
  return MakeStringError("Unknown dimensions at following indices = ", str);
}

template <size_t Rank>
raw_ostream& operator<<(raw_ostream& os, const FixedRankShape<Rank>& value) {
  os << '[';
  if (value.GetNumElements() > 0) {
    auto it = value.begin();
    os << *it;
    while (++it != value.end()) {
      os << ", " << *it;
    }
  }
  return os << ']';
}

template raw_ostream& operator<<(raw_ostream& os,
                                 const FixedRankShape<0>& value);
template raw_ostream& operator<<(raw_ostream& os,
                                 const FixedRankShape<1>& value);
template raw_ostream& operator<<(raw_ostream& os,
                                 const FixedRankShape<2>& value);
template raw_ostream& operator<<(raw_ostream& os,
                                 const FixedRankShape<3>& value);
template raw_ostream& operator<<(raw_ostream& os,
                                 const FixedRankShape<4>& value);

TensorShape GetFlattenedInnerDimsShape(const TensorShape& shape,
                                       int64_t num_out_dims) {
  llvm::SmallVector<Index, 4> orig_dims(shape.GetRank());
  shape.GetDimensions(&orig_dims);
  return TensorShape(ComputeInnerDims(orig_dims, num_out_dims));
}

}  // namespace tfrt
