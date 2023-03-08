/*
 * Copyright 2020 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// This file defines the TensorShape class.

#ifndef TFRT_TENSOR_TENSOR_SHAPE_H_
#define TFRT_TENSOR_TENSOR_SHAPE_H_

#include <array>
#include <cstdint>
#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {
class KernelRegistry;

/// Register the operations for TensorShape with the specified registry.
void RegisterTensorShapeKernels(KernelRegistry* registry);

// Represents the shape of a Tensor.
//
// A tensor's shape is denoted by its number of dimensions and a size for each
// dimension.  For example, a Tensor represented by a 3 x 4 matrix would have
// a shape of 2-D, [3, 4].  A zero-D tensor is a scalar.
//
// Due to internal storage optimizations, the TensorShape type doesn't provide
// fine grained "set dimension" sorts of accessors.  Manipulations of the shape
// should be done in a temporary SmallVector and then swapped in all at once.
class TensorShape {
 public:
  /// Create a TensorShape with the specified dimensions.
  template <typename T = Index>
  explicit TensorShape(ArrayRef<T> dims);

  template <typename Container>
  explicit TensorShape(const Container& dims)
      : TensorShape(llvm::ArrayRef(dims)) {}

  TensorShape(const TensorShape& rhs);
  TensorShape(TensorShape&& rhs);
  TensorShape& operator=(const TensorShape& rhs);
  TensorShape& operator=(TensorShape&& rhs);
  ~TensorShape();

  bool operator==(const TensorShape& other) const;
  bool operator!=(const TensorShape& other) const;

  // Returns the rank of this TensorShape.  The maximum rank is 255.
  int GetRank() const;

  // Return the total number of elements in this TensorShape.  This is all of
  // the dimensions multiplied together.
  Index GetNumElements() const;

  // Return all of the dimensions in this TensorShape in a way that is easy to
  // process.
  void GetDimensions(llvm::SmallVectorImpl<Index>* result) const;

  // A more optimized version than the above.
  template <size_t N>
  void GetDimensions(Index (&result)[N]) const {
    GetDimensions(MutableArrayRef<Index>(result, N));
  }

  template <size_t N>
  void GetDimensions(std::array<Index, N>* result) const {
    GetDimensions(MutableArrayRef<Index>(result->data(), N));
  }

  void GetDimensions(MutableArrayRef<Index> result) const;

  // Return strides of this TensorShape assuming that the data is in row major
  // layout (the last dimension is contiguous in memory).
  //
  // See address calculation section for details:
  // https://en.wikipedia.org/wiki/Row-_and_column-major_order
  void GetStrides(llvm::SmallVectorImpl<Index>* result) const;

  template <size_t N>
  void GetStrides(Index (&result)[N]) const {
    GetStrides(MutableArrayRef<Index>(result, N));
  }

  template <size_t N>
  void GetStrides(std::array<Index, N>* result) const {
    GetStrides(MutableArrayRef<Index>(result->data(), N));
  }

  void GetStrides(MutableArrayRef<Index> result) const;

  // `dim_idx` must be less than GetRank().
  Index GetDimensionSize(int dim_idx) const;

 private:
  // The storage of TensorShape is carefully laid out to always be 16-bytes in
  // size, but has to support the full generality of tensor shapes.  To do this,
  // it has two inline representations, one that can hold up to 7 dimensions
  // when they fit into 16-bits (each dimension is at most 65535 in size) or up
  // to 4 dimension where the first three fits in 32-bits and the last fits in
  // 16 bits.  If neither of these representations work, then an out of line
  // representation is used for the general case.
  // Important: Identical shapes must have the same representation kind. For
  // Rep16 and Rep32, the representation value is assumed to be deterministic.
  // This means for Rep16 and Rep32 it is sufficient to compare the memory
  // blocks to determine if two shapes are identical.
  enum class RepKind : uint8_t { kRep16, kRep32, kRepExternal };

  struct Rep16 {
    uint16_t dims[7];
    RepKind kind;
    uint8_t rank;
  };
  struct Rep32 {
    uint32_t dims[3];
    uint16_t dim3;
    RepKind kind;
    uint8_t rank;
  };

  struct RepExternal {
    size_t* dims;

    // FIXME: This isn't correct for big endian systems.  static_asserts should
    // catch this below.
    uint8_t padding[14 - sizeof(void*)];
    RepKind kind;
    uint8_t rank;
  };

  union {
    Rep16 rep16;
    Rep32 rep32;
    RepExternal rep_external;
  } representation_;

  // Return the storage representation for this TensorShape.
  RepKind GetRepresentationKind() const;

  bool IsRepresentationExternal() const {
    return GetRepresentationKind() == RepKind::kRepExternal;
  }
};

// Represents the shape of a tensor when the shape is known at C++ compile time.
template <size_t Rank>
class FixedRankShape {
 public:
  static constexpr size_t kRank = Rank;
  using value_type = Index;
  using Dims = std::array<value_type, Rank>;
  using const_iterator = typename Dims::const_iterator;
  using iterator = typename Dims::iterator;

  FixedRankShape() { dims_.fill(0); }
  explicit FixedRankShape(const Dims& dims) : dims_(dims) {}
  explicit FixedRankShape(const TensorShape& shape);

  bool operator==(const FixedRankShape& other) const {
    return dims_ == other.dims_;
  }

  bool operator!=(const FixedRankShape& other) const {
    return !(*this == other);
  }

  size_t GetNumElements() const;

  const Index& operator[](size_t i) const { return dims_[i]; }
  Index& operator[](size_t i) { return dims_[i]; }

  const_iterator begin() const { return dims_.begin(); }
  const_iterator end() const { return dims_.end(); }
  iterator begin() { return dims_.begin(); }
  iterator end() { return dims_.end(); }

  TensorShape ToTensorShape() const { return TensorShape(dims_); }

 private:
  std::array<Index, Rank> dims_;
};

// Represents the shape of a tensor whose rank can either be unknown or known
// but some dimensions may be unknown.
// This type is intended for writing kernels that model shape
// computations. Any tensor shape that eventually should be executed by the
// runtime, must be converted to a TensorShape, if it's shape is fully known,
// otherwise it is not a valid shape for the runtime and an error will be
// returned while trying to convert it.
// This represents the RankedTensorShape/UnrankedTensorShape in the MLIR
// standard types.
class PartialTensorShape {
 public:
  // Create a PartialTensorShape with the dimensions. If rank itself is unknown,
  // (dims is std::nullopt), this is an unranked. Else, it is ranked where each
  // dimensions could still be unknown (indicated by kUnknownDimSize) as well.
  explicit PartialTensorShape(std::optional<ArrayRef<Index>> dims);

  // Returns the shape of the tensor.
  // If unranked, return std::nullopt
  // If ranked, return dimensions (including kUnknownDimSize for unknown dim).
  std::optional<ArrayRef<Index>> GetShape() const;

  // Returns true if the rank is unknown. Else, returns false.
  bool IsUnranked() const;

  // Returns the rank if rank is known. If unknown, returns kUnknownDimSize.
  // The maximum rank is 255. A scalar returns a 0.
  int GetRank() const;

  // Returns the dimension at the specified index. `dim_idx` must be in range:
  // [0, GetRank()) and the shape must be ranked.
  int64_t GetDimensionSize(int dim_idx) const;

  // If unknown rank or any dimension has unknown size (< 0), it doesn't have
  // known shape. If true, this shape can be converted to a
  // TensorShape.
  bool IsShapeKnown() const;

  // Convert to a TensorShape if all dimensions are known, else return error
  // indicating all such unknown dimensions.
  Expected<TensorShape> ToTensorShape() const;

  static constexpr int64_t kUnknownDimSize = -1;
  static bool IsUnknownDim(int64_t dim) { return dim == kUnknownDimSize; }

 private:
  // We store dims in SmallVector here since PartialTensorShape is designed
  // for use in shape computations where we could alter the shape by adding/
  // removing dimensions.
  std::optional<llvm::SmallVector<Index, 4>> dims_;
};

//
// Implementation details follow.
//

static_assert(sizeof(TensorShape) == 16, "TensorShape should not grow");

raw_ostream& operator<<(raw_ostream& os, const TensorShape& value);

template <typename T>
inline TensorShape::TensorShape(ArrayRef<T> dims) {
  assert(dims.size() < 256 && "Can only handle rank up to 255");
  auto rank = static_cast<uint8_t>(dims.size());

  // We zero-initialize to ensure the representation value is determinsitic.
  memset(&representation_, 0, sizeof(representation_));

  // This code aims to make the common rep16 case perform the fewest comparisons
  // by speculating it will be fine.  It is slightly slower for rep32, and makes
  // the worst case scenario do the most comparisons.
  size_t next_dim = 0;

  // Scan all the dims breaking out if a dim is found larger than 16-bit.
  while (true) {
    if (next_dim == rank) {
      // Ok, all the dims fit in 16-bits.  We can use the Rep16 format if we
      // have 7 or fewer dims.
      if (rank > 7) break;

      for (size_t i = 0; i != rank; ++i) {
        representation_.rep16.dims[i] = uint16_t(dims[i]);
      }
      representation_.rep16.kind = RepKind::kRep16;
      representation_.rep16.rank = rank;
      return;
    }

    // If this dimension fits in 16 bits, then keep scanning.
    auto this_dim = dims[next_dim];
    if (uint16_t(this_dim) != this_dim) break;
    ++next_dim;
  }

  // Okay, we found a dimension too big to fit into rep16 - check for rep32.
  while (true) {
    if (next_dim == rank) {
      // Ok, all the dims fit in 32-bits.  We can use the Rep32 format if we
      // have 4 or fewer dims and if the last dim fits in 16-bits.
      if (rank > 4 || (rank == 4 && uint16_t(dims[3]) != dims[3])) break;
      representation_.rep32.rank = rank;
      representation_.rep32.kind = RepKind::kRep32;
      switch (rank) {
        case 4:
          representation_.rep32.dim3 = uint16_t(dims[3]);
          LLVM_FALLTHROUGH;
        case 3:
          representation_.rep32.dims[2] = dims[2];
          LLVM_FALLTHROUGH;
        case 2:
          representation_.rep32.dims[1] = dims[1];
          LLVM_FALLTHROUGH;
        case 1:
          representation_.rep32.dims[0] = dims[0];
          break;
        default:
          assert(0 && "unreachable");
      }
      return;
    }

    // If this dimension fits in 32 bits, then keep scanning.
    auto this_dim = dims[next_dim];
    if (uint32_t(this_dim) != this_dim) break;
    ++next_dim;
  }

  // Otherwise, nothing fits, use the most general representation.
  auto* elts = new size_t[rank];
  memcpy(elts, dims.data(), sizeof(size_t) * rank);
  representation_.rep_external.dims = elts;
  representation_.rep_external.rank = rank;
  representation_.rep_external.kind = RepKind::kRepExternal;
}

// Return the storage representation for this TensorShape.
inline TensorShape::RepKind TensorShape::GetRepresentationKind() const {
  static_assert(offsetof(Rep16, kind) == offsetof(Rep32, kind) &&
                    offsetof(Rep16, kind) == offsetof(RepExternal, kind),
                "Layout mismatch inside of TensorShape");
  // Because all of the representations store their kind in the same place, we
  // can just access an arbitrary one.

  // NOTE: This is technically undefined behavior, because we're accessing a
  // union through a different rep than it was potentially initialized with.
  // This is implementation-specified behavior for most compilers, but if it
  // becomes a problem we can reimplement this with `char*` and offsetof given
  // that char* is allowed to alias anything.
  return representation_.rep16.kind;
}

// Returns the rank of this TensorShape.  The maximum rank is 255.
inline int TensorShape::GetRank() const {
  static_assert(offsetof(Rep16, rank) == offsetof(Rep32, rank) &&
                    offsetof(Rep16, rank) == offsetof(RepExternal, rank),
                "Layout mismatch inside of TensorShape");
  // Because all of the representations store their rank in the same place, we
  // can just access an arbitrary one.
  return representation_.rep16.rank;
}

inline TensorShape::TensorShape(const TensorShape& rhs) {
  memcpy(&representation_, &rhs.representation_, sizeof(representation_));
  if (rhs.IsRepresentationExternal()) {
    auto rank = GetRank();
    representation_.rep_external.dims = new size_t[rank];
    memcpy(representation_.rep_external.dims,
           rhs.representation_.rep_external.dims, rank * sizeof(size_t));
  }
}

inline TensorShape::TensorShape(TensorShape&& rhs) {
  memcpy(&representation_, &rhs.representation_, sizeof(representation_));

  // We're taking the memory from the RHS, reset it to something valid.
  if (rhs.IsRepresentationExternal()) {
    rhs.representation_.rep16.kind = RepKind::kRep16;
    rhs.representation_.rep16.rank = 0;
  }
}

inline TensorShape& TensorShape::operator=(const TensorShape& rhs) {
  if (IsRepresentationExternal()) delete[] representation_.rep_external.dims;

  memcpy(&representation_, &rhs.representation_, sizeof(representation_));
  if (rhs.IsRepresentationExternal()) {
    auto rank = GetRank();
    representation_.rep_external.dims = new size_t[rank];
    memcpy(representation_.rep_external.dims,
           rhs.representation_.rep_external.dims, rank * sizeof(size_t));
  }
  return *this;
}

inline TensorShape& TensorShape::operator=(TensorShape&& rhs) {
  if (IsRepresentationExternal()) delete[] representation_.rep_external.dims;

  memcpy(&representation_, &rhs.representation_, sizeof(representation_));

  // We're taking the memory from the RHS, reset it to something valid.
  if (rhs.IsRepresentationExternal()) {
    rhs.representation_.rep16.kind = RepKind::kRep16;
    rhs.representation_.rep16.rank = 0;
  }
  return *this;
}

inline TensorShape::~TensorShape() {
  if (IsRepresentationExternal()) delete[] representation_.rep_external.dims;
}

template <size_t Rank>
size_t FixedRankShape<Rank>::GetNumElements() const {
  std::size_t result = 1;
  for (size_t d : dims_) {
    result *= d;
  }
  return result;
}

template <size_t Rank>
FixedRankShape<Rank>::FixedRankShape(const TensorShape& shape) {
  shape.GetDimensions(&dims_);
}

template <size_t Rank>
raw_ostream& operator<<(raw_ostream& os, const FixedRankShape<Rank>& value);

raw_ostream& operator<<(raw_ostream& os, const PartialTensorShape& value);

// Returns a TensorShape with rank `num_out_dims`. The returned shape is
// obtained by collapsing all dimensions in `shape` except the last
// `num_out_dims` - 1 dimensions into the first dimension. Logic based on:
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.cc#L1278
TensorShape GetFlattenedInnerDimsShape(const TensorShape& shape,
                                       int64_t num_out_dims);

}  // namespace tfrt

#endif  // TFRT_TENSOR_TENSOR_SHAPE_H_
