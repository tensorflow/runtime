/*
 * Copyright 2022 The TensorFlow Runtime Authors
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

#ifndef TFRT_BACKENDS_JITRT_INCLUDE_TFRT_JITRT_ARGUMENTS_H_
#define TFRT_BACKENDS_JITRT_INCLUDE_TFRT_JITRT_ARGUMENTS_H_

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "tfrt/dtype/dtype.h"
#include "tfrt/jitrt/types.h"

namespace tfrt {
namespace jitrt {

//===----------------------------------------------------------------------===//
// Canonical types for passing compiled kernel arguments.
//===----------------------------------------------------------------------===//

class MemrefDesc {
 public:
  MemrefDesc(DType dtype, void* data, Index offset, ArrayRef<Index> sizes,
             ArrayRef<Index> strides)
      : rank_(sizes.size()), dtype_(dtype), data_(data), offset_(offset) {
    assert(sizes.size() == strides.size() && "invalid sizes and strides pair");
    sizes_and_strides_.reserve(2 * rank_);
    sizes_and_strides_.append(sizes.begin(), sizes.end());
    sizes_and_strides_.append(strides.begin(), strides.end());
  }

  // Constructs MemrefDesc of the given rank and calls `Fill` callback to
  // initialize sizes and strides.
  //
  // Expected callback signature: void fill(MutableArrayRef<Index> sizes,
  //                                        MutableArrayRef<Index> strides);
  //
  // We pass the fill callback as a template argument to be able to inline it
  // at the call site, because MemrefDesc construction is on a hot path.
  template <typename Fill>
  MemrefDesc(unsigned rank, DType dtype, void* data, Index offset, Fill fill);

  // Ensure that MemrefDesc is always moved around instead of copying.
  MemrefDesc(const MemrefDesc&) = delete;
  MemrefDesc& operator=(const MemrefDesc&) = delete;
  MemrefDesc(MemrefDesc&&) = default;
  MemrefDesc& operator=(MemrefDesc&&) = default;

  unsigned rank() const { return rank_; }
  DType dtype() const { return dtype_; }

  // IMPORTANT: Arguments are passed to compiled kernels as pointers to values,
  // for this reason every method that is used in
  // `Executable::InitializeCallFrame` returns a reference to data member, so we
  // don't accidentally pass pointers to temporaries.

  void* const& data() const { return data_; }
  const Index& offset() const { return offset_; }

  const Index& size(size_t index) const { return sizes_and_strides_[index]; }
  const Index& stride(size_t index) const {
    return sizes_and_strides_[rank_ + index];
  }

  ArrayRef<Index> sizes() const { return {sizes_and_strides_.data(), rank_}; }
  ArrayRef<Index> strides() const {
    return {sizes_and_strides_.data() + rank_, rank_};
  }

 private:
  unsigned rank_;
  DType dtype_;
  void* data_;
  Index offset_;
  // We keep sizes and strides in a single container to save one potential
  // memory allocation for memrefs of higher ranks, and to save one vector
  // constructor/destructor call.
  llvm::SmallVector<Index, 8> sizes_and_strides_;
};

template <typename Fill>
MemrefDesc::MemrefDesc(unsigned rank, DType dtype, void* data, Index offset,
                       Fill fill)
    : rank_(rank), dtype_(dtype), data_(data), offset_(offset) {
  sizes_and_strides_.resize(2 * rank_);
  llvm::MutableArrayRef<Index> ref = sizes_and_strides_;
  fill(ref.drop_back(rank_), ref.drop_front(rank_));
}

raw_ostream& operator<<(raw_ostream& os, const MemrefDesc& desc);

//===----------------------------------------------------------------------===//
// Verify that operands types are matching runtime arguments.
//===----------------------------------------------------------------------===//

// We pass operand index to all verification functions to get a user-friendly
// error messages in case of an error.

Error VerifyMemrefOperand(unsigned index, DType element_type,
                          Optional<ArrayRef<Index>> sizes,
                          const MemrefDesc& memref);

Error VerifyMemrefOperand(unsigned index, const RankedTensorType& type,
                          const MemrefDesc& memref);

Error VerifyMemrefOperand(unsigned index, const MemrefType& type,
                          const MemrefDesc& memref);

Error VerifyMemrefOperand(unsigned index, mlir::ShapedType type,
                          const MemrefDesc& memref);

}  // namespace jitrt
}  // namespace tfrt

#endif  // TFRT_BACKENDS_JITRT_INCLUDE_TFRT_JITRT_ARGUMENTS_H_
