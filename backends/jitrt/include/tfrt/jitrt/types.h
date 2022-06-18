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

#ifndef TFRT_BACKENDS_JITRT_INCLUDE_TFRT_JITRT_TYPES_H_
#define TFRT_BACKENDS_JITRT_INCLUDE_TFRT_JITRT_TYPES_H_

#include <memory>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "mlir/IR/BuiltinTypes.h"
#include "tfrt/dtype/dtype.h"

namespace tfrt {
namespace jitrt {

//===----------------------------------------------------------------------===//
// Canonical JitRt types for the arguments of the compiled kernels.
//===----------------------------------------------------------------------===//

// Types supported by the compiled function signature. We do rely on the LLVM
// style RTTI (https://llvm.org/docs/HowToSetUpLLVMStyleRTTI.html) to avoid
// dependency on the MLIR types at runtime, because we don't want to depend
// on any of the compiler implementation details at runtime and we want to
// support lightweight loading and execution of AOT compiled programs.
//
// We rely on the RTTI for the open class hierarchies, because we want to allow
// users to define their own types for the arguments.
class Type : public llvm::RTTIExtends<Type, llvm::RTTIRoot> {
 public:
  static constexpr char ID = 0;  // NOLINT

 protected:
  Type() = default;
};

raw_ostream& operator<<(raw_ostream& os, const Type& type);

//===----------------------------------------------------------------------===//
// Async Token type corresponding to the mlir::async::TokenType
//===----------------------------------------------------------------------===//

class AsyncTokenType : public llvm::RTTIExtends<AsyncTokenType, Type> {
 public:
  static constexpr char ID = 0;  // NOLINT
};

//===----------------------------------------------------------------------===//
// Async Value type corresponding to the mlir::async::ValueType.
//===----------------------------------------------------------------------===//

class AsyncValueType : public llvm::RTTIExtends<AsyncValueType, Type> {
 public:
  static constexpr char ID = 0;  // NOLINT

  explicit AsyncValueType(std::unique_ptr<Type> value_type)
      : value_type_(std::move(value_type)) {}

  const Type& value_type() const { return *value_type_; }

 private:
  std::unique_ptr<Type> value_type_;
};

//===----------------------------------------------------------------------===//
// Ranked Tensor type corresponding to the mlir::RankedTensorType.
//===----------------------------------------------------------------------===//

class RankedTensorType : public llvm::RTTIExtends<RankedTensorType, Type> {
 public:
  static constexpr char ID = 0;  // NOLINT
  static constexpr int64_t kDynamicSize = mlir::ShapedType::kDynamicSize;

  RankedTensorType(ArrayRef<Index> sizes, DType element_type)
      : sizes_(sizes.begin(), sizes.end()), element_type_(element_type) {}

  ArrayRef<Index> sizes() const { return sizes_; }
  unsigned rank() const { return sizes_.size(); }
  DType element_type() const { return element_type_; }

 private:
  llvm::SmallVector<Index> sizes_;
  DType element_type_;
};

//===----------------------------------------------------------------------===//
// Unranked Tensor type corresponding to the mlir::UnrankedTensorType.
//===----------------------------------------------------------------------===//

class UnrankedTensorType : public llvm::RTTIExtends<UnrankedTensorType, Type> {
 public:
  static constexpr char ID = 0;  // NOLINT

  explicit UnrankedTensorType(DType element_type)
      : element_type_(element_type) {}

  DType element_type() const { return element_type_; }

 private:
  DType element_type_;
};

//===----------------------------------------------------------------------===//
// Ranked Memref type corresponding to the mlir::MemrefType.
//===----------------------------------------------------------------------===//

class MemrefType : public llvm::RTTIExtends<MemrefType, Type> {
 public:
  static constexpr char ID = 0;  // NOLINT
  static constexpr int64_t kDynamicSize = mlir::ShapedType::kDynamicSize;

  MemrefType(ArrayRef<Index> sizes, DType element_type)
      : sizes_(sizes.begin(), sizes.end()), element_type_(element_type) {}

  ArrayRef<Index> sizes() const { return sizes_; }
  unsigned rank() const { return sizes_.size(); }
  DType element_type() const { return element_type_; }

 private:
  llvm::SmallVector<Index> sizes_;
  DType element_type_;
};

//===----------------------------------------------------------------------===//
// Unranked Memref type corresponding to the mlir::UnrankedMemrefType.
//===----------------------------------------------------------------------===//

class UnrankedMemrefType : public llvm::RTTIExtends<UnrankedMemrefType, Type> {
 public:
  static constexpr char ID = 0;  // NOLINT

  explicit UnrankedMemrefType(DType element_type)
      : element_type_(element_type) {}

  DType element_type() const { return element_type_; }

 private:
  DType element_type_;
};

//===----------------------------------------------------------------------===//
// Corresponds to the RT dialect's KernelContextType.
//===----------------------------------------------------------------------===//

class KernelContextOperandType
    : public llvm::RTTIExtends<KernelContextOperandType, Type> {
 public:
  static constexpr char ID = 0;  // NOLINT
};

//===----------------------------------------------------------------------===//
// Compiled function signature type corresponding to the mlir::FunctionType.
//===----------------------------------------------------------------------===//

class FunctionType {
 public:
  const Type* operand(unsigned index) const { return operands_[index].get(); }
  const Type* result(unsigned index) const { return results_[index].get(); }

  unsigned num_operands() const { return operands_.size(); }
  unsigned num_results() const { return results_.size(); }

  // Converts MLIR function type to the runtime function type. Returns error if
  // function has unsupported operands or results types.
  static Expected<FunctionType> Convert(mlir::FunctionType type);

  FunctionType(llvm::SmallVector<std::unique_ptr<Type>> operands,
               llvm::SmallVector<std::unique_ptr<Type>> results)
      : operands_(std::move(operands)), results_(std::move(results)) {}

 private:
  llvm::SmallVector<std::unique_ptr<Type>> operands_;
  llvm::SmallVector<std::unique_ptr<Type>> results_;
};

// Converts MLIR element type to the TFRT DType.
Expected<DType> ConvertElementType(mlir::Type type);

// Converts MLIR type to the corresponding JitRt type.
Expected<std::unique_ptr<Type>> ConvertType(mlir::Type type);

//===----------------------------------------------------------------------===//
// Types for passing compiled kernel arguments and passing back results.
//===----------------------------------------------------------------------===//

// TODO(ezhulenev): Move this to arguments.h and add support for open class
// hierarchy of compiled kernel arguments (virtual base).

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
// Verify that operands types are matching runtime values.
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

#endif  // TFRT_BACKENDS_JITRT_INCLUDE_TFRT_JITRT_TYPES_H_
