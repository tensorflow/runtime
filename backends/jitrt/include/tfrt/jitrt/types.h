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
#include "mlir/IR/BuiltinTypes.h"
#include "tfrt/dtype/dtype.h"

namespace tfrt {
namespace jitrt {

//----------------------------------------------------------------------------//
// Types supported by the compiled function signature. We do rely on the LLVM
// style RTTI (https://llvm.org/docs/HowToSetUpLLVMStyleRTTI.html) to avoid
// dependency on the MLIR types at runtime, because for that we need to carry
// a separate MLIRContext with every instance of Executable which might require
// a lot of memory to hold all the uniqued attributes (large constants).
//----------------------------------------------------------------------------//

class Type {
 public:
  enum class TypeKind {
    kAsyncToken,
    kAsyncValue,
    kRankedTensor,
    kUnrankedTensor,
    kMemref,
    kUnrankedMemref,
    kKernelContext
  };

  virtual ~Type() = default;

  TypeKind kind() const { return kind_; }

 protected:
  explicit Type(TypeKind kind) : kind_(kind) {}

  // Unlike the mlir::Type which itself is a "smart pointer like" type, with the
  // underlying object owned by the MLIR context, the runtime type must be
  // wrapped in a smart pointer explicitly (e.g. in std::unique_ptr) and can't
  // be moved or copied (see the `FunctionType` below for example).
  Type(Type&&) = delete;
  Type(const Type&) = delete;
  Type& operator=(Type&&) = delete;
  Type& operator=(const Type&) = delete;

 private:
  const TypeKind kind_;
};

raw_ostream& operator<<(raw_ostream& os, const Type& type);

// Async Token type corresponding to the mlir::async::TokenType
class AsyncTokenType : public Type {
 public:
  AsyncTokenType() : Type(TypeKind::kAsyncToken) {}

  static bool classof(const Type* type) {
    return type->kind() == TypeKind::kAsyncToken;
  }
};

// Async Value type corresponding to the mlir::async::ValueType.
class AsyncValueType : public Type {
 public:
  explicit AsyncValueType(std::unique_ptr<Type> value_type)
      : Type(TypeKind::kAsyncValue), value_type_(std::move(value_type)) {}

  const Type& value_type() const { return *value_type_; }

  static bool classof(const Type* type) {
    return type->kind() == TypeKind::kAsyncValue;
  }

 private:
  std::unique_ptr<Type> value_type_;
};

// Ranked Tensor type corresponding to the mlir::RankedTensorType.
class RankedTensorType : public Type {
 public:
  static constexpr int64_t kDynamicSize = mlir::ShapedType::kDynamicSize;

  RankedTensorType(ArrayRef<Index> sizes, DType element_type)
      : Type(TypeKind::kRankedTensor),
        sizes_(sizes.begin(), sizes.end()),
        element_type_(element_type) {}

  ArrayRef<Index> sizes() const { return sizes_; }
  unsigned rank() const { return sizes_.size(); }
  DType element_type() const { return element_type_; }

  static bool classof(const Type* type) {
    return type->kind() == TypeKind::kRankedTensor;
  }

 private:
  llvm::SmallVector<Index> sizes_;
  DType element_type_;
};

// Unranked Tensor type corresponding to the mlir::UnrankedTensorType.
class UnrankedTensorType : public Type {
 public:
  explicit UnrankedTensorType(DType element_type)
      : Type(TypeKind::kUnrankedTensor), element_type_(element_type) {}

  DType element_type() const { return element_type_; }

  static bool classof(const Type* type) {
    return type->kind() == TypeKind::kUnrankedTensor;
  }

 private:
  DType element_type_;
};

// Ranked Memref type corresponding to the mlir::MemrefType.
class MemrefType : public Type {
 public:
  static constexpr int64_t kDynamicSize = mlir::ShapedType::kDynamicSize;

  MemrefType(ArrayRef<Index> sizes, DType element_type)
      : Type(TypeKind::kMemref),
        sizes_(sizes.begin(), sizes.end()),
        element_type_(element_type) {}

  ArrayRef<Index> sizes() const { return sizes_; }
  unsigned rank() const { return sizes_.size(); }
  DType element_type() const { return element_type_; }

  static bool classof(const Type* type) {
    return type->kind() == TypeKind::kMemref;
  }

 private:
  llvm::SmallVector<Index> sizes_;
  DType element_type_;
};

// Unranked Memref type corresponding to the mlir::UnrankedMemrefType.
class UnrankedMemrefType : public Type {
 public:
  explicit UnrankedMemrefType(DType element_type)
      : Type(TypeKind::kUnrankedMemref), element_type_(element_type) {}

  DType element_type() const { return element_type_; }

  static bool classof(const Type* type) {
    return type->kind() == TypeKind::kUnrankedMemref;
  }

 private:
  DType element_type_;
};

// Corresponds to the RT dialect's KernelContextType.
class KernelContextOperandType : public Type {
 public:
  KernelContextOperandType() : Type(TypeKind::kKernelContext) {}

  static bool classof(const Type* type) {
    return type->kind() == TypeKind::kKernelContext;
  }
};

// Compiled function signature type corresponding to the mlir::FunctionType.
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

//----------------------------------------------------------------------------//
// Types for passing compiled kernel arguments and passing back results.
//----------------------------------------------------------------------------//

struct MemrefDesc {
  MemrefDesc() = default;

  // Ensure that MemrefDesc is always moved around instead of copying.
  MemrefDesc(const MemrefDesc&) = delete;
  MemrefDesc& operator=(const MemrefDesc&) = delete;
  MemrefDesc(MemrefDesc&&) = default;
  MemrefDesc& operator=(MemrefDesc&&) = default;

  DType dtype;
  void* data;
  Index offset;
  llvm::SmallVector<Index, 4> sizes;
  llvm::SmallVector<Index, 4> strides;
};

raw_ostream& operator<<(raw_ostream& os, const MemrefDesc& desc);

// -------------------------------------------------------------------------- //
// Verify that operands types are matching runtime values.
// -------------------------------------------------------------------------- //

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
