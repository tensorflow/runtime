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

#include <functional>
#include <memory>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "mlir/IR/BuiltinTypes.h"
#include "tfrt/dtype/dtype.h"
#include "tfrt/jitrt/xla.h"

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
//
// If the type can be passed to the compiled function as an argument or returned
// as a result, it must define its own ABI. The ABI is defined by the MLIR to
// LLVM lowering pipeline and the runtime integration (see `runtime.h`).
class Type : public llvm::RTTIExtends<Type, llvm::RTTIRoot> {
 public:
  static constexpr char ID = 0;  // NOLINT

  // Arguments to compiled functions passed as a set of pointers. For example
  // memref descriptor passed in as a set of pointers to data, sizes and
  // strides. See `Argument::Pack` implementation for details (in `argument.h`).
  struct ArgumentAbi {
    size_t num_ptrs;
  };

  // Compiled function returns results by writing into the pre-allocated storage
  // of the given size with the requested alignment. Runtime pre-allocates
  // memory required for all results in the call frame.
  struct ResultAbi {
    size_t size;

    // TODO(ezhulenev): Add alignment to the result ABI. Alignment is an
    // important part of the result ABI that we ignore today. It all doesn't
    // crash only because all results happen to have a size that is multiple of
    // 8 bytes, and because of that all of the results are properly aligned.
    // Results memory layout in the call frame should take in account base
    // pointer alignment and alignment requirements of all results.
  };

  // Returns an Abi if the type can be used as an argument.
  virtual mlir::FailureOr<ArgumentAbi> AsArgument() const { return {}; }

  // Returns an Abi if the type can be returned as a result.
  virtual mlir::FailureOr<ResultAbi> AsResult() const { return {}; }

  virtual raw_ostream& print(raw_ostream& os) const = 0;

 protected:
  Type() = default;
};

inline raw_ostream& operator<<(raw_ostream& os, const Type& type) {
  return type.print(os);
}

//===----------------------------------------------------------------------===//
// Async Token type corresponding to the mlir::async::TokenType
//===----------------------------------------------------------------------===//

class AsyncTokenType : public llvm::RTTIExtends<AsyncTokenType, Type> {
 public:
  static constexpr char ID = 0;  // NOLINT

  mlir::FailureOr<ResultAbi> AsResult() const final;

  raw_ostream& print(raw_ostream& os) const final;
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

  mlir::FailureOr<ResultAbi> AsResult() const final;

  raw_ostream& print(raw_ostream& os) const final;

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

  raw_ostream& print(raw_ostream& os) const final;

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

  raw_ostream& print(raw_ostream& os) const final;

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

  mlir::FailureOr<ArgumentAbi> AsArgument() const final;
  mlir::FailureOr<ResultAbi> AsResult() const final;

  raw_ostream& print(raw_ostream& os) const final;

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

  raw_ostream& print(raw_ostream& os) const final;

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

  mlir::FailureOr<ArgumentAbi> AsArgument() const final;

  raw_ostream& print(raw_ostream& os) const final;
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

  FunctionType(llvm::SmallVector<std::unique_ptr<Type>> operands,
               llvm::SmallVector<std::unique_ptr<Type>> results)
      : operands_(std::move(operands)), results_(std::move(results)) {}

 private:
  llvm::SmallVector<std::unique_ptr<Type>> operands_;
  llvm::SmallVector<std::unique_ptr<Type>> results_;
};

//===----------------------------------------------------------------------===//
// Type conversion from the compile time types to the run time types.
//===----------------------------------------------------------------------===//

// Type converter converts MLIR types known at compile time to the corresponding
// types used at run time. It provides default conversions for the canonical
// types (memrefs, tensors, etc...) and allows users to register custom
// conversions for user-defined types.
class TypeConverter {
 public:
  // Conversion function must return run time type corresponding to the compile
  // time type if the conversion is successful, or `nullptr` if failed.
  using ConversionFn = std::function<std::unique_ptr<Type>(mlir::Type)>;

  // Adds a type conversion function with a type predicate.
  //
  // Example:
  //
  //   AddConversion([](mlir::TensorType) -> std::unique_ptr<Type> { ... });
  //
  // The conversion function will match only the tensor type, and return empty
  // result for all other types, and the type converter will try the next
  // conversion function (see `Convert` implementation).
  template <typename Fn, typename FnTraits = llvm::function_traits<Fn>>
  void AddConversion(Fn fn) {
    using ArgType = typename FnTraits::template arg_t<0>;
    conversions_.emplace_back(
        [fn = std::forward<Fn>(fn)](mlir::Type type) -> std::unique_ptr<Type> {
          if (auto arg = type.dyn_cast<ArgType>()) return fn(arg);
          return {};
        });
  }

  // Converts MLIR element type to the TFRT DType.
  static Expected<DType> ConvertElementType(mlir::Type type);

  // Converts MLIR type to the runtime type. Returns error if conversion was not
  // successful and the type has no corresponding run time type.
  Expected<std::unique_ptr<Type>> Convert(mlir::Type type) const;

  // Converts MLIR function type to the runtime function type. Returns error if
  // function has unsupported operands or results types.
  Expected<FunctionType> Convert(mlir::FunctionType type) const;

 private:
  llvm::SmallVector<ConversionFn> conversions_;
};

}  // namespace jitrt
}  // namespace tfrt

#endif  // TFRT_BACKENDS_JITRT_INCLUDE_TFRT_JITRT_TYPES_H_
