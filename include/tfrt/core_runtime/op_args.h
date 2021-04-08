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

// This file defines supporting classes for op arguments.

#ifndef TFRT_CORE_RUNTIME_OP_ARGS_H_
#define TFRT_CORE_RUNTIME_OP_ARGS_H_

#include <type_traits>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/PointerIntPair.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {

// OptionalOpArg indicates that an input argument is optional for an op.
//
// Example usage:
//
// 1) Metadata function with optional argument
// static TensorMetadata TestOptionalArgOpMD(
//     const TensorMetadata& input, OptionalOpArg<TensorMetadata> input2) {
//   if (input2) {
//     return *input2;
//   } else {
//     return input;
//   }
// }
//
// 2) Op implementation with optional argument
// static gpu::DenseGpuTensor TestOptionalArgOp(
//     const gpu::DenseGpuTensor& input,
//     OptionalOpArg<gpu::DenseGpuTensor> input2) {
//   if (input2) {
//     return input2->CopyRef();
//   } else {
//     return input.CopyRef();
//   }
// }
template <typename T>
class OptionalOpArg {
 public:
  OptionalOpArg() = default;
  /* implicit */
  OptionalOpArg(const T* value) : value_{value} {}

  const T& operator*() const {
    assert(value_);
    return *value_;
  }
  const T* operator->() const {
    assert(value_);
    return value_;
  }

  const T* get() const { return value_; }

  explicit operator bool() const { return value_ != nullptr; }

 private:
  const T* value_ = nullptr;
};

// VariadicOpArg indicates that an input argument is variadic for an op.
//
// Example usage:
//
// 1) Metadata function with a variadic argument
// static TensorMetadata TestVariadicArgOpMD(
//     const TensorMetadata& input, VariadicOpArg<TensorMetadata> input2) {
//   if (input2.size()) {
//     return input2[0];
//   } else {
//     return input;
//   }
// }
//
// 2) Op implementation with variadic argument
// static gpu::DenseGpuTensor TestVariadicArgOp(
//     const gpu::DenseGpuTensor& input,
//     VariadicOpArg<gpu::DenseGpuTensor> input2) {
//   if (input2.size() > 0) {
//     return input2[0].CopyRef();
//   } else {
//     return input.CopyRef();
//   }
// }
template <typename T>
class VariadicOpArg {
 public:
  explicit VariadicOpArg(ArrayRef<T> args)
      : data_and_type_(args.data(), true), size_(args.size()) {}

  template <typename U>
  explicit VariadicOpArg(ArrayRef<U*> args)
      : data_and_type_(args.data(), false), size_(args.size()) {
    static_assert(static_cast<T*>(static_cast<U*>(nullptr)) == nullptr,
                  "Incompatible type found for VariadicOpArg<T>");
    // Assert that we can convert args to the target type.
    for (U* arg : args) {
      // Note that both T and U need to support LLVM-style RTTI. See
      // http://llvm.org/docs/HowToSetUpLLVMStyleRTTI.html
      // http://llvm.org/docs/ProgrammersManual.html#the-isa-cast-and-dyn-cast-templates
      assert(cast<T>(arg) && "Incompatible arg type");
    }
  }

  size_t size() const { return size_; }
  const T& operator[](size_t i) const {
    const void* ptr = data_and_type_.getPointer();
    if (data_and_type_.getInt()) {
      return *(static_cast<const T*>(ptr) + i);
    } else {
      return **(static_cast<T* const*>(ptr) + i);
    }
  }

 private:
  // Stores T* if the int bit is true, otherwise stores T**.
  llvm::PointerIntPair<const void*, 1, bool> data_and_type_;
  size_t size_;
};

}  // namespace tfrt

#endif  // TFRT_CORE_RUNTIME_OP_ARGS_H_
