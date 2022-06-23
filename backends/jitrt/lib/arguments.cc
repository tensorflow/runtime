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

//===- arguments.cc - -----------------------------------------------------===//
// Canonical types for passing compiled kernel arguments.
//===----------------------------------------------------------------------===//

#include "tfrt/jitrt/arguments.h"

#include <string>

#include "tfrt/support/error_util.h"

namespace tfrt {
namespace jitrt {

raw_ostream& OpaqueArg::print(raw_ostream& os) const {
  return os << "OpaqueArg: ptr=" << ptr_;
}

raw_ostream& MemrefDesc::print(raw_ostream& os) const {
  auto print_arr = [&](string_view name, ArrayRef<Index> arr) {
    os << " " << name << ": [";
    if (!arr.empty()) {
      os << arr[0];
      for (int i = 1; i < arr.size(); ++i) os << ", " << arr[i];
    }
    os << "]";
  };

  os << "MemrefDesc: dtype: " << dtype() << " offset: " << offset();
  print_arr("sizes", sizes());
  print_arr("strides", strides());

  return os;
}

// -------------------------------------------------------------------------- //
// Verify that operands types are matching runtime values.
// -------------------------------------------------------------------------- //

static bool AreCompatibleTypes(DType type1, DType type2) {
  auto compatible = [&](DType fromType, DType toType) {
    return (type1 == fromType && type2 == toType) ||
           (type1 == toType && type2 == fromType);
  };
  // I1 and I8 types are compatible since they both are 1-byte size at runtime.
  if (compatible(DType::I1, DType::I8)) return true;

  // Signed and unsigned integers of the same size are compatible in memory.
  if (compatible(DType::I8, DType::UI8) ||
      compatible(DType::I16, DType::UI16) ||
      compatible(DType::I32, DType::UI32) ||
      compatible(DType::I64, DType::UI64))
    return true;

  return type1 == type2;
}

Error VerifyMemrefOperand(unsigned index, DType element_type,
                          Optional<ArrayRef<Index>> sizes,
                          const MemrefDesc& memref) {
  // Format memref operand and expected type for user-friendly error messages.
  auto format_operands = [&]() -> std::string {
    std::string err;
    llvm::raw_string_ostream os(err);

    auto dim = [](Index d) -> std::string {
      return d == MemrefType::kDynamicSize ? "?" : std::to_string(d);
    };

    auto print_shaped = [&](Optional<ArrayRef<Index>> dims, DType dtype) {
      if (!dims.hasValue()) {
        os << "[*x" << dtype << "]";
        return;
      }

      if (dims->empty()) {
        os << "[" << dtype << "]";
        return;
      }

      os << "[" << dim((*dims)[0]);
      for (int i = 1; i < dims->size(); ++i) os << "x" << dim((*dims)[i]);
      os << "x" << dtype << "]";
    };

    os << "got ";
    print_shaped({memref.sizes()}, memref.dtype());
    os << " vs expected ";
    print_shaped(sizes, element_type);

    return err;
  };

  // Check that memref data type is compatible with the operand element type.
  if (LLVM_UNLIKELY(!AreCompatibleTypes(element_type, memref.dtype()))) {
    return MakeStringError(
        "operand #", index,
        " type is not compatible with the expected element type: ",
        memref.dtype(), " vs ", element_type, " (", format_operands(), ")");
  }

  // Skip sizes verification if they are not available.
  if (!sizes.hasValue()) return Error::success();

  // Check that memref rank is the same as operand rank.
  if (LLVM_UNLIKELY(memref.rank() != sizes->size()))
    return MakeStringError(
        "operand #", index,
        " rank does not match expected input rank: ", memref.rank(), " vs ",
        sizes->size(), " (", format_operands(), ")");

  // Check that all statically known dimensions matches the memref dimensions.
  for (const auto& pair : llvm::enumerate(llvm::zip(memref.sizes(), *sizes))) {
    Index operand_dim = std::get<0>(pair.value());
    Index expected_dim = std::get<1>(pair.value());

    bool is_dynamic_dim = mlir::ShapedType::isDynamic(expected_dim);

    if (LLVM_UNLIKELY(operand_dim != expected_dim && !is_dynamic_dim))
      return MakeStringError(
          "operand #", index, " dimension #", pair.index(),
          " does not match expected input dimension: ", operand_dim, " vs ",
          expected_dim, " (", format_operands(), ")");
  }

  return Error::success();
}

Error VerifyMemrefOperand(unsigned index, const RankedTensorType& type,
                          const MemrefDesc& memref) {
  return VerifyMemrefOperand(index, type.element_type(), type.sizes(), memref);
}

Error VerifyMemrefOperand(unsigned index, const MemrefType& type,
                          const MemrefDesc& memref) {
  return VerifyMemrefOperand(index, type.element_type(), type.sizes(), memref);
}

Error VerifyMemrefOperand(unsigned index, mlir::ShapedType type,
                          const MemrefDesc& memref) {
  auto element_type = ConvertElementType(type.getElementType());
  if (auto err = element_type.takeError()) return err;

  // We do not support unranked memrefs at runtime, however we need to verify
  // operand types when we do compiled kernel specialization to shape.
  return VerifyMemrefOperand(
      index, *element_type,
      type.hasRank() ? Optional<ArrayRef<Index>>{type.getShape()} : llvm::None,
      memref);
}

}  // namespace jitrt
}  // namespace tfrt
