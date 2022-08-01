/*
 * Copyright 2021 The TensorFlow Runtime Authors
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

#ifndef TFRT_BACKENDS_JITRT_CONVERSION_RT_PASSES_H_
#define TFRT_BACKENDS_JITRT_CONVERSION_RT_PASSES_H_

#include <functional>
#include <memory>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tfrt/jitrt/conversion/custom_call_to_llvm.h"

namespace tfrt {
namespace jitrt {

// Extension points for converting `rt` dialect to the LLVM dialect.
//
// Runtime custom calls is an extension mechanism for enabling compiled programs
// to call into the APIs provided by the user. It relies on converting
// values and attributes to the LLVM types (structs and pointers) with a
// well-defined memory layout, so that they can be passed across the function
// boundary and safely decoded (without dependency on C++ ABI).
//
// All user-defined types (values and attributes) that are passed to the custom
// calls must define the argument or attribute encoding.
struct ConvertRuntimeToLLvmOpts {
  // Add type conversions for user-defined types to the corresponding LLVM
  // types. Conversion pass uses these extra conversions to convert arguments
  // of the entrypoint function and values passed to the custom calls. Custom
  // call argument encoding can further refine how values of LLVM types passed
  // to the custom call handlers by passing custom encoding (see below).
  std::function<void(mlir::TypeConverter&)> populate_type_conversions;

  // Add user-defined arguments encoding to the custom call lowering.
  std::function<void(CustomCallArgEncodingSet&)> populate_arg_encodings;

  // Add user-defined attributes type encoding to the custom call lowering.
  std::function<void(CustomCallAttrEncodingSet&)> populate_attr_encodings;
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateConvertRuntimeToLLVMPass(ConvertRuntimeToLLvmOpts opts = {});

#define GEN_PASS_REGISTRATION
#include "tfrt/jitrt/conversion/rt_gen_passes.h.inc"

}  // namespace jitrt
}  // namespace tfrt

#endif  // TFRT_BACKENDS_JITRT_CONVERSION_RT_PASSES_H_
