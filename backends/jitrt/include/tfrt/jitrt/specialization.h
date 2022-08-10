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

#ifndef TFRT_BACKENDS_JITRT_INCLUDE_TFRT_JITRT_SPECIALIZATION_H_
#define TFRT_BACKENDS_JITRT_INCLUDE_TFRT_JITRT_SPECIALIZATION_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "tfrt/jitrt/constraints.h"
#include "tfrt/jitrt/symbolic_shape.h"
#include "tfrt/support/forward_decls.h"
#include "third_party/tensorflow/compiler/xla/runtime/arguments.h"

namespace tfrt {
namespace jitrt {

// TODO(ezhulenev): A lot of specialization code is written with an assumption
// that we can only specialize Tensor arguments. Make this extendable
// to support user-defined types and user-defined specializations.

// Listener class to control notifications during specialization.
struct SpecializationListener {
  virtual ~SpecializationListener() {}

  // Called at the end of module specialization.
  // - 'operands' is a reference to the specialized operands' types.
  // - `attrs` is a list of attributes attached to operands.
  virtual void notifyModuleSpecialized(
      ArrayRef<mlir::Type> operands,
      ArrayRef<mlir::DictionaryAttr> attrs) const {}

  // Called once for every value-specialized argument.
  virtual void notifyValueSpecialized(unsigned index, mlir::Type type,
                                      mlir::Attribute value) const {}
};

// Specializes function to the runtime arguments:
//
// - updates all unknown dimensions according to the resolved symbolic shapes
// - attaches symbolic shape attribute to the operands
// - for value-specialized operands sinks small constants into the function body
//
// Returns error if arguments are not compatible with the function signature.
//
// See an example of a compiled module specialization in `jitrt.h`.
Error SpecializeFunction(
    mlir::func::FuncOp func, ArgumentsRef arguments,
    ArrayRef<SymbolicShapesResolver::SymbolicShape> symbolic_shapes,
    ArrayRef<OperandConstraint> constraints,
    const SpecializationListener* listener = nullptr);

}  // namespace jitrt
}  // namespace tfrt

#endif  // TFRT_BACKENDS_JITRT_INCLUDE_TFRT_JITRT_SPECIALIZATION_H_
