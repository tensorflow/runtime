// Copyright 2021 The TensorFlow Runtime Authors
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

#ifndef TFRT_TRAITS
#define TFRT_TRAITS

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TFRT Traits corresponding to the Cost Function (see tfrt_op_interfaces.td).
//===----------------------------------------------------------------------===//

// The cost of executing an operation is fixed and independent of the
// operation operands or attributes (e.g. simple metadata manipulation).
class TFRT_FixedCost<int value>
  : NativeOpTrait<!strconcat("tfrt::FixedCostTrait<", !cast<string>(value),
                             ">::Impl")>;

// The cost of executing operation specified by a special attribute attached to
// the operation.
def TFRT_AttrCostTrait : NativeOpTrait<"tfrt::AttrCostTrait">;

#endif // TFRT_TRAITS
