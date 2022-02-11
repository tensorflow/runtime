// Copyright 2020 The TensorFlow Runtime Authors
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

#ifdef CORERT_SYNC_OPS
#else
#define CORERT_SYNC_OPS

include "tfrt/tfrt_op_base.td"
include "tfrt/core_runtime/opdefs/corert_base.td"
include "tfrt/core_runtime/opdefs/corert_traits.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def CoreRTSync_Dialect : Dialect {
  let name = "corert_sync";

  let description = [{
    The Core Runtime Sync Dialect.

    The Core Runtime Sync Dialect is a hook to use Core Runtime op-by-op
    execution in BEF Sync function. It provides corresponding Core Runtime types
    in MLIR and a corert_sync.executeop kernel to invoke op-by-op execution.

    For more details, please refer to documents/tfrt_op_by_op_execution_design.md.
  }];

  let cppNamespace = "::tfrt::corert_sync";

  let emitAccessorPrefix = kEmitAccessorPrefix_Raw;
}

// Base class for corert dialect ops.
class CoreRTSync_Op<string mnemonic, list<Trait> traits = []> :
    Op<CoreRTSync_Dialect, mnemonic, !listconcat(traits, [IsolatedFromAbove])> {
  // Each registered op in the corert namespace needs to provide all of a
  // verifier, a printer and a parser.
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

class CreateDenseTensorOp<string dtype>
  : CoreRTSync_Op<"create_dense_tensor." # dtype, [NoSideEffect]> {
  let summary = "Create a constant dense host tensor with explicit dtype and rank.";
  let description = [{}];

  let arguments = (ins
    I64ArrayAttr:$shape,
    ArrayAttr:$value
  );

  let results = (outs
    CoreRT_TensorHandleType:$result
  );

  let assemblyFormat = "attr-dict";

  let hasVerifier = 0;
}

// Instantiate for each dtype and rank pair.
foreach dtype = ["ui8", "ui16", "ui32", "ui64",
                 "i1", "i8", "i16", "i32", "i64",
                 "bf16", "f16", "f32", "f64",
                 "complex64", "complex128"] in {
  def CoreRTSync_CreateDenseTensorOp_#dtype
    : CreateDenseTensorOp<dtype>;
}

def ConstDenseTensorOp : CoreRTSync_Op<"const_dense_tensor",
    [ConstantLike, NoSideEffect, CoreRT_TypedAttributeTrait]> {
  let summary = "Create a constant host tensor.";
  let description = [{
    ConstDenseTensorOp creates a constant dense host tensor.

    Example:
      %tensor_handle = corert_sync.const_dense_tensor dense<[0, 1, 2]>: tensor<3xi32>
  }];

  let arguments = (ins
    ElementsAttr:$value
  );

  let results = (outs
    CoreRT_TensorHandleType:$result
  );

  let assemblyFormat = "$value attr-dict";

  let hasVerifier = 0;

  let hasFolder = 1;
}

def ExecuteOp : CoreRTSync_Op<"executeop", [NoSideEffect, CoreRT_TypedAttributeTrait]> {
  let summary = "The CoreRuntime ExecuteOp";
  let description = [{
    The ExecuteOp executes an operation on the specified device.

    Example:
      %res = corert_sync.executeop(%op_handler) "some.op"(%arg) : 1

    Note that the trailing number indicates the number of results.
  }];

  let arguments = (ins
    CoreRT_OpHandlerType:$op_handler,
    Variadic<CoreRT_TensorHandleType>:$operands,
    ArrayAttr:$op_attrs,
    StrAttr:$op_name
  );

  let results = (outs
    Variadic<CoreRT_TensorHandleType>:$results
  );

  let hasFolder = 1;

  let extraClassDeclaration = [{
    void getOpAttrs(SmallVectorImpl<std::pair<StringRef, Attribute>>* op_attrs);
  }];

  let builders = [
    OpBuilder<(ins "TypeRange":$results, "Value":$device,
      "ValueRange":$operands,
      "ArrayRef<std::pair<StringRef, Attribute>>":$op_attrs,
      "StringRef":$op_name)>];
}

def CoreRTSync_GetOpHandler : CoreRTSync_Op<"get_op_handler", [NoSideEffect]> {
  let summary = "return the registered op handler";
  let description = [{
    GetOpHandler returns a registered op handler, which can be used by ExecuteOp.

    Example:
      %cpu = corert_sync.get_op_handler "cpu"
  }];

  let arguments = (ins
    StrAttr:$device_name
  );
  let results = (outs
    CoreRT_OpHandlerType:$op_handler
  );
  let assemblyFormat = "$device_name attr-dict";
  let hasVerifier = 0;
}

#endif
