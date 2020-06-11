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

#ifdef CORERT_OPS
#else
#define CORERT_OPS

include "tfrt/tfrt_op_base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def CoreRT_Dialect : Dialect {
  let name = "corert";

  let description = [{
    The Core Runtime Dialect.
  }];

  let cppNamespace = "corert";
}

def DeviceType : OpaqueType<"corert", "device", "!corert.device type">;

// Base class for corert dialect ops.
class CoreRT_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<CoreRT_Dialect, mnemonic, !listconcat(traits, [IsolatedFromAbove])> {
  // Each registered op in the corert namespace needs to provide all of a
  // verifier, a printer and a parser.
  let verifier = [{ return tfrt::corert::verify(*this); }];
  let printer = [{ return tfrt::corert::print(p, *this); }];
  let parser = [{ return tfrt::corert::parse$cppClass(parser, result); }];
}

def ConstDenseTensorOp : CoreRT_Op<"const_dense_tensor", [ConstantLike, NoSideEffect]> {
  let summary = "Create a constant host tensor.";
  let description = [{
    ConstDenseTensorOp creates a constant dense host tensor.

    Example:
      %tensor_handle = corert.const_dense_tensor dense<[0, 1, 2]>: tensor<3xi32>
  }];

  let arguments = (ins
    ElementsAttr:$value
  );

  let results = (outs
    TensorHandleType:$result
  );

  let assemblyFormat = "$value attr-dict";

  let verifier = ?;

  let hasFolder = 1;
}

def ConstStringTensorOp : CoreRT_Op<"const_string_tensor", [NoSideEffect]> {
  let summary = "Create a constant string host tensor.";
  let description = [{
    ConstStringTensorOp creates a constant string host tensor.

    Example:
      %tensor_handle = corert.const_string_tensor {shape = [1, 2], value = ["const", "string"]}
  }];

  let arguments = (ins
    I64ArrayAttr:$shape,
    StrArrayAttr:$value
  );

  let results = (outs
    TensorHandleType:$result
  );

  let assemblyFormat = "attr-dict";

  let verifier = ?;
}

def ExecuteOp : CoreRT_Op<"executeop", [NoSideEffect]> {
  let summary = "The CoreRuntime ExecuteOp";
  let description = [{
    The ExecuteOp executes an operation on the specified device.

    Example:
      %res = corert.executeop(%device) "some.op"(%arg) : 1

    Note that the trailing number indicates the number of results.
  }];

  let arguments = (ins
    DeviceType:$device,
    Variadic<TensorHandleType>:$operands,
    ArrayAttr:$op_attrs,
    StrAttr:$op_name
  );

  let results = (outs
    Variadic<TensorHandleType>:$results
  );

  let hasFolder = 1;

  let extraClassDeclaration = [{
    void getOpAttrs(SmallVectorImpl<std::pair<StringRef, Attribute>>* op_attrs);
  }];

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &state, ArrayRef<Type> results, "
    "Value device, ValueRange operands, "
    "ArrayRef<std::pair<StringRef, Attribute>> op_attrs, StringRef op_name"
  >];
}

def ExecuteOpSeq : CoreRT_Op<"executeop.seq"> {
  let summary = "The sequenced version of CoreRuntime ExecuteOp";
  let description = [{
    The ExecuteOpSeq executes an operation on the specified device, similar to
    ExecuteOp. In addition, it takes an in_op_chain and produces an out_op_chain
    that can be used to sequence op executions. ExecuteOpSeq is nonstrict on
    in_op_chain.

    Example:
      %op_ch_out, %res = corert.executeop.seq(%device, %op_ch_in) "some.op"(%arg) : 1

    Note that the trailing number indicates the number of results.
  }];

  let arguments = (ins
    DeviceType:$device,
    ChainType:$in_op_chain,
    Variadic<TensorHandleType>:$operands,
    ArrayAttr:$op_attrs,
    StrAttr:$op_name
  );

  let results = (outs
    ChainType:$out_op_chain,
    Variadic<TensorHandleType>:$results
  );
}

def CoreRT_GetDeviceOp : CoreRT_Op<"get_device"> {
  let summary = "return the registered device";
  let description = [{
    GetDeviceOp returns a registered device, which can be used by ExecuteOp.

    Example:
      %cpu = corert.get_device "cpu"
  }];

  let arguments = (ins StrAttr:$device_name);
  let results = (outs DeviceType:$device);
  let assemblyFormat = "$device_name attr-dict";
  let verifier = ?;
}

def CoreRT_RegisterOpHandlerChainOp : CoreRT_Op<"register_op_handler_chain"> {
  let summary = "Register a name for a Chain of OpHandlers.";
  let description = [{
    Register a name for the given Chain of OpHandlers and make the chain
    accessible through corert.get_device.

    Example:
      %chain = corert.register_op_handler_chain %root_op_handler, "my_chain0"
  }];

  let arguments = (ins
    DeviceType:$device,
    StrAttr:$device_name
  );
  let results = (outs ChainType:$out_op_chain);
  let assemblyFormat = "$device $device_name attr-dict";
  let verifier = ?;
}
#endif
