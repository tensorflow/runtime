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
include "tfrt/core_runtime/opdefs/corert_base.td"
include "tfrt/tensor/opdefs/tensor.td"
include "tfrt/core_runtime/opdefs/corert_traits.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for corert dialect ops.
class CoreRT_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<CoreRT_Dialect, mnemonic, !listconcat(traits, [IsolatedFromAbove])> {
  // Each registered op in the corert namespace needs to provide all of a
  // verifier, a printer and a parser.
  let verifier = [{ return tfrt::corert::verify(*this); }];
  let printer = [{ return tfrt::corert::print(p, *this); }];
  let parser = [{ return tfrt::corert::parse$cppClass(parser, result); }];
}

class CreateDenseTensorOp<string dtype>
  : CoreRT_Op<"create_dense_tensor." # dtype, [NoSideEffect]> {
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

  let verifier = ?;
}

// Instantiate for each dtype and rank pair.
foreach dtype = ["ui8", "ui16", "ui32", "ui64",
                 "i1", "i8", "i16", "i32", "i64",
                 "bf16", "f16", "f32", "f64",
                 "complex64", "complex128"] in {
  def CoreRT_CreateDenseTensorOp_#dtype
    : CreateDenseTensorOp<dtype>;
}

def ConstDenseTensorOp : CoreRT_Op<"const_dense_tensor",
    [ConstantLike, NoSideEffect, CoreRT_TypedAttributeTrait]> {
  let summary = "Create a constant dense host tensor.";
  let description = [{
    ConstDenseTensorOp creates a constant dense host tensor.

    Example:
      %tensor_handle = corert.const_dense_tensor dense<[0, 1, 2]>: tensor<3xi32>
  }];

  let arguments = (ins
    ElementsAttr:$value
  );

  let results = (outs
    CoreRT_TensorHandleType:$result
  );

  let assemblyFormat = "$value attr-dict";

  let verifier = ?;

  let hasFolder = 1;
}

// TODO(b/181257202): This should be `ConstantLike`.
def ConstStringTensorOp : CoreRT_Op<"const_string_tensor",
    [NoSideEffect, CoreRT_TypedAttributeTrait]> {
  let summary = "Create a constant string host tensor.";
  let description = [{
    ConstStringTensorOp creates a constant string host tensor.

    Example:
      %tensor_handle = corert.const_string_tensor {shape = [1, 2], value = ["const", "string"]}

    Note: if the entries in `value` are all the same, we can just keep one copy.
      In other words, the following two lines are effectively the same.
      %tensor_handle = corert.const_string_tensor {shape = [1, 2], value = ["string", "string"]}
      %tensor_handle = corert.const_string_tensor {shape = [1, 2], value = ["string"]}
  }];

  let arguments = (ins
    I64ArrayAttr:$shape,
    StrArrayAttr:$value
  );

  let results = (outs
    CoreRT_TensorHandleType:$result
  );

  let assemblyFormat = "attr-dict";

  let verifier = ?;
}

def ExecuteOp : CoreRT_Op<"executeop", [NoSideEffect, CoreRT_TypedAttributeTrait]> {
  let summary = "The CoreRuntime ExecuteOp";
  let description = [{
    The ExecuteOp executes an operation on the specified op handler.

    Example:
      %res = corert.executeop(%op_handler) "some.op"(%arg) : 1

    Note that the trailing number indicates the number of results.


    The func attrs and op attrs are placed in two separate regions.

    Example:
      %res = corert.executeop(%op_handler) "some.op"(%arg)
      {attr1 = value, attr2 = value2} {f_attr1 = "fn1", f_attr2 = "fn2"} : 1

    Note that in this case, the first braket contains op attributes other than
    function attributes, and the second braket contains function attributes,
    which represent function names.
  }];

  let arguments = (ins
    CoreRT_OpHandlerType:$op_handler,
    Variadic<CoreRT_TensorHandleType>:$operands,
    ArrayAttr:$op_attrs,
    ArrayAttr:$op_func_attrs,
    StrAttr:$op_name
  );

  let results = (outs
    Variadic<CoreRT_TensorHandleType>:$results
  );

  let hasFolder = 1;

  let extraClassDeclaration = [{
    void getOpAttrs(SmallVectorImpl<std::pair<StringRef, Attribute>>* op_attrs);

    void getOpFuncAttrs(SmallVectorImpl<std::pair<StringRef, Attribute>>* op_func_attrs);
  }];

  let builders = [
    OpBuilder<(ins "TypeRange":$results, "Value":$op_handler,
      "ValueRange":$operands,
      "ArrayRef<std::pair<StringRef, Attribute>>":$op_attrs,
      "ArrayRef<std::pair<StringRef, Attribute>>":$op_func_attrs,
      "StringRef":$op_name)>];
}

def ExecuteOpSeq : CoreRT_Op<"executeop.seq", [CoreRT_TypedAttributeTrait]> {
  let summary = "The sequenced version of CoreRuntime ExecuteOp";
  let description = [{
    The ExecuteOpSeq executes an operation on the specified op handler, similar to
    ExecuteOp. In addition, it takes an in_op_chain and produces an out_op_chain
    that can be used to sequence op executions. ExecuteOpSeq is nonstrict on
    in_op_chain.

    Example:
      %op_ch_out, %res = corert.executeop.seq(%op_handler, %op_ch_in) "some.op"(%arg) : 1

    Note that the trailing number indicates the number of results.

    The func attrs and op attrs are placed in two separate regions.

    Example:
      %op_ch_out, %res = corert.executeop.seq(%op_ch_in) "some.op"(%arg)
      {fn = function_1} {attr1 = value, attr2 = value2} : 1

    Note that in this case, the first braket contains op attributes other than
    function attributes, and the second braket contains function attributes,
    which represent function names.
  }];

  let arguments = (ins
    CoreRT_OpHandlerType:$op_handler,
    TFRT_ChainType:$in_op_chain,
    Variadic<CoreRT_TensorHandleType>:$operands,
    ArrayAttr:$op_attrs,
    ArrayAttr:$op_func_attrs,
    StrAttr:$op_name
  );

  let results = (outs
    TFRT_ChainType:$out_op_chain,
    Variadic<CoreRT_TensorHandleType>:$results
  );
}

def CoreRT_GetOpHandler : CoreRT_Op<"get_op_handler", [NoSideEffect]> {
  let summary = "return the registered op handler";
  let description = [{
    GetOpHandler returns a registered op handler, which can be used by ExecuteOp.

    Example:
      %ch0 = tfrt.new.chain
      %cpu = corert.get_op_handler %ch0 "cpu"
  }];

  let arguments = (ins
    TFRT_ChainType:$in_op_chain,
    StrAttr:$op_handler_name
  );
  let results = (outs
    CoreRT_OpHandlerType:$op_handler
  );
  let assemblyFormat = "$in_op_chain $op_handler_name attr-dict";
  let verifier = ?;
}

def CoreRT_RegisterOpHandlerOp : CoreRT_Op<"register_op_handler"> {
  let summary = "Register a name for an OpHandler.";
  let description = [{
    Register a name for the given OpHandler and make it accessible through
      corert.get_op_handler.

    Example:
      %chain = corert.register_op_handler %op_handler, "my_op_handler0"
  }];

  let arguments = (ins
    CoreRT_OpHandlerType:$op_handler,
    StrAttr:$op_handler_name
  );
  let results = (outs TFRT_ChainType:$out_op_chain);
  let assemblyFormat = "$op_handler $op_handler_name attr-dict";
  let verifier = ?;
}

def CoreRt_CondOp : CoreRT_Op<"cond"> {
  let summary = "conditional operation";
  let description = [{
    The "corert.cond" operation takes a tensorhandle as predicate, a 'true'
    function and a 'false' function.  If the condition is true, the 'true'
    function is called and the "hex.cond" operation returns the results of that
    function, otherwise the "false" function is called and its values are
    returned.

    The 'true' and 'false' functions must have the same signature, but they may
    take an arbitrary number of arguments and return values with the same types
    as the op return types. Both functions are encoded as function attribute,
    named true_fn and false_fn. The a_ and b_ prefix prevent reordering by BEF
    since BEF orders attribute by name alphabetically.

    This operation is safe to use as a 'nonstrict' op, which dispatches its body
    whenever the condition and any dependent arguments are resolved.

    Example:

      %chain, %z = corert.cond %cond, @true_fn, @false_fn (%chain, %x, %y) : (i32)
  }];
  let arguments = (ins CoreRT_TensorHandleType:$cond,
                       FlatSymbolRefAttr:$a_true_fn,
                       FlatSymbolRefAttr:$b_false_fn,
                       TFRT_ChainType:$in_op_chain,
                       Variadic<AnyType>:$fn_operands);
  let results = (outs TFRT_ChainType:$out_op_chain,
                      Variadic<AnyType>:$fn_outputs);
  // TODO(b/161008370): Add bef.nonstrict.
  let assemblyFormat = [{
      $cond $a_true_fn $b_false_fn `(` $in_op_chain (`,` $fn_operands^)? `)` attr-dict `:` `(` type($fn_operands) `)` `->` `(` type($fn_outputs) `)`
    }];
}

def CoreRt_WhileOp : CoreRT_Op<"while"> {
  let summary = "while loop operation";
  let description = [{
    The "corert.while" operation takes a 'Cond' function and a 'Body' function
    and performs the following action:
        fn_output = fn_operands; while(cond_fn(fn_output)) { fn_output = body_fn(fn_output) }

    The 'Cond' and 'Body' functions must have the same input signature, but they
    may take an arbitrary number of arguments. The 'Body' function must return
    values with the same types as the op return types. Both functions are
    encoded as function attribute, named cond_fn and body_fn. The a_ and b_
    prefix prevent reordering by BEF since BEF orders attribute by name
    alphabetically.

    This operation is safe to use as a 'nonstrict' op, which dispatches its body
    whenever the condition and any dependent arguments are resolved.

    Example:

      %chain, %z = corert.while @cond_fn, @body_fn (%chain, %x, %y) : (i32)
  }];
  let arguments = (ins FlatSymbolRefAttr:$a_cond_fn,
                       FlatSymbolRefAttr:$b_body_fn,
                       TFRT_ChainType:$in_op_chain,
                       Variadic<AnyType>:$fn_operands);
  let results = (outs TFRT_ChainType:$out_op_chain,
                      Variadic<AnyType>:$fn_outputs);
  // TODO(b/161008370): Add bef.nonstrict.
  let assemblyFormat = [{
      $a_cond_fn $b_body_fn `(` $in_op_chain (`,` $fn_operands^)? `)` attr-dict `:` `(` type($fn_operands) `)` `->` `(` type($fn_outputs) `)`
    }];
}

// TODO(donglin): remove this kernel after we have implmented the long term
// solution to get dst tensor type in a compiler pass.
def CoreRT_GetDstTensorTypeOp : CoreRT_Op<"get_dst_tensor_type"> {
  let summary = "Get the type of the tensor to be used on the dst_device";
  let description = [{
    Get the type of the tensor to be used on the dst_device.

    Example:
      %result = corert.get_dst_tensor_type %src_tensor_handle, %dst_device
  }];

  let arguments = (ins
    CoreRT_TensorHandleType:$src_tensor_handle,
    TFRT_DeviceType:$dst_device
  );
  let results = (outs TFRT_TensorTypeType:$result);
  let assemblyFormat = "operands attr-dict";
  let verifier = ?;
}

def CoreRT_TransferOp : CoreRT_Op<"transfer"> {
  let summary = "Transfer a TensorHandle to the specified device";
  let description = [{
    Transfer a TensorHandle to the specified device.

    Example:
      %dst_th = corert.transfer %src_th {device="CPU:0", dst_tensor_type_name="DenseHost"}
  }];

  let arguments = (ins
    CoreRT_TensorHandleType:$src,
    TFRT_DeviceType:$device,
    TFRT_TensorTypeType:$dst_tensor_type
  );
  let results = (outs CoreRT_TensorHandleType:$result);
  let assemblyFormat = "operands attr-dict";
  let verifier = ?;
}

def CoreRT_TransferInferredTensortypeOp : CoreRT_Op<"transfer_inferred_tensortype"> {
  let summary = "Transfer a TensorHandle to the specified device";
  let description = [{
    Transfer a TensorHandle to the specified device. Destination tensor type
    is inferred.

    Example:
      %dst_th = corert.transfer_inferred_type %src_th {device="CPU:0"}
  }];

  let arguments = (ins
    CoreRT_TensorHandleType:$src,
    TFRT_DeviceType:$device
  );
  let results = (outs CoreRT_TensorHandleType:$result);
  let assemblyFormat = "operands attr-dict";
  let verifier = ?;
}

def CoreRT_TensorHandleToIntOp : CoreRT_Op<"tensorhandle_to_int32"> {
  let summary = "Cast a TensorHandle to int32.";
  let description = [{
    Cast a TensorHandle to int32.

    Example:
      %one = corert.tensorhandle_to_int32 %src_th
  }];

  let arguments = (ins CoreRT_TensorHandleType:$src);
  let results = (outs I32:$result);
  let assemblyFormat = "operands attr-dict";
  let verifier = ?;
}

#endif
