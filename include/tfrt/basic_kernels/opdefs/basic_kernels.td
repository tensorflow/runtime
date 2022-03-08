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

//===- basic_kernels.td ---------------------------------------------------===//
//
// Operation definitions for basic_kernels.
//
//===----------------------------------------------------------------------===//

#ifdef BASIC_OPS
#else
#define BASIC_OPS

include "tfrt/basic_kernels/opdefs/tfrt_base.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

// Base class for TFRT dialect ops.
class TFRT_Op<string mnemonic, list<Trait> traits = []> :
    Op<TFRT_Dialect, mnemonic, !listconcat(traits, [IsolatedFromAbove])> {

  // Each registered op in the TFRT namespace needs to provide all of a printer,
  // parser and verifier.
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def CallOp : TFRT_Op<"call", [CallOpInterface]> {
  let summary = "host executor call operation";
  let description = [{
    The "tfrt.call" operation represents a direct call to a function.  The
    operands and result types of the call must match the specified function
    type.  The callee is encoded as a function attribute named "callee".

      %2 = tfrt.call @my_add(%0, %1) : (f32, f32) -> f32
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let extraClassDeclaration = [{
    StringRef getCallee() { return callee(); }
    ::mlir::FunctionType getCalleeType();

    // The following methods are needed by CallOpInterface.

    // Gets the argument operands to the called function.
    ::mlir::Operation::operand_range getArgOperands() { return operands(); }
    // Returns the callee of this operation.
    ::mlir::CallInterfaceCallable getCallableForCallee() { return calleeAttr(); }
  }];
}

class ConstantOp<string suffix, Type baseType, Attr attr>
  : TFRT_Op<"constant." # suffix, [NoSideEffect]> {
  let summary = "host executor constant value constructor";

  let arguments = (ins attr:$value);
  let results = (outs baseType);
  let assemblyFormat = "$value attr-dict";
  let hasVerifier = 0;
}

// MLIR's OpBase.td defines F32Attr and F64Attr, but no F16Attr or BF16Attr
def F16Attr : FloatAttrBase<F16, "16-bit float attribute">;
def BF16Attr : TypedAttrBase<
  BF16, "FloatAttr",
  And<[CPred<"$_self.isa<::mlir::FloatAttr>()">,
       CPred<"$_self.cast<::mlir::FloatAttr>().getType().isBF16()">]>,
  "bfloat16 attribute"> {
  let returnType = [{ ::llvm::APFloat }];
}

def ConstantI1Op  : ConstantOp<"i1", I1, I1Attr>;
def ConstantI32Op : ConstantOp<"i32", I32, I32Attr>;
def ConstantI64Op : ConstantOp<"i64", I64, I64Attr>;
def ConstantBF16Op : ConstantOp<"bf16", BF16, BF16Attr>;
def ConstantF16Op : ConstantOp<"f16", F16, F16Attr>;
def ConstantF32Op : ConstantOp<"f32", F32, F32Attr>;
def ConstantF64Op : ConstantOp<"f64", F64, F64Attr>;
def ConstantUI32Op : ConstantOp<"ui32", UI32, UI32Attr>;
def ConstantUI64Op : ConstantOp<"ui64", UI64, UI64Attr>;

class ConstantComplexOp<string suffix, Type baseType, Attr attr>
  : TFRT_Op<"constant.complex" # suffix, [NoSideEffect]> {
  let summary = "host executor complex constant value constructor";

  let arguments = (ins attr:$real, attr:$imaginary);
  let results = (outs Complex<baseType>);
  let assemblyFormat = "$real `,` $imaginary attr-dict";
  let hasVerifier = 0;
}

def ConstantComplexF32Op : ConstantComplexOp<"64", F32, F32Attr>;
def ConstantComplexF64Op : ConstantComplexOp<"128", F64, F64Attr>;


// TODO(tf-runtime-team): Consider reusing op defitions for sync/async ops.
// TODO(tf-runtime-team): Consider create `tfrt_sync` dialect.
class SyncConstantOp<string suffix, Type baseType, Attr attr>
  : TFRT_Op<"constant_s." # suffix, [NoSideEffect]> {
  let summary = "host executor constant value constructor";

  let arguments = (ins attr:$value);
  let results = (outs baseType);
  let assemblyFormat = "$value attr-dict";
  let hasVerifier = 0;
}

def SyncConstantI64Op : SyncConstantOp<"i64", I64, I64Attr>;
def SyncConstantF16Op : SyncConstantOp<"f16", F16, F16Attr>;
def SyncConstantF32Op : SyncConstantOp<"f32", F32, F32Attr>;
def SyncConstantF64Op : SyncConstantOp<"f64", F64, F64Attr>;

def CaseOp : TFRT_Op<"case"> {
  let summary = "An n-way switch statement which calls a single branch function.";
  let description = [{
    An n-way switch statement, implementing the following:
        ```
        switch (branch_index) {
          case 0:
            output = branches[0](input);
            break;
          case 1:
            output = branches[1](input);
            break;
          ...
          case [[nbranches-1]]:
          default:
            output = branches[nbranches-1](input);
            break;
        }
        ```
    Example: %ch1, %res = tfrt.case %branch_idx [@branch0, @branch1] (%ch0, %arg0, %arg1) : (i32, i32) -> i32
  }];

  let arguments = (ins I32:$branch_index,
                       ArrayAttr:$branches,
                       TFRT_ChainType:$in_op_chain,
                       Variadic<AnyType>:$branch_operands);

  let results = (outs TFRT_ChainType:$out_op_chain,
                      Variadic<AnyType>:$branch_outputs);
  let assemblyFormat = [{
    $branch_index $branches `(` $in_op_chain (`,` $branch_operands^)? `)` attr-dict `:` `(` type($branch_operands) `)` `->` `(` type($branch_outputs) `)`
  }];
  let hasVerifier = 0;
}

def IfOp : TFRT_Op<"if"> {
  let summary = "if operation";
  let description = [{
    The "tfrt.if" operation takes an i1 condition, a 'then' body and an 'else'
    body.  If the condition is true, the 'then' body is run and the "tfrt.if"
    operation returns the results of that region, otherwise the "else" body is
    run and its values are returned.

    The body regions must have the same signature, but they may take an
    arbitrary number of arguments and return values with the same types as the
    op return types, using a 'tfrt.return' operation at the end of the
    region.  The initial values are specified as operands of "tfrt.if" (along
    with the condition), and the types of the values are specified with a
    function type.

    This operation is safe to use as a 'nonstrict' op, which dispatches its body
    whenever the condition and any dependent arguments are resolved.

    Example:

      %res = tfrt.if %cond, %x, %y : (i32, f32) -> i32 {
        %v1 = "use"(%x, %y) : (i32, f32) -> i32
        tfrt.return %v1 : i32
      } else {
        %v2 = "abuse"(%x, %y) : (i32, f32) -> i32
        tfrt.return %v2 : i32
      }
  }];
  let arguments = (ins I1:$cond, Variadic<AnyType>);
  let results = (outs Variadic<AnyType>);
  let regions = (region SizedRegion<1>:$then_region,
                        SizedRegion<1>:$else_region);
}

def CondOp : TFRT_Op<"cond"> {
  let summary = "conditional operation";
  let description = [{
    The "tfrt.cond" operation takes an i1 condition, a 'true' function and a
    'false' function.  If the condition is true, the 'true' function is called
    and the "tfrt.cond" operation returns the results of that function, otherwise
    the "false" function is called and its values are returned.

    The 'true' and 'false' functions must have the same signature, but they may
    take an arbitrary number of arguments and return values with the same types
    as the op return types. Both functions are encoded as function attribute,
    named true_fn and false_fn. The a_ and b_ prefix prevent reordering by BEF
    since BEF orders attribute by name alphabetically.

    This operation is safe to use as a 'nonstrict' op, which dispatches its body
    whenever the condition and any dependent arguments are resolved.

    Example:

      %res = tfrt.cond %cond @true_fn @false_fn (%x, %y) : (i32, f32) -> (i32)
  }];
  let arguments = (ins I1:$cond,
                       FlatSymbolRefAttr:$a_true_fn,
                       FlatSymbolRefAttr:$b_false_fn,
                       Variadic<AnyType>:$fn_operands);
  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    $cond $a_true_fn $b_false_fn `(` $fn_operands `)` attr-dict `:` `(` type($fn_operands) `)` `->` `(` type($outputs) `)`
  }];
}

def WhileOp : TFRT_Op<"while"> {
  let summary = "while operation";
  let description = [{
    cond: The boolean to control whether the first iteration should be
      executed.
    operands: The arguments to the first iteration.
    results: The results of the last iteration. The number and types of results
      are the same as the number and types of operands.
    body_fn: The body function that takes the arguments and returns the results
      and an I1 value to indicate whether next iteration should be executed.
    parallel_iterations: The max number of iterations that can be dispatched in parallel.

    The pseudo code:

    while(cond) {
      results, cond = body_fn(operands)
      operands = results
    }
    return results

  }];

  let arguments = (ins I1:$cond,
                       Variadic<AnyType>:$operands,
                       FlatSymbolRefAttr:$body_fn,
                       I64Attr:$parallel_iterations);

  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    $cond $body_fn `(` $operands `)` attr-dict `parallel_iterations` `(` $parallel_iterations `)` `:` `(` type($operands) `)` `->` `(` type($results) `)`
  }];

  let builders = [
    OpBuilder<(ins "::mlir::TypeRange":$results, "::mlir::Value":$cond, "::mlir::ValueRange":$operands, "::llvm::StringRef":$body_fn, CArg<"int64_t", "1">:$parallel_iterations), [{
      build($_builder, $_state, results, cond, operands, ::mlir::FlatSymbolRefAttr::get($_builder.getStringAttr(body_fn)), $_builder.getI64IntegerAttr(parallel_iterations));
    }]>
  ];

  let hasVerifier = 0;
}

def OnceOp : TFRT_Op<"once"> {
  let summary = "once operation";
  let description = [{
    Calls 'function' exactly once and stores the result in the resource context.
    Subsequent calls return the same result without calling the function again.

    Example:

    func @once(%arg0 : int32) -> i32 {
      tfrt.return %arg0 : i32
    }

    // %1 always has the same value as %0 of the first execution.
    %1 = tfrt.once @once(%0)
  }];

  let arguments = (ins Variadic<AnyType>:$operands,
                       FlatSymbolRefAttr:$function);

  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    $function `(` $operands `)` attr-dict
    `:` `(` type($operands) `)` `->` `(` type($results) `)`
  }];

  let hasVerifier = 0;
}

def NewChainOp : TFRT_Op<"new.chain", [NoSideEffect]> {
  let summary = "host executor chain constructor";
  let description = [{
    The "tfrt.new.chain" operation returns a !tfrt.chain value, which can be fed
    into a side-effecting operation that does not depend on anything.
    For example:

      func @foo() {
        ...
        %x = tfrt.new.chain
        "side.effecting.thing"(%x) : (!tfrt.chain) -> ()
      }
  }];

  let arguments = (ins);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 0;
}

def MergeChainsOp : TFRT_Op<"merge.chains", [NoSideEffect]> {
  let summary = "host executor merge chain";
  let description = [{
    The "tfrt.merge.chains" operation merges variadic number of inputs and
    outputs a single !tfrt.chain output.

    Example:

        %c1 = tfrt.new.chain
        %c2 = tfrt.new.chain
        %c3 = tfrt.new.chain
        %merged_c = tfrt.merge.chains %c1, %c2, %c3
  }];

  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
  let hasVerifier = 0;
}

def RepeatI32Op : TFRT_Op<"repeat.i32"> {
  let summary = "repeat.i32 operation";
  let description = [{
    The "tfrt.repeat.i32" operation is a repeat loop that executes its body N
    times, where N is a 32-bit value specified by its first operand.  It has one
    region which represents the loop body.

    The body region takes an arbitrary number of arguments and must return
    values with the same types, using a 'tfrt.return' operation at the end of the
    region.  The initial values for the loop are specified as operands (along
    with the trip count), and the types of the loop values are specified with a
    type list.  The operation returns the results of the final iteration.

    This operation is safe to use as a 'nonstrict' op, which dispatches its body
    whenever dependent arguments are resolved.

    Example:

      %res1, %res2 = tfrt.repeat.i32 %i, %loopval1, %loopval2 : i32, f32 {
        "use"(%loopval1, %loopval2) : (i32, f32) -> ()
        ...
        tfrt.return %loopval1, %loopval2 : i32, f32
      }
  }];
  let arguments = (ins I32:$trip_count, Variadic<AnyType>);
  let results = (outs Variadic<AnyType>);
  let regions = (region SizedRegion<1>:$region);
}

def ParallelForI32Op : TFRT_Op<"parallel_for.i32"> {
  let summary = "parallel_for.i32 operation";
  let description = [{
    The "tfrt.parallel_for.i32" operation is a parallel for loop that executes
    its body function for non overlapping sub ranges. It has one region which
    represents the loop body function (see host_context/parallel_for.h C++
    equivalent).

    The body region takes an arbitrary number of arguments and must return a
    single chain value using a 'tfrt.return' operation at the end of the region
    if it completes asynchronously. If the body region completes synchronously
    it must have empty returns (`tfrt.return` without operands).

    Inside the loop region `%start` and `%end` values are bound to the parallel
    block start and end offsets (see example below).

    This is a TFRT counterpart of the native C++ ParallelFor operation defined
    in: `host_context/parallel_for.h`.

    Example:

      %from       = tfrt.constant.i32 0
      %to         = tfrt.constant.i32 100
      %block_size = tfrt.constant.i32 10

      %val = ... : !my.type

      %done = tfrt.parallel_for.i32 %from to %to fixed %block_size,
                                   %val : !my.type {
        // Possible values for [%start, %end) are: [0, 10), [10, 20), ...
        "use"(%start, %end, %val) : (i32, i32, !my.type) -> ()

        // Async regions must return a chain to signal completion.
        tfrt.return %chain : !tfrt.chain

        // Sync regions must have an empty return.
        tfrt.return
      }
  }];
  let arguments = (ins I32:$start, I32:$end, I32:$block_size,
                   Variadic<AnyType>);
  let results = (outs Variadic<AnyType>);
  let regions = (region SizedRegion<1>:$region);
}

def ParallelCallI32Op : TFRT_Op<"parallel_call.i32"> {
  let summary = "parallel_call.i32 operation";
  let description = [{
    The "tfrt.parallel_call.i32" is an alternative to "tfrt.parallel_for.i32" with
    a parallel region defined as a standalone function.

    Example:

      func @compute(%start : i32, %end : i32, %val : !my_type) -> !tfrt.chain {
        ...
      }

      %from       = tfrt.constant.i32 0
      %to         = tfrt.constant.i32 100
      %block_size = tfrt.constant.i32 10

      %val = ... : !my.type

      %done = tfrt.parallel_call.i32 %from to %to fixed %block_size
              @compute(%val) : !my.type
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee,
                   I32:$start, I32:$end, I32:$block_size,
                   Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);
}

def ReturnOp : TFRT_Op<"return", [Terminator]> {
  let summary = "host executor return operation";
  let description = [{
    The "tfrt.return" operation represents a return operation within a function.
    The operation takes variable number of operands and produces no results.
    The operand number and types must match the signature of the function
    that contains the operation. For example:

      func @foo() : (i32, f8) {
        ...
        tfrt.return %0, %1 : i32, f8
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins),
    [{ build($_builder, $_state, llvm::None); }]>];
}

class AddOp<string suffix, Type type>
  : TFRT_Op<"add." # suffix, [NoSideEffect]> {
  let summary = "tfrt.add operation";
  let description = [{
     An operation that takes two inputs and returns their sum as the result.

     Example:
       %2 = tfrt.add.i32 %0, %1
  }];
  let arguments = (ins type, type);
  let results = (outs type);
  let assemblyFormat = "operands attr-dict";
  let hasVerifier = 0;
}

def AddI32Op : AddOp<"i32", I32>;
def AddI64Op : AddOp<"i64", I64>;
def AddF32Op : AddOp<"f32", F32>;
def AddF64Op : AddOp<"f64", F64>;

class AndOp<string suffix, Type type>
  : TFRT_Op<"and." # suffix, [NoSideEffect]> {
  let summary = "tfrt.and operation";
  let description = [{
     An operation that takes two inputs and returns their conjunction as the
     result.

     Example:
       %2 = tfrt.and.i1 %0, %1
  }];
  let arguments = (ins type, type);
  let results = (outs type);
  let assemblyFormat = "operands attr-dict";
  let hasVerifier = 0;
}

def AndI1Op : AndOp<"i1", I1>;

class EqualOp<string suffix, Type type>
  : TFRT_Op<"equal." # suffix, [NoSideEffect]> {
  let summary = "tfrt.equal operation";
  let description = [{
     An operation that takes two inputs and returns their equality as the
     result.

     Example:
       %2 = tfrt.equal.i32 %0, %1
  }];
  let arguments = (ins type, type);
  let results = (outs I1);
  let assemblyFormat = "operands attr-dict";
  let hasVerifier = 0;
}

def EqualI32Op : EqualOp<"i32", I32>;
def EqualI64Op : EqualOp<"i64", I64>;

class DivOp<string suffix, Type type>
  : TFRT_Op<"div." # suffix, [NoSideEffect]> {
  let summary = "tfrt.div operation";
  let description = [{
     An operation that takes two inputs and returns their quotient and remainder
     as the result.

     Example:
       %2, %3 = tfrt.div.i32 %0, %1
  }];
  let arguments = (ins type, type);
  let results = (outs type, type);
  let assemblyFormat = "operands attr-dict";
  let hasVerifier = 0;
}

def DivI32Op : DivOp<"i32", I32>;
def DivI64Op : DivOp<"i64", I64>;

class MulOp<string suffix, Type type>
  : TFRT_Op<"mul." # suffix, [NoSideEffect]> {
  let summary = "tfrt.mul operation";
  let description = [{
     An operation that takes two inputs and returns their product.

     Example:
       %2 = tfrt.mul.i32 %0, %1
  }];
  let arguments = (ins type, type);
  let results = (outs type);
  let assemblyFormat = "operands attr-dict";
  let hasVerifier = 0;
}

def MulI32Op : MulOp<"i32", I32>;
def MulI64Op : MulOp<"i64", I64>;

class SyncMulOp<string suffix, Type type>
  : TFRT_Op<"mul_s." # suffix, [NoSideEffect]> {
  let summary = "tfrt.mul operation";
  let description = [{
     An operation that takes two inputs and returns their product.

     Example:
       %2 = tfrt.mul.i32 %0, %1
  }];
  let arguments = (ins type, type);
  let results = (outs type);
  let assemblyFormat = "operands attr-dict";
  let hasVerifier = 0;
}

def SyncMulI32Op : SyncMulOp<"i32", I32>;
def SyncMulI64Op : SyncMulOp<"i64", I64>;

class PrintOp<string suffix, Type type> : TFRT_Op<"print." # suffix> {
  let summary = "tfrt.print operation";
  let description = [{
     An operation takes a number input and a chain input.
     It prints the number to stdout and returns a chain output.
     The chain input must be the second operand.

     Example:
       %2 = tfrt.print.i32 %0, %1
  }];
  let arguments = (ins type, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = "operands attr-dict";
  let hasVerifier = 0;
}

def PrintI1Op : PrintOp<"i1", I1>;
def PrintI32Op : PrintOp<"i32", I32>;
def PrintI64Op : PrintOp<"i64", I64>;
def PrintF32Op : PrintOp<"f32", F32>;
def PrintF64Op : PrintOp<"f64", F64>;

def GetDeviceOp : TFRT_Op<"get_device"> {
  let summary = "Get the device object from the given device name";
  let description = [{
    Get the device object from the given device name.

    Example:
      %result = tfrt.get_device %chain { device_name="GPU:0" }
  }];

  let arguments = (ins
    TFRT_ChainType:$chain_in,
    StrAttr:$device_name
  );
  let results = (outs TFRT_DeviceType:$result);
  let assemblyFormat = "operands attr-dict";
  let hasVerifier = 0;
}

#endif  // BASIC_OPS
