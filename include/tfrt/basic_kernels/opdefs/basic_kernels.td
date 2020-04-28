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

include "tfrt/tfrt_op_base.td"
include "mlir/Interfaces/SideEffects.td"

// "hex" Host Executor dialect.
def Hex_Dialect : Dialect {
  let name = "hex";

  let description = [{
    The Host Executor dialect.

    This dialect contains common host operations.
  }];

  let cppNamespace = "hex";
}

// Base class for Hex dialect ops.
class Hex_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Hex_Dialect, mnemonic, !listconcat(traits, [IsolatedFromAbove])> {

  // Each registered op in the Hex namespace needs to provide all of a printer,
  // parser and verifier.
  let printer = [{ return tfrt::hex::print(p, *this); }];
  let verifier = [{ return tfrt::hex::verify(*this); }];
  let parser = [{ return tfrt::hex::parse$cppClass(parser, result); }];
}

def CallOp : Hex_Op<"call"> {
  let summary = "host executor call operation";
  let description = [{
    The "hex.call" operation represents a direct call to a function.  The
    operands and result types of the call must match the specified function
    type.  The callee is encoded as a function attribute named "callee".

      %2 = hex.call @my_add(%0, %1) : (f32, f32) -> f32
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let extraClassDeclaration = [{
    StringRef getCallee() { return callee(); }
    mlir::FunctionType getCalleeType();
  }];
}

class ConstantOp<string suffix, Type baseType, Attr attr>
  : Hex_Op<"constant." # suffix, [NoSideEffect]> {
  let summary = "host executor constant value constructor";

  let arguments = (ins attr:$value);
  let results = (outs baseType);
}

def ConstantBoolOp  : ConstantOp<"bool", I1, BoolAttr>;
def ConstantI1Op  : ConstantOp<"i1", I1, I1Attr>;
def ConstantI32Op : ConstantOp<"i32", I32, I32Attr>;
def ConstantI64Op : ConstantOp<"i64", I64, I64Attr>;
def ConstantF32Op : ConstantOp<"f32", F32, F32Attr>;
def ConstantF64Op : ConstantOp<"f64", F64, F64Attr>;

def IfOp : Hex_Op<"if"> {
  let summary = "if operation";
  let description = [{
    The "hex.if" operation takes an i1 condition, a 'then' body and an 'else'
    body.  If the condition is true, the 'then' body is run and the "hex.if"
    operation returns the results of that region, otherwise the "else" body is
    run and its values are returned.

    The body regions must have the same signature, but they may take an
    arbitrary number of arguments and return values with the same types as the
    op return types, using a 'hex.return' operation at the end of the
    region.  The initial values are specified as operands of "hex.if" (along
    with the condition), and the types of the values are specified with a
    function type.

    This operation is safe to use as a 'nonstrict' op, which dispatches its body
    whenever the condition and any dependent arguments are resolved.

    Example:

      %res = hex.if %cond, %x, %y : (i32, f32) -> i32 {
        %v1 = "use"(%x, %y) : (i32, f32) -> i32
        hex.return %v1 : i32
      } else {
        %v2 = "abuse"(%x, %y) : (i32, f32) -> i32
        hex.return %v2 : i32
      }
  }];
  let arguments = (ins I1:$cond, Variadic<AnyType>);
  let results = (outs Variadic<AnyType>);
  let regions = (region SizedRegion<1>:$then_region,
                        SizedRegion<1>:$else_region);
}

def NewChainOp : Hex_Op<"new.chain", [NoSideEffect]> {
  let summary = "host executor chain constructor";
  let description = [{
    The "hex.new.chain" operation returns a !hex.chain value, which can be fed
    into a side-effecting operation that does not depend on anything.
    For example:

      func @foo() {
        ...
        %x = hex.new.chain
        "side.effecting.thing"(%x) : (!hex.chain) -> ()
      }
  }];

  let arguments = (ins);
  let results = (outs ChainType);
  let assemblyFormat = "attr-dict";
  let verifier = ?;
}

def MergeChainsOp : Hex_Op<"merge.chains", [NoSideEffect]> {
  let summary = "host executor merge chain";
  let description = [{
    The "hex.merge.chains" operation merges variadic number of !hex.chain inputs
    and outputs a single !hex.chain output.

    Example:

        %c1 = hex.new.chain
        %c2 = hex.new.chain
        %c3 = hex.new.chain
        %merged_c = hex.merge.chains %c1, %c2, %c3
  }];

  let arguments = (ins Variadic<ChainType>:$chains);
  let results = (outs ChainType);
  let assemblyFormat = "$chains attr-dict";
  let verifier = ?;
}

def RepeatI32Op : Hex_Op<"repeat.i32"> {
  let summary = "repeat.i32 operation";
  let description = [{
    The "hex.repeat.i32" operation is a repeat loop that executes its body N
    times, where N is a 32-bit value specified by its first operand.  It has one
    region which represents the loop body.

    The body region takes an arbitrary number of arguments and must return
    values with the same types, using a 'hex.return' operation at the end of the
    region.  The initial values for the loop are specified as operands (along
    with the trip count), and the types of the loop values are specified with a
    type list.  The operation returns the results of the final iteration.

    This operation is safe to use as a 'nonstrict' op, which dispatches its body
    whenever dependent arguments are resolved.

    Example:

      %res1, %res2 = hex.repeat.i32 %i, %loopval1, %loopval2 : i32, f32 {
        "use"(%loopval1, %loopval2) : (i32, f32) -> ()
        ...
        hex.return %loopval1, %loopval2 : i32, f32
      }
  }];
  let arguments = (ins I32:$trip_count, Variadic<AnyType>);
  let results = (outs Variadic<AnyType>);
  let regions = (region SizedRegion<1>:$region);
}

def ReturnOp : Hex_Op<"return", [Terminator]> {
  let summary = "host executor return operation";
  let description = [{
    The "hex.return" operation represents a return operation within a function.
    The operation takes variable number of operands and produces no results.
    The operand number and types must match the signature of the function
    that contains the operation. For example:

      func @foo() : (i32, f8) {
        ...
        hex.return %0, %1 : i32, f8
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<
    "OpBuilder &b, OperationState &result", [{ build(b, result, llvm::None); }]
  >];
}

class AddOp<string suffix, Type type>
  : Hex_Op<"add." # suffix, [NoSideEffect]> {
  let summary = "hex.add operation";
  let description = [{
     An operation that takes two inputs and returns their sum as the result.

     Example:
       %2 = hex.add.i32 %0, %1
  }];
  let arguments = (ins type, type);
  let results = (outs type);
  let assemblyFormat = "operands attr-dict";
  let verifier = ?;
}

def AddI32Op : AddOp<"i32", I32>;
def AddI64Op : AddOp<"i64", I64>;
def AddF32Op : AddOp<"f32", F32>;
def AddF64Op : AddOp<"f64", F64>;

class AndOp<string suffix, Type type>
  : Hex_Op<"and." # suffix, [NoSideEffect]> {
  let summary = "hex.and operation";
  let description = [{
     An operation that takes two inputs and returns their conjunction as the
     result.

     Example:
       %2 = hex.and.i1 %0, %1
  }];
  let arguments = (ins type, type);
  let results = (outs type);
  let assemblyFormat = "operands attr-dict";
  let verifier = ?;
}

def AndI1Op : AndOp<"i1", I1>;

class EqualOp<string suffix, Type type>
  : Hex_Op<"equal." # suffix, [NoSideEffect]> {
  let summary = "hex.equal operation";
  let description = [{
     An operation that takes two inputs and returns their equality as the
     result.

     Example:
       %2 = hex.equal.i32 %0, %1
  }];
  let arguments = (ins type, type);
  let results = (outs I1);
  let assemblyFormat = "operands attr-dict";
  let verifier = ?;
}

def EqualI32Op : EqualOp<"i32", I32>;
def EqualI64Op : EqualOp<"i64", I64>;

class DivOp<string suffix, Type type>
  : Hex_Op<"div." # suffix, [NoSideEffect]> {
  let summary = "hex.div operation";
  let description = [{
     An operation that takes two inputs and returns their quotient and remainder
     as the result.

     Example:
       %2, %3 = hex.div.i32 %0, %1
  }];
  let arguments = (ins type, type);
  let results = (outs type, type);
  let assemblyFormat = "operands attr-dict";
  let verifier = ?;
}

def DivI32Op : DivOp<"i32", I32>;
def DivI64Op : DivOp<"i64", I64>;

class PrintOp<string suffix, Type type> : Hex_Op<"print." # suffix> {
  let summary = "hex.print operation";
  let description = [{
     An operation takes a number input and a chain input.
     It prints the number to stdout and returns a chain output.
     The chain input must be the second operand.

     Example:
       %2 = hex.print.i32 %0, %1
  }];
  let arguments = (ins type, ChainType);
  let results = (outs ChainType);
  let assemblyFormat = "operands attr-dict";
  let verifier = ?;
}

def PrintI1Op : PrintOp<"i1", I1>;
def PrintI32Op : PrintOp<"i32", I32>;
def PrintI64Op : PrintOp<"i64", I64>;
def PrintF32Op : PrintOp<"f32", F32>;
def PrintF64Op : PrintOp<"f64", F64>;

#endif  // BASIC_OPS
