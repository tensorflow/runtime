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

//===- tensor_shape.td ----------------------------------------------------===//
//
// Operation definitions for tensor shape ops.
//
//===----------------------------------------------------------------------===//

#ifdef TS_OPS
#else
#define TS_OPS

include "tfrt/tfrt_op_base.td"
include "tfrt/tensor/opdefs/tensor_shape_base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Tensor shape dialect.
def TS_Dialect : Dialect {
  let name = "ts";

  let description = [{
    The Tensor Shape dialect.

    This dialect contains operations for working with tensor shapes.
  }];

  let cppNamespace = "ts";
}

// Base class for the operation in this dialect
class TS_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<TS_Dialect, mnemonic, traits> {
  let parser = [{ return tfrt::ts::parse$cppClass(parser, result); }];
  let printer = " return tfrt::ts::printOpWithOperands(p, *this)" "; ";
}

def TS_BuildShapeOp : TS_Op<"build_shape", [NoSideEffect]> {
  let summary = "Build tensor shape operation";

  let description = [{
    An operation that builds a tensor shape of a given rank and extents.
  }];

  let arguments = (ins I64ArrayAttr:$value);
  let results = (outs TS_Shape:$output);
  let assemblyFormat = "$value attr-dict";
}

def TS_EqualShapeOp : TS_Op<"equal_shape"> {
  let summary = "Checks if two tensor shapes are equal";

  let description = [{
    An operation that checks if two tensor shapes are equal.
  }];

  let arguments = (ins TS_Shape,TS_Shape);
  let results = (outs I1);
  let assemblyFormat = "operands attr-dict";
}

def TS_GetDimensionOp : TS_Op<"get_dimension"> {
  let summary = "Returns the dimension size at the given shape index";

  let description = [{
    An operation that returns the dimension size at the given shape index.
  }];

  let arguments = (ins TS_Shape,I32);
  let results = (outs I64);
  let assemblyFormat = "operands attr-dict";
}

def TS_PrintShapeOp : TS_Op<"print_shape"> {
  let summary = "Print tensor shape operation";

  let description = [{
    An operation that prints a tensor shape.
  }];

  let arguments = (ins TS_Shape:$shape);
  let assemblyFormat = "operands attr-dict";
}

class TS_FixedRankShape<int rank>
  : OpaqueType<"ts", "fixed_rank_shape." # rank,
               "fixed_rank_shape">;

class AsFixedRankShapeOp<int rank> : TS_Op<"as_fixed_rank_shape." # rank, [NoSideEffect]> {
  let summary = "Convert TensorShape to FixedRankShape operation";

  let description = [{
    An operation that converts a TensorShape to a FixedRankShape
  }];

  let arguments = (ins TS_Shape);
  let results = (outs TS_FixedRankShape<rank>:$output);
  let assemblyFormat = "operands attr-dict";
}

class PrintFixedRankShapeOp<int rank> : TS_Op<"print_fixed_rank_shape." # rank> {
  let summary = "Print FixedRankShape operation";

  let description = [{
    An operation that prints a FixedRankShape.
  }];

  let arguments = (ins TS_FixedRankShape<rank>:$shape);
  let assemblyFormat = "operands attr-dict";
}

// Instantiate for each rank.
foreach rank = [0, 1, 2, 3] in {
  def TS_AsFixedRankShapeOp_#rank : AsFixedRankShapeOp<rank>;
  def TS_PrintFixedRankShapeOp_#rank : PrintFixedRankShapeOp<rank>;
}

def TS_BuildPartialShapeOp : TS_Op<"build_partial_shape", [NoSideEffect]> {
  let summary = "Build partial tensor shape operation";

  let description = [{
    An operation that builds a partial tensor shape.
  }];

  let arguments = (ins I64ArrayAttr:$value);
  let results = (outs TS_PartialShape:$output);
  let assemblyFormat = "$value attr-dict";
}

def TS_BuildUnrankedPartialShapeOp : TS_Op<"build_unranked_partial_shape", [NoSideEffect]> {
  let summary = "Build unranked partial tensor shape operation";

  let description = [{
    An operation that builds an unranked partial tensor shape.
  }];

  let results = (outs TS_PartialShape:$output);
  let assemblyFormat = "operands attr-dict";
}

def TS_PrintPartialShapeOp : TS_Op<"print_partial_shape"> {
  let summary = "Prints the shape of the given partial tensor shape";

  let description = [{
    An operation that prints the shape of the given partial tensor shape.
  }];

  let arguments = (ins TS_PartialShape:$shape);
  let assemblyFormat = "operands attr-dict";
}

def TS_ToTensorShapeOp : TS_Op<"to_shape"> {
  let summary = "Converts partial tensor shape to tensor shape operation";

  let description = [{
    An operation that converts a partial tensor shape to tensor shape.
  }];

  let arguments = (ins TS_PartialShape:$shape);
  let results = (outs TS_Shape:$output);
  let assemblyFormat = "operands attr-dict";
}

#endif  // TS_OPS
