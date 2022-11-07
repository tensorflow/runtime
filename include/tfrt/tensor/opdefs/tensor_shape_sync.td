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

#ifdef TS_SYNC_OPS
#else
#define TS_SYNC_OPS

include "tfrt/tfrt_op_base.td"
include "tfrt/tensor/opdefs/tensor_shape_base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Tensor shape sync dialect.
def TensorShapeSyncDialect : Dialect {
  let name = "ts_sync";

  let description = [{
    The Tensor Shape sync dialect.

    This dialect contains operations for working with tensor shapes.
  }];

  let cppNamespace = "::tfrt::ts_sync";
}

// Base class for the operation in this dialect
class TS_SYNC_Op<string mnemonic, list<Trait> traits = []> :
    Op<TensorShapeSyncDialect, mnemonic, traits> {}

def TS_SyncBuildShapeOp : TS_SYNC_Op<"build_shape", [Pure]> {
  let summary = "Build tensor shape operation";

  let description = [{
    An operation that builds a tensor shape of a given rank and extents.
  }];

  let arguments = (ins I64ArrayAttr:$value);
  let results = (outs TS_Shape:$output);
  let assemblyFormat = "$value attr-dict";
}

def TS_SyncGetNumElementsOp : TS_SYNC_Op<"get_num_elements"> {
  let summary = "Returns the number of elements in the shape";

  let description = [{
    An operation that returns the number of elements in the shape.
  }];

  let arguments = (ins TS_Shape);
  let results = (outs I64);
  let assemblyFormat = "operands attr-dict";
}

def TS_SyncBuildPartialShapeOp : TS_SYNC_Op<"build_partial_shape", [Pure]> {
  let summary = "Build partial tensor shape operation";

  let description = [{
    An operation that builds a partial tensor shape.
  }];

  let arguments = (ins I64ArrayAttr:$value);
  let results = (outs TS_PartialShape:$output);
  let assemblyFormat = "$value attr-dict";
}

def TS_SyncToTensorShapeOp : TS_SYNC_Op<"to_shape"> {
  let summary = "Converts partial tensor shape to tensor shape operation";

  let description = [{
    An operation that converts a partial tensor shape to tensor shape.
  }];

  let arguments = (ins TS_PartialShape:$shape);
  let results = (outs TS_Shape:$output);
  let assemblyFormat = "operands attr-dict";
}

def TS_SyncToPartialTensorShapeOp : TS_SYNC_Op<"to_partial_shape"> {
  let summary = "Converts tensor shape to partial tensor shape operation";

  let description = [{
    An operation that converts a tensor shape to partial tensor shape.
  }];

  let arguments = (ins TS_Shape:$shape);
  let results = (outs TS_PartialShape:$output);
  let assemblyFormat = "operands attr-dict";
}

#endif  // TS_SYNC_OPS
