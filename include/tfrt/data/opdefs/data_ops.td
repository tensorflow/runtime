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

//===- data_ops.td --------------------------------------------------------===//
//
// Operation definitions for data_ops.
//
//===----------------------------------------------------------------------===//

#ifdef DATA_OPS
#else
#define DATA_OPS

include "tfrt/tfrt_op_base.td"

// "data" dialect
def Data_Dialect : Dialect {
  let name = "data";

  let description = [{
    The Data dialect.

    This dialect contains common data operations.
  }];

  let cppNamespace = "data";
}

// Type definition.
def DatasetType: OpaqueType<"hex", "dataset", "!hex.dataset type">;

// Base class for Data dialect ops.
class Data_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Data_Dialect, mnemonic, !listconcat(traits, [IsolatedFromAbove])> {

  // Each registered op in the Data namespace needs to provide all of a printer,
  // parser and verifier.
  let printer = [{ return tfrt::data::print(p, *this); }];
  let parser = [{ return tfrt::data::parse$cppClass(parser, result); }];
}

class RangeDatasetOp<string suffix>
  : Data_Op<"range_dataset." # suffix> {
  let summary = "data range_dataset operation";
  let description = [{
    data.range_dataset yields a step-separated range of values from
    start (inclusive) to stop (exclusive).

    Example:
      %dataset = data.range_dataset.i32 %start, %stop, %step
  }];

  let arguments = (ins
     I64:$start,
     I64:$stop,
     I64:$step
  );
  let results = (outs
     DatasetType:$handle
  );

  let assemblyFormat = "operands attr-dict";
}

def RangeDatasetI32Op : RangeDatasetOp<"i32">;
def RangeDatasetI64Op : RangeDatasetOp<"i64">;

#endif  // DATA_OPS
