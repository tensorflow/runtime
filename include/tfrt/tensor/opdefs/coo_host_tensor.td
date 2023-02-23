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

//===- coo_host_tensor.td -------------------------------------------------===//
//
// Operation definitions for COO host tensor ops.
//
//===----------------------------------------------------------------------===//

#ifdef COO_OPS
#else
#define COO_OPS

include "tfrt/tfrt_op_base.td"

// Dense host tensor dialect.
def COO_Dialect : Dialect {
  let name = "coo";

  let description = [{
    The CooHostTensor dialect.

    This dialect contains operations for working with sparse tensors.
  }];

  let cppNamespace = "tfrt::coo";
}

// Base class for the operation in this dialect.
class COO_Op<string mnemonic, list<Trait> traits = []> :
    Op<COO_Dialect, mnemonic, traits>;

class ConvertToDHTOp<string dtype, int rank>
  : COO_Op<"convert_coo_to_dht." # dtype # "." # rank> {
  let summary = "coo.convert_coo_to_dht operation";

  let description = [{
    Convert a sparse tensor to a dense host tensor.
    It takes a sparse tensor and chain as input and outputs dht tensor and
    chain.

    Example:
      %4, %3 = coo.convert_coo_to_dht.i32.0 %1, %0
      %5, %6 = coo.convert_coo_to_dht.i32.2 %2, %0
  }];

  let arguments = (ins TensorType, TFRT_ChainType);
  let results = (outs TensorType, TFRT_ChainType);
  let assemblyFormat = "operands attr-dict";
}

class ConvertFromDHTOp<string dtype, int rank>
  : COO_Op<"convert_dht_to_coo." # dtype # "." # rank> {
  let summary = "coo.convert_dht_to_coo operation";

  let description = [{
    Convert a dense host tensor into a coo sparse host tensor.
    It takes a dht tensor and chain as input and outputs a coo tensor and chain.

    Example:
      %4, %3 = coo.convert_dht_to_coo.i32.0 %1, %0
      %5, %6 = coo.convert_dht_to_coo.i32.2 %2, %0
  }];

  let arguments = (ins TensorType, TFRT_ChainType);
  let results = (outs TensorType, TFRT_ChainType);
  let assemblyFormat = "operands attr-dict";
}

class PrintSparseTensorOp : COO_Op<"print_tensor"> {
  let summary = "coo.print_tensor operation";

  let description = [{
    An operation that prints a tensor.
    It takes a tensor and chain as input and outputs a chain.

    Example:
      %3 = coo.print_tensor %1, %2
  }];

  let arguments = (ins TensorType, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = "operands attr-dict";
}

class SparseTensorEqualOp<string dtype, int rank>
  : COO_Op<"tensor_equal." # dtype # "." # rank> {
  let summary = "coo.tensor_equal operation";

  let description = [{
    An operation that checks if two input tensors are equal.
    It takes two input tensors and a chain as inputs and outputs an i1 result
    and a chain output.

    The chain output value is needed to prevent write-after-read hazard.

    Example:
      %4, %5 = "coo.tensor_equal.i32.2 %1, %2, %3
  }];

  let arguments = (ins TensorType, TensorType, TFRT_ChainType);
  let results = (outs I1, TFRT_ChainType);
  let assemblyFormat = "operands attr-dict";
}

// Instantiate for each dtype and rank pair.
foreach dtype = ["i32", "f32"] in {
  foreach rank = [0, 1, 2, 3] in {
    def COO_ConvertToDHTOp_#dtype#_#rank : ConvertToDHTOp<dtype, rank>;
    def COO_ConvertFromDHTOp_#dtype#_#rank : ConvertFromDHTOp<dtype, rank>;
    def COO_TensorEqualOp_#dtype#_#rank : SparseTensorEqualOp<dtype, rank>;
  }
}

def COO_PrintTensorOp : PrintSparseTensorOp;

#endif  // COO_OPS
