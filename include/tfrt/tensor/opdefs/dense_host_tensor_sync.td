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

//===- dense_host_tensor.td -----------------------------------------------===//
//
// Operation definitions for dense host tensor ops.
//
//===----------------------------------------------------------------------===//

#ifdef DHT_SYNC_OPS
#else
#define DHT_SYNC_OPS

include "tfrt/tfrt_op_base.td"
include "tfrt/tensor/opdefs/tensor_shape_base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Dense host tensor dialect.
def DHT_Sync_Dialect : Dialect {
  let name = "tfrt_dht_sync";

  let description = [{
    The DenseHostTensor dialect sync.

    This dialect contains synchronous operations for working with dense host tensors.
  }];

  let cppNamespace = "dht";
}

// Base class for the operation in this dialect
class DHT_Sync_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<DHT_Sync_Dialect, mnemonic, traits>;

class SyncCreateUninitializedTensorOp<string dtype, int rank>
  : DHT_Sync_Op<"create_uninitialized_tensor." # dtype # "." # rank,
          [NoSideEffect]> {
  let summary = "tfrt_dht_sync.create_uninitialized_tensor operation";

  let description = [{
    An operation that creates an uninitialized tensor.
    It takes a shape attribute and outputs a tensor as output.

    Example:
      %2 = tfrt_dht_sync.create_uninitialized_tensor.i32.0 []
      %3 = tfrt_dht_sync.create_uninitialized_tensor.i32.2 [3 : i64, 2 : i64]
  }];

  let arguments = (ins I64ArrayAttr:$shape);
  let results = (outs TensorType);
  let assemblyFormat = "$shape attr-dict";
}

def SyncPrintTensorOp : DHT_Sync_Op<"print_tensor"> {
  let summary = "tfrt_dht_sync.print_tensor operation";

  let description = [{
    An operation that prints a tensor. It takes a tensor as input.

    Example:
      tfrt_dht_sync.print_tensor %1
  }];

  let arguments = (ins TensorType);
  let results = (outs);
  let assemblyFormat = "operands attr-dict";
}

class SyncSetTensorOp<string dtype>
  : DHT_Sync_Op<"set_tensor_with_constant_values." # dtype> {
  let summary = "tfrt_dht_sync.set_tensor_with_constant_values operation";

  let description = [{
    An operation that sets an input tensor with given values. It takes an input
    tensor and a list of values as attribute, and outputs nothing.

    Example:
      %2 = tfrt_dht.create_uninitialized_tensor.i32.2 [1 : i32, 2 : i32]
      tfrt_dht.set_tensor_with_constant_values.i32 %2 [0 : i32, 1 : i32]
  }];

  let arguments = (ins TensorType:$tensor, ArrayAttr:$values);
  let results = (outs);
  let assemblyFormat = "$tensor $values attr-dict";
}

// Instantiate for each dtype and rank pair.
foreach dtype = ["ui8", "ui16", "ui32", "ui64", "i32", "f32", "i64", "bool", "complex64", "complex128"] in {
  foreach rank = [0, 1, 2, 3, 4] in {
    def DHT_SyncCreateUninitializedTensorOp_#dtype#_#rank
      : SyncCreateUninitializedTensorOp<dtype, rank>;
  }
  def DHT_SyncSetTensorOp_#dtype : SyncSetTensorOp<dtype>;
}

#endif  // DHT_SYNC_OPS
