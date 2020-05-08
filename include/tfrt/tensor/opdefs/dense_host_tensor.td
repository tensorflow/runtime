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

#ifdef DHT_OPS
#else
#define DHT_OPS

include "tfrt/tfrt_op_base.td"
include "tfrt/tensor/opdefs/tensor_shape_base.td"
include "mlir/Interfaces/SideEffects.td"

// Dense host tensor dialect.
def DHT_Dialect : Dialect {
  let name = "dht";

  let description = [{
    The DenseHostTensor dialect.

    This dialect contains operations for working with dense host tensors.
  }];

  let cppNamespace = "dht";
}

// Base class for the operation in this dialect
class DHT_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<DHT_Dialect, mnemonic, traits>;

class CreateUninitializedTensorOp<string dtype, int rank>
  : DHT_Op<"create_uninitialized_tensor." # dtype # "." # rank,
          [NoSideEffect]> {
  let summary = "dht.create_uninitialized_tensor operation";

  let description = [{
    An operation that creates an uninitialized tensor.
    It takes a shape attribute and outputs a tensor as output.

    Example:
      %2 = dht.create_uninitialized_tensor.i32.0 []
      %3 = dht.create_uninitialized_tensor.i32.2 [3 : i64, 2 : i64]
  }];

  let arguments = (ins);
  let results = (outs TensorType);

  let printer = [{ return tfrt::dht::printCreateUninitTensorOp(p, *this); }];
  let parser = " return tfrt::dht::parseCreateUninitTensorOp(parser, result);";
}

class MakeTensorOp<string dtype>
  : DHT_Op<"make_tensor." # dtype> {
  let summary = "dht.make_tensor operation";

  let description = [{
    An operation that creates a tensor from the given shape and buffer.
    Created tensor shares the memory with the given buffer.

    The size of the buffer must match the size needed to hold the tensor,
    i.e. the number of elements, of requested dtype, in the given shape.

    Example:
      %ch0 = hex.new.chain
      %cuda_host_buf, %ch1 = cuda.mem.allocate_host %size, %alignment, %ch0
      %tensor, %ch2 = dht.make_tensor.f32 %cuda_host_buf, %shape, %ch0
  }];

  let arguments = (ins HostBufferType, TS_Shape, ChainType);
  let results = (outs TensorType, ChainType);
  let assemblyFormat = "operands attr-dict";
}

class FillTensorOp<string dtype>
  : DHT_Op<"fill_tensor_with_constant." # dtype> {
  let summary = "dht.fill_tensor_with_constant operation";

  let description = [{
    An operation that fills an input tensor with a value. It takes an input
    tensor and a chain as inputs and a constant value attribute, and outputs a
    chain.

    Example:
      %1 = hex.new.chain
      %2 = dht.create_uninitialized_tensor.i32.2 [3 : i32, 2 : i32]
      %3 = dht.fill_tensor_with_constant.i32 %2, %1 0 : i32
  }];

  let arguments = (ins TensorType, ChainType);
  let results = (outs ChainType);

  let printer = [{ return tfrt::dht::printFillTensorOp(p, *this); }];
  let parser = " return tfrt::dht::parseFillTensorOp(parser, result);";
}

def PrintTensorOp : DHT_Op<"print_tensor"> {
  let summary = "dht.print_tensor operation";

  let description = [{
    An operation that prints a tensor. It takes a tensor and chain as input and
    outputs a chain.

    Example:
      %3 = dht.print_tensor %1, %2
  }];

  let arguments = (ins TensorType, ChainType);
  let results = (outs ChainType);
  let assemblyFormat = "operands attr-dict";
}

class SetTensorOp<string dtype>
  : DHT_Op<"set_tensor_with_constant_values." # dtype> {
  let summary = "dht.set_tensor_with_constant_values operation";

  let description = [{
    An operation that sets an input tensor with given values. It takes an input
    tensor and a chain as inputs and a list of values as attribute, and outputs
    a chain.

    Example:
      %1 = hex.new.chain
      %2 = dht.create_uninitialized_tensor.i32.2 [1 : i32, 2 : i32]
      %3 = dht.set_tensor_with_constant_values.i32 %2, %1 [0 : i32, 1 : i32]
  }];

  let arguments = (ins TensorType, ChainType);
  let results = (outs ChainType);

  let printer = [{ return tfrt::dht::printSetTensorOp(p, *this); }];
  let parser = " return tfrt::dht::parseSetTensorOp(parser, result);";
}

class TensorEqualOp<string dtype>
  : DHT_Op<"tensor_equal." # dtype> {
  let summary = "dht.tensor_equal operation";

  let description = [{
    An operation that checks if two input tensors are equal. It takes two input
    tensors and a chain as inputs and outputs an i1 result and a chain output.

    The chain output value is needed to prevent write-after-read hazard.

    Example:
      %4, %5 = dht.tensor_equal.i32 %1, %2, %3
  }];

  let arguments = (ins TensorType, TensorType, ChainType);
  let results = (outs I1, ChainType);
  let assemblyFormat = "operands attr-dict";
}

def GetBufferOp : DHT_Op<"get_buffer"> {
  let summary = "dht.get_buffer operation";

  let description = [{
    dht.get_buffer returns the HostBuffer that backs the given dense host
    tensor.

    Example:
      %t = dht.create_uninitialized_tensor.i32.2 [3 : i32, 2 : i32]
      %ch1 = hex.new.chain
      %buf, %ch2 = dht.get_buffer %t, %ch1
  }];

  let arguments = (ins TensorType, ChainType);
  let results = (outs HostBufferType, ChainType);
  let assemblyFormat = "operands attr-dict";
}

def PrintBufferOp : DHT_Op<"print_buffer"> {
  let summary = "dht.print_buffer operation";

  let description = [{
    An operation that prints a buffer.

    Example:
      %chain1 = dht.print_buffer %buffer, %chain0
  }];

  let arguments = (ins HostBufferType, ChainType);
  let results = (outs ChainType);
  let assemblyFormat = "operands attr-dict";
}

// Instantiate for each dtype and rank pair.
foreach dtype = ["ui8", "i32", "f32", "i64", "bool", "complex64"] in {
  foreach rank = [0, 1, 2, 3, 4] in {
    def DHT_CreateUninitializedTensorOp_#dtype#_#rank
      : CreateUninitializedTensorOp<dtype, rank>;
  }
  def DHT_FillTensorOp_#dtype : FillTensorOp<dtype>;
  def DHT_MakeTensorOp_#dtype : MakeTensorOp<dtype>;
  def DHT_SetTensorOp_#dtype : SetTensorOp<dtype>;
  def DHT_TensorEqualOp_#dtype : TensorEqualOp<dtype>;
}

#endif  // DHT_OPS
