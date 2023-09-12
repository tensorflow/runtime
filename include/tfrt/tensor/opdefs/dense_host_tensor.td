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
include "mlir/Interfaces/SideEffectInterfaces.td"

// Dense host tensor dialect.
def DHT_Dialect : Dialect {
  let name = "tfrt_dht";

  let description = [{
    The DenseHostTensor dialect.

    This dialect contains operations for working with dense host tensors.
  }];

  let cppNamespace = "tfrt::dht";
}

// Base class for the operation in this dialect
class DHT_Op<string mnemonic, list<Trait> traits = []> :
    Op<DHT_Dialect, mnemonic, traits>;

class CreateUninitializedTensorOp<string dtype, int rank>
  : DHT_Op<"create_uninitialized_tensor." # dtype # "." # rank,
          [Pure]> {
  let summary = "tfrt_dht.create_uninitialized_tensor operation";

  let description = [{
    An operation that creates an uninitialized tensor.
    It takes a shape attribute and outputs a tensor as output.

    Example:
      %2 = tfrt_dht.create_uninitialized_tensor.i32.0 []
      %3 = tfrt_dht.create_uninitialized_tensor.i32.2 [3 : i64, 2 : i64]
  }];

  let arguments = (ins I64ArrayAttr:$shape);
  let results = (outs TensorType);
  let assemblyFormat = "$shape attr-dict";
}

class MakeTensorOp<string dtype>
  : DHT_Op<"make_tensor." # dtype> {
  let summary = "tfrt_dht.make_tensor operation";

  let description = [{
    An operation that creates a tensor from the given shape and buffer.
    Created tensor shares the memory with the given buffer.

    The size of the buffer must match the size needed to hold the tensor,
    i.e. the number of elements, of requested dtype, in the given shape.

    Example:
      %ch0 = tfrt.new.chain
      %cuda_host_buf, %ch1 = cuda.mem.allocate_host %size, %alignment, %ch0
      %tensor, %ch2 = tfrt_dht.make_tensor.f32 %cuda_host_buf, %shape, %ch0
  }];

  let arguments = (ins HostBufferType, TS_Shape, TFRT_ChainType);
  let results = (outs TensorType, TFRT_ChainType);
  let assemblyFormat = "operands attr-dict";
}

class FillTensorOp<string dtype>
  : DHT_Op<"fill_tensor_with_constant." # dtype> {
  let summary = "tfrt_dht.fill_tensor_with_constant operation";

  let description = [{
    An operation that fills an input tensor with a value. It takes an input
    tensor and a chain as inputs and a constant value attribute, and outputs a
    chain.

    Example:
      %1 = tfrt.new.chain
      %2 = tfrt_dht.create_uninitialized_tensor.i32.2 [3 : i32, 2 : i32]
      %3 = tfrt_dht.fill_tensor_with_constant.i32 %2, %1 0 : i32
  }];

  let arguments = (ins TensorType, TFRT_ChainType, AnyAttr:$value);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = "operands $value attr-dict";
}

def PrintTensorOp : DHT_Op<"print_tensor"> {
  let summary = "tfrt_dht.print_tensor operation";

  let description = [{
    An operation that prints a tensor. It takes a tensor and chain as input and
    outputs a chain.

    Example:
      %3 = tfrt_dht.print_tensor %1, %2
  }];

  let arguments = (ins TensorType, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = "operands attr-dict";
}

class SetTensorOp<string dtype>
  : DHT_Op<"set_tensor_with_constant_values." # dtype> {
  let summary = "tfrt_dht.set_tensor_with_constant_values operation";

  let description = [{
    An operation that sets an input tensor with given values. It takes an input
    tensor and a chain as inputs and a list of values as attribute (for complex
    element type, real and complex values are interleaved, resulting in twice
    the number of values overall). This operation outputs a chain.

    Example:
      %1 = tfrt.new.chain
      %2 = tfrt_dht.create_uninitialized_tensor.i32.2 [1 : i32, 2 : i32]
      %3 = tfrt_dht.set_tensor_with_constant_values.i32 %2, %1 [8 : i32, 9 : i32]
      %4 = tfrt_dht.create_uninitialized_tensor.complex64.2 [1 : i32, 2 : i32]
      %5 = tfrt_dht.set_tensor_with_constant_values.complex64 %2, %1
          [8.000e+00 : f32, 0.000e+00 : f32, 9.000e+00 : f32, 0.000e+00 : f32]
  }];

  let arguments = (ins TensorType, TFRT_ChainType, ArrayAttr:$values);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = "operands $values attr-dict";
}

class TensorEqualOp<string dtype>
  : DHT_Op<"tensor_equal." # dtype> {
  let summary = "tfrt_dht.tensor_equal operation";

  let description = [{
    An operation that checks if two input tensors are equal. It takes two input
    tensors and a chain as inputs and outputs an i1 result and a chain output.

    The chain output value is needed to prevent write-after-read hazard.

    Example:
      %4, %5 = tfrt_dht.tensor_equal.i32 %1, %2, %3
  }];

  let arguments = (ins TensorType, TensorType, TFRT_ChainType);
  let results = (outs I1, TFRT_ChainType);
  let assemblyFormat = "operands attr-dict";
}

def AllocateBufferOp : DHT_Op<"allocate_buffer"> {
  let summary = "tfrt_dht.allocate_buffer operation";

  let description = [{
    An operation that creates an buffer.
    It takes in the size of the buffer in bytes and the alignment in bytes and
    outputs a buffer of that size.

    Example:
      %size = tfrt.constant.i64 164
      %alignment = tfrt.constant.i64 8
      %buf = tfrt_dht.allocate %size, %alignment
  }];

  let arguments = (ins I64, I64);
  let results = (outs HostBufferType);
  let assemblyFormat = "operands attr-dict";
}

def GetBufferOp : DHT_Op<"get_buffer"> {
  let summary = "tfrt_dht.get_buffer operation";

  let description = [{
    tfrt_dht.get_buffer returns the HostBuffer that backs the given dense host
    tensor.

    Example:
      %t = tfrt_dht.create_uninitialized_tensor.i32.2 [3 : i32, 2 : i32]
      %ch1 = tfrt.new.chain
      %buf, %ch2 = tfrt_dht.get_buffer %t, %ch1
  }];

  let arguments = (ins TensorType, TFRT_ChainType);
  let results = (outs HostBufferType, TFRT_ChainType);
  let assemblyFormat = "operands attr-dict";
}

def GetBufferSliceOp : DHT_Op<"get_buffer_slice"> {
  let summary = "tfrt_dht.get_buffer_slice operation";

  let description = [{
    An operation that gets a slice of a buffer.
    It takes in the parent buffer and the size and offset of the new buffer. It
    returns a buffer that points to the data allocated by the parent buffer.

    Example:
      %parent_buf = tfrt_dht.allocate %parent_size, %alignment
      %offset = tfrt.constant.i64 0
      %size = tfrt.constant.i64 64
      %buf_a = tfrt_dht.get_buffer_slice %parent_buf, %offset, %size
  }];

  let arguments = (ins HostBufferType, I64, I64);
  let results = (outs HostBufferType);
  let assemblyFormat = "operands attr-dict";
}

def PrintBufferOp : DHT_Op<"print_buffer"> {
  let summary = "tfrt_dht.print_buffer operation";

  let description = [{
    An operation that prints a buffer.

    Example:
      %chain1 = tfrt_dht.print_buffer %buffer, %chain0
  }];

  let arguments = (ins HostBufferType, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = "operands attr-dict";
}

def GetTensorShapeOp : DHT_Op<"get_tensor_shape", [Pure]> {
  let summary = "tfrt_dht.get_tensor_shape operation";

  let description = [{
    An operation that returns the shape of the dense host tensor.

    Example:
      %0 = tfrt_dht.get_tensor_shape %t
  }];

  let arguments = (ins TensorType);
  let results = (outs TS_Shape);
  let assemblyFormat = "operands attr-dict";
}

// Instantiate for each dtype and rank pair.
foreach dtype = ["ui8", "ui16", "ui32", "ui64", "i32", "f32", "i64", "bool", "complex64", "complex128"] in {
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
