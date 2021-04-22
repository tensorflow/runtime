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

//===- gpu_test_ops.td ---------------------------------------------------===//
//
// Operation and dialect definitions for GPU tests.
//
//===----------------------------------------------------------------------===//

#ifdef GPU_TEST_OPS
#else
#define GPU_TEST_OPS

include "tfrt/gpu/kernels/gpu_ops_base.td"

// "tfrt_gpu_test" dialect
def GPU_TestDialect : Dialect {
  let name = "tfrt_gpu_test";

  let description = [{
    The CUDA dialect.

    This dialect contains common CUDA operations.
  }];

  let cppNamespace = "tfrt::gpu";
}

def GPU_TestCpyTensorHtoDOp : Op<GPU_TestDialect, "copy_tensor_host_to_device"> {
  let summary = "tfrt_gpu_test.copy_tensor_host_to_device operation";
  let description = [{
    tfrt_gpu_test.copy_tensor_host_to_device copies a host tensor to the device.

    Returns the newly allocated buffer.

    Example:
      %tensor = tfrt_dht.create_uninitialized_tensor.f32.1 [1 : i64]
      %ch1 = tfrt_dht.set_tensor_with_constant_values.f32 %tensor, %ch0 [1.0 : f32]
      %gpu_buffer = tfrt_gpu_test.copy_tensor_host_to_device %allocator, %stream, %tensor, %ch1
  }];
  let arguments = (ins GPU_AllocatorType, GPU_StreamType, TensorType, TFRT_ChainType);
  let results = (outs GPU_BufferType);
  let assemblyFormat = "operands attr-dict";
}

#endif  // GPU_TEST_OPS
