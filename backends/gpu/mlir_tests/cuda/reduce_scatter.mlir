// Copyright 2021 The TensorFlow Runtime Authors
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

// RUN: bef_executor_lite %s.bef -work_queue_type=mstd | FileCheck %s

func.func @reduce_scatter(
  %rank : i32, %count : i32, %id : !tfrt_gpu.ccl.id
) -> (!tfrt.chain, !t.tensor) {
  %ch0 = tfrt.new.chain
  %device = tfrt_gpu.device.get CUDA, %rank
  %context = tfrt_gpu.context.primary %device

  %tensor_in = tfrt_dht.create_uninitialized_tensor.i32.1 [2: i64]
  %ch1 = tfrt_dht.set_tensor_with_constant_values.i32 %tensor_in, %ch0 [1: i32, 2: i32]
  %buffer_in:2 = tfrt_dht.get_buffer %tensor_in, %ch0
  %pinned_in = tfrt_gpu.mem.register %context, %buffer_in#0

  %tensor_out = tfrt_dht.create_uninitialized_tensor.i32.1 [1: i64]
  %ch2 = tfrt_dht.set_tensor_with_constant_values.i32 %tensor_out, %ch1 [0: i32]
  %buffer_out:2 = tfrt_dht.get_buffer %tensor_out, %ch0
  %pinned_out = tfrt_gpu.mem.register %context, %buffer_out#0

  %ccl = tfrt_gpu.ccl.create %context, %rank, %count, %id
  %ch3 = tfrt_gpu.ccl.reduce_scatter %ccl, %pinned_in, %pinned_out, ncclInt32, ncclSum, %ch2

  %stream = tfrt_gpu.stream.create %context
  %ch4 = tfrt_gpu.ccl.execute %stream, %ccl, %ch3
  %ch5 = tfrt_gpu.stream.synchronize %stream, %ch4

  tfrt.return %ch5, %tensor_out : !tfrt.chain, !t.tensor
}

// CHECK-LABEL: --- Running 'reduce_scatter_test'
func.func @reduce_scatter_test() {
  %count = tfrt.constant.i32 2
  %id = tfrt_gpu.ccl.unique_id CUDA

  %rank0 = tfrt.constant.i32 0
  %rank1 = tfrt.constant.i32 1

  %ch0, %t0 = tfrt.call @reduce_scatter(%rank0, %count, %id)
              : (i32, i32, !tfrt_gpu.ccl.id) -> (!tfrt.chain, !t.tensor)
  %ch1, %t1 = tfrt.call @reduce_scatter(%rank1, %count, %id)
              : (i32, i32, !tfrt_gpu.ccl.id) -> (!tfrt.chain, !t.tensor)

  // CHECK: DenseHostTensor dtype = i32, shape = [1], values = [2]
  %ch2 = tfrt_dht.print_tensor %t0, %ch0
  %ch3 = tfrt.merge.chains %ch1, %ch2 : !tfrt.chain, !tfrt.chain
  // CHECK: DenseHostTensor dtype = i32, shape = [1], values = [4]
  %ch4 = tfrt_dht.print_tensor %t1, %ch3

  tfrt.return
}
