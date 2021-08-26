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

func @all_reduce(
  %rank : i32, %count : i32, %id : !tfrt_gpu.ccl.id
) -> (!tfrt.chain, !t.tensor) {
  %ch0 = tfrt.new.chain
  %device = tfrt_gpu.device.get CUDA, %rank
  %context = tfrt_gpu.context.primary %device

  %tensor = tfrt_dht.create_uninitialized_tensor.i32.1 [1: i64]
  %ch1 = tfrt_dht.set_tensor_with_constant_values.i32 %tensor, %ch0 [1: i32]
  %buffer:2 = tfrt_dht.get_buffer %tensor, %ch0
  %pinned = tfrt_gpu.mem.register %context, %buffer#0

  %ccl = tfrt_gpu.ccl.create %context, %rank, %count, %id
  %ch2 = tfrt_gpu.ccl.all_reduce %ccl, %pinned, %pinned, ncclInt32, ncclSum, %ch1

  %stream = tfrt_gpu.stream.create %context
  %ch3 = tfrt_gpu.ccl.execute %stream, %ccl, %ch2
  %ch4 = tfrt_gpu.stream.synchronize %stream, %ch3

  tfrt.return %ch4, %tensor : !tfrt.chain, !t.tensor
}

// CHECK-LABEL: --- Running 'all_reduce_test'
func @all_reduce_test() {
  %count = tfrt.constant.i32 2
  %id = tfrt_gpu.ccl.unique_id CUDA

  %rank0 = tfrt.constant.i32 0
  %rank1 = tfrt.constant.i32 1

  %ch0, %t0 = tfrt.call @all_reduce(%rank0, %count, %id)
              : (i32, i32, !tfrt_gpu.ccl.id) -> (!tfrt.chain, !t.tensor)
  %unused:2 = tfrt.call @all_reduce(%rank1, %count, %id)
              : (i32, i32, !tfrt_gpu.ccl.id) -> (!tfrt.chain, !t.tensor)

  // CHECK: DenseHostTensor dtype = i32, shape = [1], values = [2]
  %ch1 = tfrt_dht.print_tensor %t0, %ch0

  tfrt.return
}
