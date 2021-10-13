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

// RUN: tfrt_gpu_opt %s \
// RUN:   -test-gpu-async-conversion \
// RUN:   -allow-unregistered-dialect \
// RUN: | FileCheck %s

func @test_wrap_async_execute() {

  // CHECK: "other.op"() : () -> ()
  "other.op"() : () -> ()
  // CHECK: "tfrt_gpu_conversion.async.execute"() ( {
  // CHECK: ^bb0(%arg0: !tfrt.chain, %arg1: !tfrt_gpu.stream):
  // CHECK:   "wrap.op"() : () -> ()
  // CHECK:   "wrap.op"() : () -> ()
  // CHECK:   tfrt.return %arg0 : !tfrt.chain
  // CHECK: }) : () -> ()
  "wrap.op"() : () -> ()
  "wrap.op"() : () -> ()

  // CHECK: "other.op"() : () -> ()
  "other.op"() : () -> ()
  // CHECK: "tfrt_gpu_conversion.async.execute"() ( {
  // CHECK: ^bb0(%arg0: !tfrt.chain, %arg1: !tfrt_gpu.stream):
  // CHECK:   "wrap.op"() : () -> ()
  // CHECK:   "wrap.op"() : () -> ()
  // CHECK:   tfrt.return %arg0 : !tfrt.chain
  // CHECK: }) : () -> ()
  "wrap.op"() : () -> ()
  "wrap.op"() : () -> ()

  // CHECK: return
  return
}

func @test_fold_memref_view(%arg0 : memref<64xi8>) {
  %zero = arith.constant 0 : index
  // CHECK-NOT: memref.view
  // CHECK: builtin.unrealized_conversion_cast %arg0
  // CHECK-SAME : memref<64xi8> to !tfrt_gpu.buffer
  %view = "memref.view"(%arg0, %zero) : (memref<64xi8>, index) -> (memref<4x4xf32>)
  // CHECK: return
  return
}
