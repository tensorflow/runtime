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
// RUN:   -tfrt-streamify-ops=ops=wrap.op,tfrt.call \
// RUN:   -allow-unregistered-dialect \
// RUN: | FileCheck %s

// CHECK-LABEL: @test_wrap_streamify
func.func @test_wrap_streamify() {

  // CHECK: "other.op"() : () -> ()
  "other.op"() : () -> ()

  // CHECK: tfrt_gpu.streamify {
  // CHECK: ^bb0(%arg0: !tfrt.chain, %arg1: !tfrt_gpu.stream):
  // CHECK:   "wrap.op"() : () -> ()
  // CHECK:   "wrap.op"() : () -> ()
  // CHECK:   tfrt.return %arg0 : !tfrt.chain
  // CHECK: }
  "wrap.op"() : () -> ()
  "wrap.op"() : () -> ()

  // CHECK: "other.op"() : () -> ()
  "other.op"() : () -> ()

  // CHECK: %[[value:.*]] = tfrt_gpu.streamify {
  // CHECK: ^bb0(%arg0: !tfrt.chain, %arg1: !tfrt_gpu.stream):
  // CHECK:   %[[results:.*]]:2 = tfrt.call @returns_values() : () -> (f32, f32)
  // CHECK:   tfrt.call @takes_argument(%[[results]]#0) : (f32) -> ()
  // CHECK:   tfrt.return %arg0, %[[results]]#1 : !tfrt.chain, f32
  // CHECK: } : f32
  %values:2 = tfrt.call @returns_values() : () -> (f32, f32)
  tfrt.call @takes_argument(%values#0) : (f32) -> ()

  // CHECK: "other.op"() : () -> ()
  "other.op"() : () -> ()

  // CHECK: tfrt_gpu.streamify {
  // CHECK: ^bb0(%arg0: !tfrt.chain, %arg1: !tfrt_gpu.stream):
  // CHECK:   tfrt.call @takes_argument(%[[value]]) : (f32) -> ()
  // CHECK:   tfrt.return %arg0 : !tfrt.chain
  // CHECK: }
  tfrt.call @takes_argument(%values#1) : (f32) -> ()

  // CHECK: return
  func.return
}

func.func private @returns_values() -> (f32, f32)
func.func private @takes_argument(%arg0: f32)

// Note on 'DISABLED' below: There is temporarily no pass in TFRT GPU that
// exercises the memref conversion patterns. TODO(csigg): Enable checks again.

// CHECK-LABEL: @test_fold_memref_view
func.func @test_fold_memref_view(%arg0: memref<64xi8>) -> memref<4x4xf32> {
  %zero = arith.constant 0 : index
  // DISABLED-NOT: memref.view
  // DISABLED: %[[buffer:.*]] = builtin.unrealized_conversion_cast %arg0 : memref<64xi8> to !tfrt_gpu.buffer
  // DISABLED: %[[memref:.*]] = builtin.unrealized_conversion_cast %[[buffer]] : !tfrt_gpu.buffer to memref<4x4xf32>
  %view = memref.view %arg0[%zero][] : memref<64xi8> to memref<4x4xf32>
  // DISABLED: return %[[memref]]
  func.return %view : memref<4x4xf32>
}

// CHECK-LABEL: @test_fold_memref_cast
func.func @test_fold_memref_cast(%arg0: memref<64xi8>) -> memref<8x8xi8> {
  // DISABLED-NOT: memref.reinterpret_cast
  // DISABLED: %[[buffer:.*]] = builtin.unrealized_conversion_cast %arg0 : memref<64xi8> to !tfrt_gpu.buffer
  // DISABLED: %[[memref:.*]] = builtin.unrealized_conversion_cast %[[buffer]] : !tfrt_gpu.buffer to memref<8x8xi8>
  %cast = memref.reinterpret_cast %arg0
    to offset: [0], sizes: [8, 8], strides: [8, 1]
    : memref<64xi8> to memref<8x8xi8>
  // DISABLED: return %[[memref]]
  func.return %cast : memref<8x8xi8>
}

// CHECK-LABEL: @test_rewrite_alloc
func.func @test_rewrite_alloc() {
  // DISABLED: %[[memref:.*]] = gpu.alloc  () : memref<64xi8>
  %memref = memref.alloc() : memref<64xi8>
  // DISABLED: "other.op"() : () -> ()
  "other.op"() : () -> ()
  // DISABLED: gpu.dealloc  %[[memref]] : memref<64xi8>
  memref.dealloc %memref : memref<64xi8>
  // DISABLED: %[[tmp:.*]] = gpu.alloc  () : memref<64xi8>
  %temp = memref.alloca() : memref<64xi8>
  // DISABLED: return
  func.return
}
