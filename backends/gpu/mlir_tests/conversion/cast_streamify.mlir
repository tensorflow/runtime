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

// RUN: tfrt_gpu_opt %s -cast-tfrt-streamify | FileCheck %s

// CHECK-LABEL: @cast_to_event
func @cast_to_event(
  %arg0 : !tfrt.chain,
  %arg1 : !tfrt_gpu.stream
) -> (!tfrt.chain, !tfrt_gpu.event) {
  // CHECK: %[[ctx:.*]] = tfrt_gpu.stream.get_context %arg1
  // CHECK: %[[event:.*]] = tfrt_gpu.event.create %[[ctx]]
  // CHECK: %[[ch:.*]] = tfrt_gpu.event.record %[[event]], %arg1, %arg0
  %token = builtin.unrealized_conversion_cast %arg0, %arg1
      : !tfrt.chain, !tfrt_gpu.stream to !gpu.async.token
  %ch, %event = builtin.unrealized_conversion_cast %token
      : !gpu.async.token to !tfrt.chain, !tfrt_gpu.event
  // CHECK: tfrt.return %[[ch]], %[[event]] : !tfrt.chain, !tfrt_gpu.event
  tfrt.return %ch, %event : !tfrt.chain, !tfrt_gpu.event
}

// CHECK-LABEL: @const_cast
func @const_cast() -> (i32, i64, ui32, ui64) {
  %0 = arith.constant 42 : index
  // CHECK: %[[c1:.*]] = tfrt.constant.i32 42
  %1 = builtin.unrealized_conversion_cast %0 : index to i32
  // CHECK: %[[c2:.*]] = tfrt.constant.i64 42
  %2 = builtin.unrealized_conversion_cast %0 : index to i64
  // CHECK: %[[c3:.*]] = tfrt.constant.ui32 42
  %3 = builtin.unrealized_conversion_cast %0 : index to ui32
  // CHECK: %[[c4:.*]] = tfrt.constant.ui64 42
  %4 = builtin.unrealized_conversion_cast %0 : index to ui64
  // CHECK: tfrt.return %[[c1]], %[[c2]], %[[c3]], %[[c4]] : i32, i64, ui32, ui64
  tfrt.return %1, %2, %3, %4 : i32, i64, ui32, ui64
}
