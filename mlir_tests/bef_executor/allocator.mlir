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

// RUN: bef_executor_lite --host_allocator_type test_fixed_size_1k %s.bef 2>&1 | FileCheck %s

// CHECK: --- Running 'dense_array'
func @dense_array() {
  // CHECK: Allocating 192 bytes
  // CHECK: Attempted to allocate 4160 bytes.
  // CHECK: Allocating 192 bytes

  %x0 = tfrt_dht.create_uninitialized_tensor.f32.1 [32 : i64]
  %x1 = tfrt_dht.create_uninitialized_tensor.f32.1 [32 : i64]
  // expected-error @+1 {{runtime error: Cannot allocate tensor}}
  %x2 = tfrt_dht.create_uninitialized_tensor.f32.1 [1024 : i64]

  tfrt.return
}
