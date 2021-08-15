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

// RUN: bef_executor %s.bef | FileCheck %s

// CHECK-LABEL: --- Running 'tensor_io'
func @tensor_io() {
  // test_tensor.btf contains three tensors:
  // tensor 0: [[1, 2], [3, 4]], dtype=float32
  // tensor 1: [1, 2, 3, 4, 5], dtype=float32
  // tensor 2: [], dtype=float32

  %c0 = tfrt.new.chain
  %path = "tfrt_test.get_string"() { value = "backends/cpu/mlir_tests/mnist/test_data/test_tensor.btf" } : () -> !tfrt.string

  %zero = tfrt.constant.i32 0
  %one = tfrt.constant.i32 1
  %two = tfrt.constant.i32 2

  %t0 = "btf.read_dense_tensor.i32.2"(%path, %zero) : (!tfrt.string, i32) -> (!t.tensor)
  // CHECK-NEXT: shape = [2, 2], values = [1, 2, 3, 4]
  %c1 = tfrt_dht.print_tensor %t0, %c0

  %t1 = "btf.read_dense_tensor.i32.1"(%path, %one) : (!tfrt.string, i32) -> (!t.tensor)
  // CHECK-NEXT: shape = [5], values = [0, 1, 2, 3, 4]
  %c2 = tfrt_dht.print_tensor %t1, %c1

  %t2 = "btf.read_dense_tensor.i32.1"(%path, %two) : (!tfrt.string, i32) -> (!t.tensor)
  // CHECK-NEXT: shape = [0], values = []
  %c3 = tfrt_dht.print_tensor %t2, %c2

  tfrt.return
}

// CHECK-LABEL: --- Running 'tensor_io_invalid_path'
func @tensor_io_invalid_path() {
  %c0 = tfrt.new.chain
  %path = "tfrt_test.get_string"() { value = "/tmp/invalid_path" } : () -> !tfrt.string
  %zero = tfrt.constant.i32 0
  // expected-error @+1 {{failed to open file /tmp/invalid_path for reading}}
  %t0 = "btf.read_dense_tensor.i32.2"(%path, %zero) : (!tfrt.string, i32) -> (!t.tensor)

  tfrt.return
}
