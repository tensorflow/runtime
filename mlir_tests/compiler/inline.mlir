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

// RUN: tfrt_opt -inline %s | FileCheck %s -dump-input=fail

// CHECK-NOT: @callee
func private @callee(%ch: !tfrt.chain) -> (!tfrt.chain, !corert.tensorhandle) {
  %0 = corert.const_dense_tensor dense<1> : tensor<i32>
  tfrt.return %ch, %0 : !tfrt.chain, !corert.tensorhandle
}

// CHECK-LABEL: @caller
// CHECK-SAME: ([[ch:%.*]]: !tfrt.chain)
func @caller(%ch: !tfrt.chain) -> (!tfrt.chain, !corert.tensorhandle) {
  // CHECK-NOT: tfrt.call
  // CHECK: [[const:%.*]] = corert.const_dense_tensor dense<1> : tensor<i32>
  // CHECK-NEXT: tfrt.return [[ch]], [[const]]
  %0, %1 = tfrt.call @callee(%ch) : (!tfrt.chain) -> (!tfrt.chain, !corert.tensorhandle)
  tfrt.return %0, %1 : !tfrt.chain, !corert.tensorhandle
}
