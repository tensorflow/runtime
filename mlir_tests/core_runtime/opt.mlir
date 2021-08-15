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

// RUN: tfrt_opt %s | tfrt_opt | FileCheck %s

// CHECK-LABEL: func @basic
// CHECK-SAME: ([[arg:%.*]]: !corert.tensorhandle) -> !corert.tensorhandle
func @basic(%arg : !corert.tensorhandle) -> !corert.tensorhandle {
  // CHECK: [[chain:%.*]] = tfrt.new.chain
  %ch = tfrt.new.chain

  // CHECK: [[device:%.*]] = corert.get_op_handler %0 "cpu"
   %cpu = corert.get_op_handler %ch "cpu"

  // CHECK: [[r0:%.*]] = corert.executeop([[device]]) "some.op"([[arg]])
  // CHECK-SAME: {shape = [1, 65536], values = [1.000000e+00 : f32]} : 1
  %res0 = corert.executeop(%cpu) "some.op"(%arg)
    {shape = [1, 65536], values = [1.0 : f32]} : 1

  // CHECK: corert.executeop([[device]]) "some.op"([[arg]])
  corert.executeop(%cpu) "some.op"(%arg)

  // CHECK: [[r2:%.*]] = corert.executeop([[device]]) "some.op"() : 1
  %res2 = corert.executeop(%cpu) "some.op"() : 1

  // CHECK: [[op_ch4:%.*]], [[r3:%.*]] = corert.executeop.seq([[device]], [[chain]]) "some.op"() : 1
  %op_ch3, %res3 = corert.executeop.seq(%cpu, %ch) "some.op"() : 1

  tfrt.return %res2 : !corert.tensorhandle
}

// CHECK-LABEL: func @const_string_tensor
func @const_string_tensor() -> !corert.tensorhandle {
  // CHECK: corert.const_string_tensor {shape = [1, 2], value = ["const", "string"]}
  %res = corert.const_string_tensor {shape = [1, 2], value = ["const", "string"]}
  tfrt.return %res : !corert.tensorhandle
}

// CHECK-LABEL: func @corert_shape_attr
func @corert_shape_attr() {
  // CHECK: #corert.shape<1x?x3>
  "corert.test"() {shape = #corert.shape<1x?x3>} : () -> ()
  tfrt.return
}

// CHECK-LABEL: func @corert_string_type_attr
func @corert_string_type_attr() {
  // CHECK: !corert.string
  "corert.test"() {type = !corert.string} : () -> ()
  tfrt.return
}

// CHECK-LABEL: func @corert_create_dense_tensor
func @corert_create_dense_tensor() {
  // CHECK: corert.create_dense_tensor.ui64 {shape = [1, 1], value = [1 : ui64]}
  %0 = corert.create_dense_tensor.ui64 {shape = [1, 1], value = [1 : ui64]}
  tfrt.return
}
