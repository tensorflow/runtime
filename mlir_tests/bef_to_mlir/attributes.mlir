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

// RUN: tfrt_translate -bef-to-mlir %s.bef | tfrt_opt -allow-unregistered-dialect | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @integer1.constant() -> i1
func @integer1.constant() -> i1 {
  // CHECK-NEXT: [[REG:%.*]] = tfrt.constant.i1 true
  // CHECK-NEXT: tfrt.return [[REG]] : i1

  %x = tfrt.constant.i1 true
  tfrt.return %x : i1
}

// CHECK-LABEL: func @integer32.constant() -> i32
func @integer32.constant() -> i32 {
  // CHECK-NEXT: [[REG:%.*]] = tfrt.constant.i32 42
  // CHECK-NEXT: tfrt.return [[REG]] : i32

  %x = tfrt.constant.i32 42
  tfrt.return %x : i32
}

// CHECK-LABEL: func @integer64.constant() -> i64
func @integer64.constant() -> i64 {
  // CHECK-NEXT: [[REG:%.*]] = tfrt.constant.i64 42
  // CHECK-NEXT: tfrt.return [[REG]] : i64

  %x = tfrt.constant.i64 42
  tfrt.return %x : i64
}

// CHECK-LABEL: func @float32.constant() -> f32
func @float32.constant() -> f32 {
  // CHECK-NEXT: [[REG:%.*]] = "simple.op"() {value = {{.*}} : f32} : () -> f32
  // CHECK-NEXT: tfrt.return [[REG]] : f32

  %x = "simple.op"() {value = -0.1 : f32} : () -> f32
  tfrt.return %x : f32
}

// CHECK-LABEL: func @string.constant() -> f32
func @string.constant() -> f32 {
  // CHECK-NEXT: [[REG:%.*]] = "simple.op"()
  // CHECK-SAME: {value = "string constant"} : () -> f32
  // CHECK-NEXT: tfrt.return [[REG]] : f32

  %x = "simple.op"() {value = "string constant"} : () -> f32
  tfrt.return %x : f32
}

// CHECK-LABEL: func @array.constant() -> i32
func @array.constant() -> i32 {
  // CHECK: [true]
  // CHECK: [1 : i8]
  // CHECK: [1 : i16]
  // CHECK: [1 : i32]
  // CHECK: [1]
  // CHECK: [1 : ui8]
  // CHECK: [1 : ui16]
  // CHECK: [1 : ui32]
  // CHECK: [1 : ui64]
  // CHECK: [1.000000e+00 : bf16]
  // CHECK: [1.000000e+00 : f16]
  // CHECK: [1.000000e+00 : f32]
  // CHECK: [1.000000e+00]
  %0 = "simple.op"() {value = [true]} : () -> i32
  %1 = "simple.op"() {value = [1 : i8]} : () -> i32
  %2 = "simple.op"() {value = [1 : i16]} : () -> i32
  %3 = "simple.op"() {value = [1 : i32]} : () -> i32
  %4 = "simple.op"() {value = [1 : i64]} : () -> i32
  %5 = "simple.op"() {value = [1 : ui8]} : () -> i32
  %6 = "simple.op"() {value = [1 : ui16]} : () -> i32
  %7 = "simple.op"() {value = [1 : ui32]} : () -> i32
  %8 = "simple.op"() {value = [1 : ui64]} : () -> i32
  %9 = "simple.op"() {value = [1.0 : bf16]} : () -> i32
  %10 = "simple.op"() {value = [1.0 : f16]} : () -> i32
  %11 = "simple.op"() {value = [1.0 : f32]} : () -> i32
  %12 = "simple.op"() {value = [1.0 : f64]} : () -> i32

  // CHECK: {value = [dense<0> : tensor<4xi64>, dense<1> : tensor<4xi64>]} : () -> i32
  %d = "simple.op"() {value = [dense<0> : tensor<4xi64>, dense<1> : tensor<4xi64>]} : () -> i32
  tfrt.return %d : i32
}

// CHECK-LABEL: func @aggregate.constant() -> i32
func @aggregate.constant() -> i32 {
  // CHECK-NEXT: [[REG:%.*]] = "simple.op"()
  // CHECK-SAME: {value = [1 : i32, [2 : i32, "string"]]} : () -> i32
  // CHECK-NEXT: tfrt.return [[REG]] : i32

  %x = "simple.op"() {value = [1: i32, [2: i32, "string"]]} : () -> i32
  tfrt.return %x : i32
}

// CHECK-LABEL: func @type.attribute() -> (i32, i32, i32)
func @type.attribute() -> (i32, i32, i32) {
  // CHECK-NEXT: [[REG0:%.*]] = "get_width"() {value = i32} : () -> i32
  // CHECK-NEXT: [[REG1:%.*]] = "get_width"() {value = f64} : () -> i32
  // CHECK-NEXT: [[REG2:%.*]] = "get_widths"() {value = [f64, f32]} : () -> i32
  // CHECK-NEXT: tfrt.return [[REG0]], [[REG1]], [[REG2]] : i32, i32, i32

  %x = "get_width"() {value = i32} : () -> i32
  %y = "get_width"() {value = f64} : () -> i32
  %z = "get_widths"() {value = [f64, f32]} : () -> i32
  tfrt.return %x, %y, %z: i32, i32, i32
}


// CHECK-LABEL: func @dense_elements.constant()
func @dense_elements.constant() {
  // CHECK: dense<true> : tensor<1xi1>
  // CHECK: dense<1> : tensor<1xi8>
  // CHECK: dense<1> : tensor<1xi16>
  // CHECK: dense<1> : tensor<1xi32>
  // CHECK: dense<1> : tensor<1xi64>
  // CHECK: dense<1> : tensor<1xui8>
  // CHECK: dense<1> : tensor<1xui16>
  // CHECK: dense<1> : tensor<1xui32>
  // CHECK: dense<1> : tensor<1xui64>
  // CHECK: dense<1.000000e+00> : tensor<1xbf16>
  // CHECK: dense<1.000000e+00> : tensor<1xf16>
  // CHECK: dense<1.000000e+00> : tensor<1xf32>
  // CHECK: dense<1.000000e+00> : tensor<1xf64>
  // CHECK: dense<(1.000000e+00,2.000000e+00)> : tensor<1xcomplex<f32>>
  // CHECK: dense<(1.000000e+00,2.000000e+00)> : tensor<1xcomplex<f64>>
  %0 = "simple.op"() {value = dense<true> : tensor<1xi1>} : () -> i32
  %1 = "simple.op"() {value = dense<1> : tensor<1xi8>} : () -> i32
  %2 = "simple.op"() {value = dense<1> : tensor<1xi16>} : () -> i32
  %3 = "simple.op"() {value = dense<1> : tensor<1xi32>} : () -> i32
  %4 = "simple.op"() {value = dense<1> : tensor<1xi64>} : () -> i32
  %5 = "simple.op"() {value = dense<1> : tensor<1xui8>} : () -> i32
  %6 = "simple.op"() {value = dense<1> : tensor<1xui16>} : () -> i32
  %7 = "simple.op"() {value = dense<1> : tensor<1xui32>} : () -> i32
  %8 = "simple.op"() {value = dense<1> : tensor<1xui64>} : () -> i32
  %9 = "simple.op"() {value = dense<1.0> : tensor<1xbf16>} : () -> i32
  %10 = "simple.op"() {value = dense<1.0> : tensor<1xf16>} : () -> i32
  %11 = "simple.op"() {value = dense<1.0> : tensor<1xf32>} : () -> i32
  %12 = "simple.op"() {value = dense<1.0> : tensor<1xf64>} : () -> i32
  %13 = "simple.op"() {value = dense<(1.0, 2.0)> : tensor<1xcomplex<f32>>} : () -> i32
  %14 = "simple.op"() {value = dense<(1.0, 2.0)> : tensor<1xcomplex<f64>>} : () -> i32

  tfrt.return
}

// CHECK-LABEL: @shape_attr
func @shape_attr() {
  // CHECK: #corert.shape<2x?x3>
  "simple.op"() {shape = #corert.shape<2x?x3>} : () -> ()
  // CHECK: #corert.shape<*>
  "simple.op"() {shape = #corert.shape<*>} : () -> ()
  tfrt.return
}

// CHECK-LABEL: @data_type_attr
func @data_type_attr() {
  // CHECK: i1
  // CHECK: i8
  // CHECK: i16
  // CHECK: i32
  // CHECK: i64
  // CHECK: ui8
  // CHECK: ui16
  // CHECK: ui32
  // CHECK: ui64
  // CHECK: bf16
  // CHECK: !corert.quint8
  // CHECK: !corert.quint16
  // CHECK: !corert.qint8
  // CHECK: !corert.qint16
  // CHECK: !corert.qint32
  // CHECK: complex<f32>
  // CHECK: complex<f64>
  // CHECK: !corert.string
  // CHECK: !corert.resource
  // CHECK: !corert.variant
  "simple.op"() {type = i1} : () -> ()
  "simple.op"() {type = i8} : () -> ()
  "simple.op"() {type = i16} : () -> ()
  "simple.op"() {type = i32} : () -> ()
  "simple.op"() {type = i64} : () -> ()
  "simple.op"() {type = ui8} : () -> ()
  "simple.op"() {type = ui16} : () -> ()
  "simple.op"() {type = ui32} : () -> ()
  "simple.op"() {type = ui64} : () -> ()
  "simple.op"() {type = bf16} : () -> ()
  "simple.op"() {type = f16} : () -> ()
  "simple.op"() {type = f32} : () -> ()
  "simple.op"() {type = f64} : () -> ()
  "simple.op"() {type = !corert.quint8} : () -> ()
  "simple.op"() {type = !corert.quint16} : () -> ()
  "simple.op"() {type = !corert.qint8} : () -> ()
  "simple.op"() {type = !corert.qint16} : () -> ()
  "simple.op"() {type = !corert.qint32} : () -> ()
  "simple.op"() {type = complex<f32>} : () -> ()
  "simple.op"() {type = complex<f64>} : () -> ()
  "simple.op"() {type = !corert.string} : () -> ()
  "simple.op"() {type = !corert.resource} : () -> ()
  "simple.op"() {type = !corert.variant} : () -> ()
  tfrt.return
}
