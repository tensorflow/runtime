// Copyright 2022 The TensorFlow Runtime Authors
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

// RUN: bef_executor %s.bef | FileCheck %s --dump-input=always

module @multiply attributes { tfrt.compiled } {
  func.func private @multiply.cc(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>)
    attributes { rt.custom_call = "testlib.multiply" }

  func.func @main(%input: memref<?x?xf32>) -> memref<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.dim %input, %c0 : memref<?x?xf32>
    %1 = memref.dim %input, %c1 : memref<?x?xf32>

    // Reverse dimension order to test invalid custom call arguments below.
    %output = memref.alloc(%1, %0) : memref<?x?xf32>

    func.call @multiply.cc(%input, %output) { cst = 2.0 : f32 }
      : (memref<?x?xf32>, memref<?x?xf32>) -> ()

    func.return %output : memref<?x?xf32>
  }
}

// Prints all attributes passed to the custom call handler.
module @print_attrs attributes { tfrt.compiled } {
  func.func private @print_attrs.cc()
     attributes { rt.custom_call = "testlib.print_attrs" }

  func.func @main() {
    func.call @print_attrs.cc() {
      i32 = 101 : i32,
      i64 = 102 : i64,
      f32 = 1.0 : f32,
      f64 = 2.0 : f64,
      i32_arr = dense<[101, 102, 103, 104]> : tensor<4xi32>,
      i64_arr = dense<[105, 106, 107, 108]> : tensor<4xi64>,
      f32_arr = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>,
      f64_arr = dense<[5.0, 6.0, 7.0, 8.0]> : tensor<4xf64>,
      str = "some string"
    } : () -> ()

    func.return
  }
}

// Prints all arguments passed to the custom call handler. Intended for testing
// custom call handlers with variadic arguments.
module @variadic_args attributes { tfrt.compiled } {
  func.func private @variadic_args.cc(%arg0: memref<?xf32>,
                                      %arg1: memref<?xf32>)
     attributes { rt.custom_call = "testlib.variadic_args" }

  func.func private @memref_and_variadic_args.cc(%arg1: memref<?xf32>,
                                                 %arg2: memref<?xf32>)
    attributes { rt.custom_call = "testlib.memref_and_variadic_args" }

  func.func @main() {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    %i1 = memref.alloc(%c1) : memref<?xf32>
    %i2 = memref.alloc(%c2) : memref<?xf32>

    func.call @variadic_args.cc(%i1, %i2)
      : (memref<?xf32>, memref<?xf32>) -> ()

    func.call @memref_and_variadic_args.cc(%i1, %i2)
      : ( memref<?xf32>, memref<?xf32>) -> ()

    memref.dealloc %i1 : memref<?xf32>
    memref.dealloc %i2 : memref<?xf32>

    func.return
  }
}

// CHECK: --- Running 'compiled_custom_call'
func.func @compiled_custom_call() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  // Allocate and initialize input tensor.
  %input = tfrt_dht.create_uninitialized_tensor.f32.2 [16 : i64, 16 : i64]
  %ch1 = tfrt_dht.fill_tensor_with_constant.f32 %input, %ch0 1.0 : f32

  // Allocate and initialize expected tensor.
  %expected = tfrt_dht.create_uninitialized_tensor.f32.2 [16 : i64, 16 : i64]
  %ch2 = tfrt_dht.fill_tensor_with_constant.f32 %expected, %ch1 2.0 : f32

  // Compile a kernel with a custom call.
  %executable = jitrt.compile { kernel = @multiply::@main }

  // Execute compiled kernel with tensor operands.
  %output = jitrt.execute %executable[%ch1](%input) : (!t.tensor) -> !t.tensor

  // Wait for the execution completion and compare result with expected.
  %cmp, %cmp_ch = "tfrt_dht.tensor_allclose.f32"(%expected, %output, %ch2)
    : (!t.tensor, !t.tensor, !tfrt.chain) -> (i1, !tfrt.chain)

  // CHECK: int1 = 1
  %printed = tfrt.print.i1 %cmp, %cmp_ch

  tfrt.return %printed : !tfrt.chain
}

// CHECK: --- Running 'compiled_custom_call_error'
func.func @compiled_custom_call_error() -> !t.tensor {
  %ch0 = tfrt.new.chain

  // Allocate and initialize input tensor.
  %input = tfrt_dht.create_uninitialized_tensor.f32.2 [16 : i64, 4 : i64]
  %ch1 = tfrt_dht.fill_tensor_with_constant.f32 %input, %ch0 1.0 : f32

  // Compile a kernel with a custom call.
  %executable = jitrt.compile { kernel = @multiply::@main }

  // Execute compiled kernel with tensor operands.
  %output = jitrt.execute %executable[%ch1](%input) : (!t.tensor) -> !t.tensor

  // CHECK: returned <<error: compiled kernel run time error:
  // CHECK-SAME: custom call 'testlib.multiply' failed>>
  tfrt.return %output : !t.tensor
}

// CHECK: --- Running 'compiled_custom_call_print_attrs'
func.func @compiled_custom_call_print_attrs() {
  %ch0 = tfrt.new.chain

  // CHECK: Called from: jitrt.execute
  // CHECK: i32: 101
  // CHECK: i64: 102
  // CHECK: f32: 1.000000e+00
  // CHECK: f64: 2.000000e+00
  // CHECK: i32[4] 101, 102, 103, 104
  // CHECK: i64[4] 105, 106, 107, 108
  // CHECK: f32[4] 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00
  // CHECK: f64[4] 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00
  // CHECK: str: some string
  %executable = jitrt.compile { kernel = @print_attrs::@main }
  jitrt.execute %executable[%ch0]() : () -> ()

  tfrt.return
}

// CHECK: --- Running 'compiled_custom_call_variadic_args'
func.func @compiled_custom_call_variadic_args() {
  %ch0 = tfrt.new.chain

  // CHECK: Number of variadic arguments: 2
  // CHECK: arg[0]: MemrefDesc: dtype: f32 offset: 0 sizes: [1] strides: [1]
  // CHECK: arg[1]: MemrefDesc: dtype: f32 offset: 0 sizes: [2] strides: [1]

  // CHECK: arg: MemrefDesc: dtype: f32 offset: 0 sizes: [1] strides: [1]
  // CHECK: Number of variadic arguments: 1
  // CHECK: arg[0]: MemrefDesc: dtype: f32 offset: 0 sizes: [2] strides: [1]
  %executable = jitrt.compile { kernel = @variadic_args::@main }
  jitrt.execute %executable[%ch0]() : () -> ()

  tfrt.return
}
