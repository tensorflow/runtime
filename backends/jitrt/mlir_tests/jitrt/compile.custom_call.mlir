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

module @kernels attributes { tfrt.compiled } {
  func.func @multiply(%input: memref<?x?xf32>) -> memref<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.dim %input, %c0 : memref<?x?xf32>
    %1 = memref.dim %input, %c1 : memref<?x?xf32>

    // Reverse dimension order to test invalid custom call arguments below.
    %output = memref.alloc(%1, %0) : memref<?x?xf32>

    %status = rt.custom_call "testlib.multiply"(%input, %output)
      { cst = 2.0 : f32 }
      : (memref<?x?xf32>, memref<?x?xf32>) -> ()
    %ok = rt.is_ok %status
    cf.assert %ok, "failed to call custom call 'testlib.multiply'"

    func.return %output : memref<?x?xf32>
  }

  func.func @print_attrs() {
    %status = rt.custom_call "testlib.print_attrs"()
      {
        i8  = 100 : i8,
        i32 = 101 : i32,
        i64 = 102 : i64,
        f32 = 123.00 : f32,
        f64 = 456.00 : f64
      }
      : () -> ()
    %ok = rt.is_ok %status
    cf.assert %ok, "failed to call custom call 'testlib.print_attrs'"

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
  %executable = jitrt.compile { kernel = @kernels::@multiply }

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
  %executable = jitrt.compile { kernel = @kernels::@multiply }

  // Execute compiled kernel with tensor operands.
  %output = jitrt.execute %executable[%ch1](%input) : (!t.tensor) -> !t.tensor

  // CHECK: returned <<error: compiled kernel run time error:
  // CHECK-SAME: failed to call custom call 'testlib.multiply'>>
  tfrt.return %output : !t.tensor
}

// CHECK: --- Running 'compiled_custom_call_attrs'
func.func @compiled_custom_call_attrs() {
  %ch0 = tfrt.new.chain

  // CHECK:  i8: 100
  // CHECK: i32: 101
  // CHECK: i64: 102
  // CHECK: f32: 1.230000e+02
  // CHECK: f64: 4.560000e+02
  %executable = jitrt.compile { kernel = @kernels::@print_attrs }
  jitrt.execute %executable[%ch0]() : () -> ()

  tfrt.return
}
