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
  func.func private @multiply.cc(%arg0: memref<?x?xf32>,
                                 %arg1: memref<?x?xf32>)
    attributes { rt.dynamic, rt.custom_call = "testlib.multiply" }

  func.func private @multiply.x3.cc(%arg0: memref<?x?xf32>,
                                    %arg1: memref<?x?xf32>)
    attributes { rt.dynamic, rt.custom_call = "testlib.multiply.x3" }

  func.func @main(%input: memref<?x?xf32>) -> (memref<?x?xf32>,
                                               memref<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.dim %input, %c0 : memref<?x?xf32>
    %1 = memref.dim %input, %c1 : memref<?x?xf32>

    // Reverse dimension order to test invalid custom call arguments below.
    %out0 = memref.alloc(%1, %0) : memref<?x?xf32>
    %out1 = memref.alloc(%1, %0) : memref<?x?xf32>

    func.call @multiply.cc(%input, %out0) { cst = 2.0 : f32 }
      : (memref<?x?xf32>, memref<?x?xf32>) -> ()

    func.call @multiply.x3.cc(%input, %out1)
      : (memref<?x?xf32>, memref<?x?xf32>) -> ()

    func.return %out0, %out1 : memref<?x?xf32>, memref<?x?xf32>
  }
}

// Test that custom calls with incorrect signatures emit error messages.
module @multiply_errors attributes { tfrt.compiled } {

  // "testlib.multiply" custom call expects two arguments.
  func.func private @multiply.wrong_args.cc(
      %arg0: memref<?x?xf32>,
      %arg1: memref<?x?xf32>,
      %arg1: memref<?x?xf32>
    ) attributes { rt.dynamic, rt.custom_call = "testlib.multiply" }

  func.func @main(%input: memref<?x?xf32>) -> memref<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.dim %input, %c0 : memref<?x?xf32>
    %1 = memref.dim %input, %c1 : memref<?x?xf32>

    %out = memref.alloc(%0, %1) : memref<?x?xf32>

    func.call @multiply.wrong_args.cc(%input, %input, %out) { cst = 2.0 : f32 }
      : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()

    return %out : memref<?x?xf32>
  }
}

// Prints all attributes passed to the custom call handler.
module @print_attrs attributes { tfrt.compiled } {
  func.func private @print_attrs.cc()
     attributes { rt.dynamic, rt.custom_call = "testlib.print_attrs" }

  func.func @main() {
    func.call @print_attrs.cc() {
      i32 = 101 : i32,
      i64 = 102 : i64,
      f32 = 1.0 : f32,
      f64 = 2.0 : f64,
      i32_dense = dense<[101, 102, 103, 104]> : tensor<4xi32>,
      i64_dense = dense<[105, 106, 107, 108]> : tensor<4xi64>,
      f32_dense = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>,
      f64_dense = dense<[5.0, 6.0, 7.0, 8.0]> : tensor<4xf64>,
      i64_2d_dense = dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>,
      i32_array = [101 : i32, 102 : i32, 103 : i32, 104 : i32],
      i64_array = [105, 106, 107, 108],
      f32_array = [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32],
      f64_array = [5.0, 6.0, 7.0, 8.0],
      i64_dense_array = array<i64: 1, 2, 3>,
      empty_array = [],
      str = "some string",
      aaa = "these are unused attributes to test",
      mmm = "that custom call only decodes attributes",
      xxx = "defined in the custom call binding and",
      zzz = "ignores everything else"
    } : () -> ()

    func.return
  }
}

module @print_variant_attrs attributes { tfrt.compiled } {
  func.func private @print_variant_attrs.cc()
     attributes { rt.dynamic, rt.custom_call = "testlib.print_variant_attrs" }

  func.func @main() {
    func.call @print_variant_attrs.cc() {
      i32 = 101 : i32,
      f32 = 1.0 : f32,
      i32_array = [101 : i32, 102 : i32, 103 : i32, 104 : i32],
      i64_array = [105, 106, 107, 108],
      str = "some string"
    } : () -> ()

    func.return
  }
}

// Prints dialect specific attributes passed to the custom call handler.
module @print_dialect_attrs attributes { tfrt.compiled } {
  func.func private @print_dialect_attrs.cc()
     attributes { rt.dynamic, rt.custom_call = "testlib.print_dialect_attrs" }

  func.func @main() {

    func.call @print_dialect_attrs.cc() {
      enum = #testlib.enum_type<Baz>,
      runtime_enum = #testlib.another_enum_type<Bar>,
      dims = #testlib.pair_of_dims<2, [1, 1], [2, 2]>
    } : () -> ()

    func.return
  }
}

// Check that direct custom call handler for "testlib.print_attrs" skips
// attributes names checks.
module @direct_print_attrs attributes { tfrt.compiled } {
  func.func private @print_attrs.cc()
     attributes { rt.custom_call = "testlib.print_attrs" }

  func.func @main() {
    func.call @print_attrs.cc() {
      invalid_attr_name_i32 = 101 : i32,
      invalid_attr_name_i64 = 102 : i64,
      invalid_attr_name_f32 = 1.0 : f32,
      invalid_attr_name_f64 = 2.0 : f64,
      invalid_attr_name_i32_dense = dense<[101, 102, 103, 104]> : tensor<4xi32>,
      invalid_attr_name_i64_dense = dense<[105, 106, 107, 108]> : tensor<4xi64>,
      invalid_attr_name_f32_dense = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>,
      invalid_attr_name_f64_dense = dense<[5.0, 6.0, 7.0, 8.0]> : tensor<4xf64>,
      invalid_attr_name_i64_2d_dense =
        dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>,
      invalid_attr_name_i32_arr = [101 : i32, 102 : i32, 103 : i32, 104 : i32],
      invalid_attr_name_i64_arr = [105, 106, 107, 108],
      invalid_attr_name_f32_arr = [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32],
      invalid_attr_name_f64_arr = [5.0, 6.0, 7.0, 8.0],
      invalid_attr_name_i64_dense_arr = array<i64: 1, 2, 3>,
      invalid_attr_name_empty_arr = [],
      invalid_attr_name_str = "some string"
    } : () -> ()

    func.return
  }
}

// Prints all arguments passed to the custom call handler. Intended for testing
// custom call handlers with variadic arguments.
module @variadic_args attributes { tfrt.compiled } {
  func.func private @variadic_args.cc(
      %arg0: i32,
      %arg1: i64,
      %arg2: f32,
      %arg3: f64,
      %arg4: memref<?xi64>,
      %arg5: memref<?xf32>,
      %arg6: memref<16x3xf32, affine_map<(d0, d1) -> (d0 + d1 * 16)>>)
    attributes { rt.dynamic, rt.custom_call = "testlib.variadic_args" }

  func.func private @memref_and_variadic_args.cc(
      %arg0: memref<?xi64>,
      %arg1: i32,
      %arg2: i64,
      %arg3: f32,
      %arg4: f64,
      %arg5: memref<?xf32>)
    attributes { rt.dynamic, rt.custom_call = "testlib.memref_and_variadic_args" }

  func.func @main() {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    %arg0 = arith.constant 123 : i32
    %arg1 = arith.constant 456 : i64
    %arg2 = arith.constant 1.0 : f32
    %arg3 = arith.constant 2.0 : f64
    %arg4 = memref.alloc(%c1) : memref<?xi64>
    %arg5 = memref.alloc(%c2) : memref<?xf32>
    %arg6 = memref.alloc() : memref<3x16xf32>

    %mem = memref.reinterpret_cast %arg6
      to offset: [0], sizes: [16, 3], strides: [1, 16] : memref<3x16xf32>
      to memref<16x3xf32, affine_map<(d0, d1) -> (d0 + d1 * 16)>>

    func.call @variadic_args.cc(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %mem)
      : (i32, i64, f32, f64, memref<?xi64>, memref<?xf32>,
         memref<16x3xf32, affine_map<(d0, d1) -> (d0 + d1 * 16)>>) -> ()

    func.call @memref_and_variadic_args.cc(%arg4, %arg0, %arg1, %arg2, %arg3,
                                           %arg5)
      : (memref<?xi64>, i32, i64, f32, f64, memref<?xf32>) -> ()

    memref.dealloc %arg4 : memref<?xi64>
    memref.dealloc %arg5 : memref<?xf32>
    memref.dealloc %arg6 : memref<3x16xf32>

    func.return
  }
}

module @variant_arg attributes { tfrt.compiled } {
  func.func private @variant_arg.cc(%arg0: i32, %arg1: i64, %arg2: memref<?xi64>)
    attributes { rt.dynamic, rt.custom_call = "testlib.variant_arg" }

  func.func @main() {
    %arg0 = arith.constant 123 : i32
    %arg1 = arith.constant 456 : i64

    %c1 = arith.constant 1 : index
    %arg2 = memref.alloc(%c1) : memref<?xi64>

    func.call @variant_arg.cc(%arg0, %arg1, %arg2)
      : (i32, i64, memref<?xi64>) -> ()

    memref.dealloc %arg2 : memref<?xi64>

    func.return
  }
}

module @custom_call attributes { tfrt.compiled } {
  func.func private @custom_call.cc()
     attributes { rt.custom_call = "testlib.direct_call" }

  func.func @main() {
    func.call @custom_call.cc() : () -> ()
    func.return
  }
}

// CHECK: --- Running 'compiled_custom_call'
func.func @compiled_custom_call() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  // Allocate and initialize input tensor.
  %input = tfrt_dht.create_uninitialized_tensor.f32.2 [16 : i64, 16 : i64]
  %ch1 = tfrt_dht.fill_tensor_with_constant.f32 %input, %ch0 1.0 : f32

  // Allocate tensor for the expected values.
  %expected = tfrt_dht.create_uninitialized_tensor.f32.2 [16 : i64, 16 : i64]

  // Compile a kernel with a custom call.
  %executable = jitrt.compile { kernel = @multiply::@main }

  // Execute compiled kernel with tensor operands.
  %output0, %output1 = jitrt.execute %executable[%ch1](%input)
    : (!t.tensor) -> (!t.tensor, !t.tensor)

  // Wait for the execution completion and compare result with expected.
  %ch2 = tfrt_dht.fill_tensor_with_constant.f32 %expected, %ch1 2.0 : f32
  %cmp0, %cmp0_ch = "tfrt_dht.tensor_allclose.f32"(%expected, %output0, %ch2)
    : (!t.tensor, !t.tensor, !tfrt.chain) -> (i1, !tfrt.chain)

  %ch3 = tfrt_dht.fill_tensor_with_constant.f32 %expected, %cmp0_ch 3.0 : f32
  %cmp1, %cmp1_ch = "tfrt_dht.tensor_allclose.f32"(%expected, %output1, %ch3)
    : (!t.tensor, !t.tensor, !tfrt.chain) -> (i1, !tfrt.chain)

  // CHECK: int1 = 1
  // CHECK: int1 = 1
  %printed0 = tfrt.print.i1 %cmp0, %cmp1_ch
  %printed1 = tfrt.print.i1 %cmp0, %printed0

  tfrt.return %printed1 : !tfrt.chain
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

  // CHECK: returned <<error: run time error:
  // CHECK-SAME: custom call 'testlib.multiply' failed:
  // CHECK-SAME: Unsupported floating point dtype>>
  tfrt.return %output : !t.tensor
}

// CHECK: --- Running 'compiled_custom_call_error_args'
func.func @compiled_custom_call_error_args() -> !t.tensor {
  %ch0 = tfrt.new.chain

  // Allocate and initialize input tensor.
  %input = tfrt_dht.create_uninitialized_tensor.f32.2 [16 : i64, 4 : i64]
  %ch1 = tfrt_dht.fill_tensor_with_constant.f32 %input, %ch0 1.0 : f32

  // Compile a kernel with a custom call.
  %executable = jitrt.compile { kernel = @multiply_errors::@main }

  // Execute compiled kernel with tensor operands.
  %output = jitrt.execute %executable[%ch1](%input) : (!t.tensor) -> !t.tensor

  // CHECK: returned <<error: run time error:
  // CHECK-SAME: custom call 'testlib.multiply' failed:
  // CHECK-SAME: Wrong number of arguments: expected 2 got 3>>
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
  // CHECK: i64[2x2] ((1, 2), (3, 4))
  // CHECK: i32[4] 101, 102, 103, 104
  // CHECK: i64[4] 105, 106, 107, 108
  // CHECK: f32[4] 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00
  // CHECK: f64[4] 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00
  // CHECK: i64[3] 1, 2, 3
  // CHECK: i64[0]
  // CHECK: str: some string

  // Check that attributes not in the custom call signature are ignored.
  // CHECK-NOT: unused attributes to test
  %executable = jitrt.compile { kernel = @print_attrs::@main }
  jitrt.execute %executable[%ch0]() : () -> ()

  tfrt.return
}

// CHECK: --- Running 'compiled_custom_call_print_variant_attrs'
func.func @compiled_custom_call_print_variant_attrs() {
  %ch0 = tfrt.new.chain

  // CHECK: i32: 101
  // CHECK: f32: 1.000000e+00
  // CHECK: i32[4] 101, 102, 103, 104
  // CHECK: i64[4] 105, 106, 107, 108
  // CHECK: str: some string
  %executable = jitrt.compile { kernel = @print_variant_attrs::@main }
  jitrt.execute %executable[%ch0]() : () -> ()

  tfrt.return
}

// CHECK: --- Running 'compiled_custom_call_print_dialect_attrs'
func.func @compiled_custom_call_print_dialect_attrs() {
  %ch0 = tfrt.new.chain

  // CHECK: Enum: Baz
  // CHECK: Runtime Enum: RuntimeBar
  // CHECK: PairOfDims: rank = 2 a = [1, 1] b = [2, 2]
  %executable = jitrt.compile { kernel = @print_dialect_attrs::@main }
  jitrt.execute %executable[%ch0]() : () -> ()

  tfrt.return
}

// CHECK: --- Running 'compiled_custom_call_direct_print_attrs'
func.func @compiled_custom_call_direct_print_attrs() {
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
  // CHECK: i64[2x2] ((1, 2), (3, 4))
  // CHECK: i32[4] 101, 102, 103, 104
  // CHECK: i64[4] 105, 106, 107, 108
  // CHECK: f32[4] 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00
  // CHECK: f64[4] 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00
  // CHECK: i64[3] 1, 2, 3
  // CHECK: i64[0]
  // CHECK: str: some string

  %executable = jitrt.compile { kernel = @direct_print_attrs::@main }
  jitrt.execute %executable[%ch0]() : () -> ()

  tfrt.return
}

// CHECK: --- Running 'compiled_custom_call_variadic_args'
func.func @compiled_custom_call_variadic_args() {
  %ch0 = tfrt.new.chain

  // CHECK: Number of variadic arguments: 7
  // CHECK: arg[0]: i32: 123
  // CHECK: arg[1]: i64: 456
  // CHECK: arg[2]: f32: 1.000000e+00
  // CHECK: arg[3]: f64: 2.000000e+00

  // CHECK: arg[4]: StridedMemrefView: dtype: s64 sizes: [1] strides: [1]
  // CHECK-SAME:    MemrefView: dtype: s64 sizes: [1]
  // CHECK-SAME:    FlatMemrefView: dtype: s64 size_in_bytes: 8

  // CHECK: arg[5]: StridedMemrefView: dtype: f32 sizes: [2] strides: [1]
  // CHECK-SAME:    MemrefView: dtype: f32 sizes: [2]
  // CHECK-SAME:    FlatMemrefView: dtype: f32 size_in_bytes: 8

  // CHECK: arg[6]: StridedMemrefView: {{.*}} sizes: [16, 3] strides: [1, 16]
  // CHECK-SAME:    None / None

  // CHECK: arg: MemrefView: dtype: s64 sizes: [1]
  // CHECK: Number of variadic arguments: 5
  // CHECK: arg[0]: i32: 123
  // CHECK: arg[1]: i64: 456
  // CHECK: arg[2]: f32: 1.000000e+00
  // CHECK: arg[3]: f64: 2.000000e+00

  // CHECK: arg[4]: StridedMemrefView: dtype: f32 sizes: [2] strides: [1]
  // CHECK-SAME:    MemrefView: dtype: f32 sizes: [2]
  // CHECK-SAME:    FlatMemrefView: dtype: f32 size_in_bytes: 8
  %executable = jitrt.compile { kernel = @variadic_args::@main }
  jitrt.execute %executable[%ch0]() : () -> ()

  tfrt.return
}

// CHECK: --- Running 'compiled_custom_call_variant_arg'
func.func @compiled_custom_call_variant_arg() {
  %ch0 = tfrt.new.chain

  // CHECK: i32: 123
  // CHECK: i64: 456
  // CHECK: StridedMemrefView: dtype: s64 sizes: [1] strides: [1]
  // CHECK-SAME: MemrefView: dtype: s64 sizes: [1]
  // CHECK-SAME: FlatMemrefView: dtype: s64 size_in_bytes: 8
  %executable = jitrt.compile { kernel = @variant_arg::@main }
  jitrt.execute %executable[%ch0]() : () -> ()

  tfrt.return
}

// CHECK: --- Running 'compiled_direct_custom_call'
func.func @compiled_direct_custom_call() {
  %ch0 = tfrt.new.chain

  // CHECK: Direct custom call: num_args=0; num_attrs=0
  // CHECK-SAME: str=Called from: jitrt.execute
  %executable = jitrt.compile { kernel = @custom_call::@main }
  jitrt.execute %executable[%ch0]() : () -> ()

  tfrt.return
}
