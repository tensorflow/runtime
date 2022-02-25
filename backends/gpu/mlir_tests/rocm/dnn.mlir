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

// RUN: bef_executor_lite %s.bef | FileCheck %s

// CHECK-LABEL: --- Running 'dnn_pooling_test'
func @dnn_pooling_test() {
  %ch2 = tfrt.new.chain
  %ordinal = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get ROCm, %ordinal
  %context = tfrt_gpu.context.create %device
  %allocator = tfrt_gpu.allocator.create %context
  %stream = tfrt_gpu.stream.create %context

  %k_in_size = tfrt.constant.i32 36  //  2 * 2 * 3 * 3
  %k_out_size = tfrt.constant.i32 4  //  2 * 2 * 1 * 1

  // Input tesnsor
  %i00 = tfrt.constant.f32 0.262123
  %i01 = tfrt.constant.f32 -0.448813
  %i02 = tfrt.constant.f32 0.073700
  %i03 = tfrt.constant.f32 -0.144819
  %i04 = tfrt.constant.f32 2.388026
  %i05 = tfrt.constant.f32 -0.374544
  %i06 = tfrt.constant.f32 0.110159
  %i07 = tfrt.constant.f32 -1.542872
  %i08 = tfrt.constant.f32 -0.614891
  %i09 = tfrt.constant.f32 -2.051789
  %i10 = tfrt.constant.f32 0.549311
  %i11 = tfrt.constant.f32 -0.514576
  %i12 = tfrt.constant.f32 -0.359810
  %i13 = tfrt.constant.f32 -0.658335
  %i14 = tfrt.constant.f32 -0.187685
  %i15 = tfrt.constant.f32 0.648840
  %i16 = tfrt.constant.f32 -0.516337
  %i17 = tfrt.constant.f32 -0.868445
  %i18 = tfrt.constant.f32 0.362668
  %i19 = tfrt.constant.f32 1.031871
  %i20 = tfrt.constant.f32 -0.771410
  %i21 = tfrt.constant.f32 0.062409
  %i22 = tfrt.constant.f32 -0.374612
  %i23 = tfrt.constant.f32 -0.486497
  %i24 = tfrt.constant.f32 0.432054
  %i25 = tfrt.constant.f32 2.402000
  %i26 = tfrt.constant.f32 -0.441910
  %i27 = tfrt.constant.f32 2.352234
  %i28 = tfrt.constant.f32 0.581970
  %i29 = tfrt.constant.f32 -0.111883
  %i30 = tfrt.constant.f32 -0.888563
  %i31 = tfrt.constant.f32 0.514422
  %i32 = tfrt.constant.f32 0.561516
  %i33 = tfrt.constant.f32 -0.330782
  %i34 = tfrt.constant.f32 0.647885
  %i35 = tfrt.constant.f32 0.257522
  %input = tfrt_dht.create_uninitialized_tensor.f32.4 [2 : i64, 2 : i64, 3 : i64, 3 : i64]
  %ch7 = "tfrt_dht.set_tensor_with_values.f32"(%input, %ch2, %i00, %i01, %i02, %i03, %i04, %i05, %i06, %i07, %i08, %i09, %i10, %i11, %i12, %i13, %i14, %i15, %i16, %i17, %i18, %i19, %i20, %i21, %i22, %i23, %i24, %i25, %i26, %i27, %i28, %i29, %i30, %i31, %i32, %i33, %i34, %i35): (!t.tensor, !tfrt.chain, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) -> !tfrt.chain
  %ch8 = tfrt_dht.print_tensor %input, %ch7

  // Expected output tensor
  %o0 = tfrt.constant.f32 0.549311
  %o1 = tfrt.constant.f32 0.073700
  %o2 = tfrt.constant.f32 2.388026
  %o3 = tfrt.constant.f32 2.402000
  %output = tfrt_dht.create_uninitialized_tensor.f32.4 [2 : i64, 2 : i64, 1 : i64, 1 : i64]
  %ch9 = "tfrt_dht.set_tensor_with_values.f32"(%output, %ch7, %o0, %o1, %o2, %o3): (!t.tensor, !tfrt.chain, f32, f32, f32, f32) -> !tfrt.chain
  %ch10 = tfrt_dht.print_tensor %output, %ch9

  // Expected gradient tensor for the backpool
  %o00 = tfrt.constant.f32 0.000000
  %o01 = tfrt.constant.f32 0.000000
  %o02 = tfrt.constant.f32 0.073700
  %o03 = tfrt.constant.f32 0.000000
  %o04 = tfrt.constant.f32 2.388030
  %o05 = tfrt.constant.f32 0.000000
  %o06 = tfrt.constant.f32 0.000000
  %o07 = tfrt.constant.f32 0.000000
  %o08 = tfrt.constant.f32 0.000000
  %o09 = tfrt.constant.f32 0.000000
  %o10 = tfrt.constant.f32 0.549311
  %o11 = tfrt.constant.f32 0.000000
  %o12 = tfrt.constant.f32 0.000000
  %o13 = tfrt.constant.f32 0.000000
  %o14 = tfrt.constant.f32 0.000000
  %o15 = tfrt.constant.f32 0.000000
  %o16 = tfrt.constant.f32 0.000000
  %o17 = tfrt.constant.f32 0.000000
  %o18 = tfrt.constant.f32 0.000000
  %o19 = tfrt.constant.f32 0.000000
  %o20 = tfrt.constant.f32 0.000000
  %o21 = tfrt.constant.f32 0.000000
  %o22 = tfrt.constant.f32 0.000000
  %o23 = tfrt.constant.f32 0.000000
  %o24 = tfrt.constant.f32 0.000000
  %o25 = tfrt.constant.f32 21.61800
  %o26 = tfrt.constant.f32 0.000000
  %o27 = tfrt.constant.f32 14.11340
  %o28 = tfrt.constant.f32 0.000000
  %o29 = tfrt.constant.f32 0.000000
  %o30 = tfrt.constant.f32 0.000000
  %o31 = tfrt.constant.f32 0.000000
  %o32 = tfrt.constant.f32 3.369100
  %o33 = tfrt.constant.f32 0.000000
  %o34 = tfrt.constant.f32 3.239430
  %o35 = tfrt.constant.f32 0.257522
  %gradient = tfrt_dht.create_uninitialized_tensor.f32.4 [2 : i64, 2 : i64, 3 : i64, 3 : i64]
  %ch11 = "tfrt_dht.set_tensor_with_values.f32"(%gradient, %ch10, %o00, %o01, %o02, %o03, %o04, %o05, %o06, %o07, %o08, %o09, %o10, %o11, %o12, %o13, %o14, %o15, %o16, %o17, %o18, %o19, %o20, %o21, %o22, %o23, %o24, %o25, %o26, %o27, %o28, %o29, %o30, %o31, %o32, %o33, %o34, %o35):(!t.tensor, !tfrt.chain, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) -> !tfrt.chain
  %ch12 = tfrt_dht.print_tensor %gradient, %ch11

  %dnn = tfrt_gpu.dnn.create %context

  %mode = tfrt.constant.ui32 0
  %nan_propagation = tfrt.constant.ui32 0
  %pooling_desc = tfrt_gpu.dnn.create_pooling_descriptor %context, %mode,
    %nan_propagation, [3 : i32, 3 : i32], [0 : i32, 0 : i32], [1 : i32, 1 : i32]

  %in_desc = tfrt_gpu.dnn.create_tensor_descriptor miopenFloat,
    [2 : i32, 2 : i32, 10 : i32, 10 : i32],
    [200 : i32, 100 : i32, 10 : i32, 1 : i32]

  %out_desc = tfrt_gpu.dnn.create_tensor_descriptor miopenFloat,
    [2 : i32, 2 : i32, 8 : i32, 8 : i32],
    [128 : i32, 64 : i32, 8 : i32, 1 : i32]

  %alpha = tfrt.constant.f32 1.0
  %beta = tfrt.constant.f32 0.0

  %kinsize = tfrt.constant.i64 144  //  (2 * 2 * 3 * 3) * (size of f32 = 4);
  %koutsize = tfrt.constant.i64 16  //  (2 * 2 * 1 * 1) * (size of f32 = 4);

  %input_device_buffer = tfrt_gpu.mem.allocate %allocator, %stream, %kinsize, %ch12
  %output_device_buffer = tfrt_gpu.mem.allocate %allocator, %stream, %koutsize, %ch12

  %host_input_buffer, %ch28 = tfrt_dht.get_buffer %input, %ch12
  %ch29 = tfrt_gpu.mem.copy %input_device_buffer, %host_input_buffer, %stream, %ch28 : !tfrt_gpu.buffer, !ht.host_buffer

  %output_tensor = tfrt_dht.create_uninitialized_tensor.f32.4 [2 : i64, 2 : i64, 1 : i64, 1 : i64]
  %ch29_1 = tfrt_dht.fill_tensor_with_constant.f32 %output_tensor, %ch29 0.0 : f32
  // CHECK: shape = [2, 2, 1, 1], values = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]
  %ch30 = tfrt_dht.print_tensor %output_tensor, %ch29_1

  %output_host_buffer, %ch31 = tfrt_dht.get_buffer %output_tensor, %ch30
  %ch32 = tfrt_gpu.mem.copy %output_device_buffer, %output_host_buffer, %stream, %ch31 : !tfrt_gpu.buffer, !ht.host_buffer

  %ch33 = tfrt_gpu.dnn.pooling_forward %dnn, %stream, %pooling_desc, %alpha, %in_desc, %input_device_buffer, %beta, %out_desc, %output_device_buffer, %ch32

  %ch34a = tfrt_gpu.mem.copy %output_host_buffer, %output_device_buffer, %stream, %ch33 : !ht.host_buffer, !tfrt_gpu.buffer
  %ch34b = tfrt_gpu.stream.synchronize %stream, %ch34a

  // Need to make sure that for pooling forward
  // expected result matches computed result:
  // CHECK: shape = [2, 2, 1, 1], values = [5.493110e-01, 7.370000e-02, 2.388026e+00, 2.402000e+00]
  %ch35 = tfrt_dht.print_tensor %output, %ch34b
  // CHECK: shape = [2, 2, 1, 1], values = [5.493110e-01, 7.370000e-02, 2.388026e+00, 2.402000e+00]
  %ch36 = tfrt_dht.print_tensor %output_tensor, %ch35

  %in_grad_device_buffer = tfrt_gpu.mem.allocate %allocator, %stream, %kinsize, %ch36
  %in_grad_tensor = tfrt_dht.create_uninitialized_tensor.f32.4 [2 : i64, 2 : i64, 3 : i64, 3 : i64]
  %ch38 = tfrt_dht.fill_tensor_with_constant.f32 %in_grad_tensor, %ch36 0.0 : f32
  %ch39 = tfrt_dht.print_tensor %in_grad_tensor, %ch38
  %in_grad_host_buffer, %ch40 = tfrt_dht.get_buffer %in_grad_tensor, %ch39
  %ch41 = tfrt_gpu.mem.copy %in_grad_device_buffer, %in_grad_host_buffer, %stream, %ch40 : !tfrt_gpu.buffer, !ht.host_buffer

  %ch42 = tfrt_gpu.dnn.pooling_backward %dnn, %stream, %pooling_desc, %alpha, %out_desc, %output_device_buffer, %out_desc, %output_device_buffer, %in_desc, %input_device_buffer, %beta, %in_desc, %in_grad_device_buffer, %ch41

  %ch43 = tfrt_gpu.mem.copy %in_grad_host_buffer, %in_grad_device_buffer, %stream, %ch42 : !ht.host_buffer, !tfrt_gpu.buffer

  // Need to make sure that for pooling backward
  // expected result matches computed result:
  %ch44 = tfrt_dht.print_tensor %gradient, %ch43
  %ch45 = tfrt_dht.print_tensor %in_grad_tensor, %ch44
  // TODO(gkg): How to compare with some epsilon two F32 tensors?
  // Manual inspection shows that they match.

  tfrt.return
}
