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

// RUN: bef_executor $(bef_name %s)
//
// RUN: bef_executor $(bef_name %s) | FileCheck %s --dump-input=fail
// RUN: tfrt_gpu_opt %s | tfrt_gpu_opt

// CHECK-LABEL: --- Running 'dnn_pooling_test'
func @dnn_pooling_test() {
  %ch2 = tfrt.new.chain
  %index = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get CUDA, %index
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

  %dnn = tfrt_gpu.dnn.create %stream

  %dim0 = tfrt.constant.i32 3
  %dim1 = tfrt.constant.i32 3
  %window_dimensions = tfrt_dht.create_uninitialized_tensor.i32.1 [2 : i64]
  %ch15 = "tfrt_dht.set_tensor_with_values.i32"(%window_dimensions, %ch12, %dim0, %dim1):(!t.tensor, !tfrt.chain, i32, i32) -> !tfrt.chain
  %p0 = tfrt.constant.i32 0
  %p1 = tfrt.constant.i32 0
  %paddings = tfrt_dht.create_uninitialized_tensor.i32.1 [2 : i64]
  %ch16 = "tfrt_dht.set_tensor_with_values.i32"(%paddings, %ch15, %p0, %p1):(!t.tensor, !tfrt.chain, i32, i32) -> !tfrt.chain
  %s0 = tfrt.constant.i32 1
  %s1 = tfrt.constant.i32 1
  %strides = tfrt_dht.create_uninitialized_tensor.i32.1 [2 : i64]
  %ch17 = "tfrt_dht.set_tensor_with_values.i32"(%strides, %ch16, %s0, %s1):(!t.tensor, !tfrt.chain, i32, i32) -> !tfrt.chain
  %mode = tfrt.constant.ui32 0
  %nan_propagation = tfrt.constant.ui32 0
  %pooling_desc = tfrt_gpu.dnn.create_pooling_descriptor %context, %mode, %nan_propagation, %window_dimensions, %paddings, %strides, %ch17

  %din0 = tfrt.constant.i32 2
  %din1 = tfrt.constant.i32 2
  %din2 = tfrt.constant.i32 10
  %din3 = tfrt.constant.i32 10
  %dim_in = tfrt_dht.create_uninitialized_tensor.i32.1 [4 : i64]
  %ch19 = "tfrt_dht.set_tensor_with_values.i32"(%dim_in, %ch17, %din0, %din1, %din2, %din3):(!t.tensor, !tfrt.chain, i32, i32, i32, i32) -> !tfrt.chain
  %sin0 = tfrt.constant.i32 200
  %sin1 = tfrt.constant.i32 100
  %sin2 = tfrt.constant.i32 10
  %sin3 = tfrt.constant.i32 1
  %stride_in = tfrt_dht.create_uninitialized_tensor.i32.1 [4 : i64]
  %ch20 = "tfrt_dht.set_tensor_with_values.i32"(%stride_in, %ch19, %sin0, %sin1, %sin2, %sin3):(!t.tensor, !tfrt.chain, i32, i32, i32, i32) -> !tfrt.chain
  %in_desc = tfrt_gpu.dnn.create_tensor_descriptor CUDNN_DATA_FLOAT, %dim_in, %stride_in, %ch20

  %dout0 = tfrt.constant.i32 2
  %dout1 = tfrt.constant.i32 2
  %dout2 = tfrt.constant.i32 8
  %dout3 = tfrt.constant.i32 8
  %dim_out = tfrt_dht.create_uninitialized_tensor.i32.1 [4 : i64]
  %ch23 = "tfrt_dht.set_tensor_with_values.i32"(%dim_out, %ch20, %dout0, %dout1, %dout2, %dout3):(!t.tensor, !tfrt.chain, i32, i32, i32, i32) -> !tfrt.chain
  %sout0 = tfrt.constant.i32 128
  %sout1 = tfrt.constant.i32 64
  %sout2 = tfrt.constant.i32 8
  %sout3 = tfrt.constant.i32 1
  %stride_out = tfrt_dht.create_uninitialized_tensor.i32.1 [4 : i64]
  %ch24 = "tfrt_dht.set_tensor_with_values.i32"(%stride_out, %ch23, %sout0, %sout1, %sout2, %sout3):(!t.tensor, !tfrt.chain, i32, i32, i32, i32) -> !tfrt.chain
  %out_desc = tfrt_gpu.dnn.create_tensor_descriptor CUDNN_DATA_FLOAT, %dim_out, %stride_out, %ch24

  %alpha = tfrt.constant.f32 1.0
  %beta = tfrt.constant.f32 0.0

  %kinsize = tfrt.constant.i64 144  //  (2 * 2 * 3 * 3) * (size of f32 = 4);
  %koutsize = tfrt.constant.i64 16  //  (2 * 2 * 1 * 1) * (size of f32 = 4);

  %input_device_buffer = tfrt_gpu.mem.allocate %allocator, %stream, %kinsize, %ch24
  %output_device_buffer = tfrt_gpu.mem.allocate %allocator, %stream, %koutsize, %ch24

  %host_input_buffer, %ch28 = tfrt_dht.get_buffer %input, %ch24
  %ch29 = tfrt_gpu.mem.copy_host_to_device %input_device_buffer, %host_input_buffer, %kinsize, %stream, %ch28

  %output_tensor = tfrt_dht.create_uninitialized_tensor.f32.4 [2 : i64, 2 : i64, 1 : i64, 1 : i64]
  %ch29_1 = tfrt_dht.fill_tensor_with_constant.f32 %output_tensor, %ch29 0.0 : f32
  // CHECK: shape = [2, 2, 1, 1], values = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]
  %ch30 = tfrt_dht.print_tensor %output_tensor, %ch29_1

  %output_host_buffer, %ch31 = tfrt_dht.get_buffer %output_tensor, %ch30
  %ch32 = tfrt_gpu.mem.copy_host_to_device %output_device_buffer, %output_host_buffer, %koutsize, %stream, %ch31

  %ch33 = tfrt_gpu.dnn.pooling_forward %dnn, %pooling_desc, %alpha, %in_desc, %input_device_buffer, %beta, %out_desc, %output_device_buffer, %ch32

  %ch34 = tfrt_gpu.mem.copy_device_to_host %output_host_buffer, %output_device_buffer, %koutsize, %stream, %ch33

  // Need to make sure that for pooling forward
  // expected result matches computed result:
  // CHECK: shape = [2, 2, 1, 1], values = [5.493110e-01, 7.370000e-02, 2.388026e+00, 2.402000e+00]
  %ch35 = tfrt_dht.print_tensor %output, %ch34
  // CHECK: shape = [2, 2, 1, 1], values = [5.493110e-01, 7.370000e-02, 2.388026e+00, 2.402000e+00]
  %ch36 = tfrt_dht.print_tensor %output_tensor, %ch35

  %in_grad_device_buffer = tfrt_gpu.mem.allocate %allocator, %stream, %kinsize, %ch36
  %in_grad_tensor = tfrt_dht.create_uninitialized_tensor.f32.4 [2 : i64, 2 : i64, 3 : i64, 3 : i64]
  %ch38 = tfrt_dht.fill_tensor_with_constant.f32 %in_grad_tensor, %ch36 0.0 : f32
  %ch39 = tfrt_dht.print_tensor %in_grad_tensor, %ch38
  %in_grad_host_buffer, %ch40 = tfrt_dht.get_buffer %in_grad_tensor, %ch39
  %ch41 = tfrt_gpu.mem.copy_host_to_device %in_grad_device_buffer, %in_grad_host_buffer, %kinsize, %stream, %ch40

  %ch42 = tfrt_gpu.dnn.pooling_backward %dnn, %pooling_desc, %alpha, %out_desc, %output_device_buffer, %out_desc, %output_device_buffer, %in_desc, %input_device_buffer, %beta, %in_desc, %in_grad_device_buffer, %ch41

  %ch43 = tfrt_gpu.mem.copy_device_to_host %in_grad_host_buffer, %in_grad_device_buffer, %kinsize, %stream, %ch42

  // Need to make sure that for pooling backward
  // expected result matches computed result:
  %ch44 = tfrt_dht.print_tensor %gradient, %ch43
  %ch45 = tfrt_dht.print_tensor %in_grad_tensor, %ch44
  // TODO(gkg): How to compare with some epsilon two F32 tensors?
  // Manual inspection shows that they match.

  tfrt.return
}
