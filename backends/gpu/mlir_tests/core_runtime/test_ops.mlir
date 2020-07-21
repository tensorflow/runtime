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

// RUN: bef_executor -devices=cpu,gpu $(bef_name %s) | FileCheck %s --dump-input=fail

// CHECK: --- Running 'basic_test_create_dense_gpu_tensor'
func @basic_test_create_dense_gpu_tensor() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %gpu_handle = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [1, 5], values = [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32, 5.0 : f32] } : 1

  // CHECK: DenseGpuTensor<dtype=F32, shape=[1, 5], pointer=0x{{[0-9a-f]+}} (CUDA)>
  %ch_print_gpu = "corert.print_tensorhandle"(%gpu_handle, %ch_epoch) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  %cpu_handle = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%gpu_handle) : 1
  // CHECK: DenseHostTensor dtype = F32, shape = [1, 5], values = [1, 2, 3, 4, 5]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%cpu_handle) : 0
  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'basic_add_dense_gpu_tensors_f32'
func @basic_add_dense_gpu_tensors_f32() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %gpu_handle_lhs = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [1, 5], values = [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32, 5.0 : f32] } : 1

  %gpu_handle_rhs = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [1, 5], values = [10.0 : f32, 20.0 : f32, 30.0 : f32, 40.0 : f32, 50.0 : f32] } : 1

  %gpu_handle_result = corert.executeop(%gpu) "tfrt_test.add"(%gpu_handle_lhs, %gpu_handle_rhs) : 1

  %cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%gpu_handle_result) : 1
  // CHECK: DenseHostTensor dtype = F32, shape = [1, 5], values = [11, 22, 33, 44, 55]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0
  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'basic_add_dense_gpu_tensors_int32'
func @basic_add_dense_gpu_tensors_int32() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %gpu_handle_lhs = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [1, 5], values = [1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32] } : 1

  %gpu_handle_rhs = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [1, 5], values = [10 : i32, 20 : i32, 30 : i32, 40 : i32, 50 : i32] } : 1

  %gpu_handle_result = corert.executeop(%gpu) "tfrt_test.add"(%gpu_handle_lhs, %gpu_handle_rhs) : 1

  %cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%gpu_handle_result) : 1
  // CHECK: DenseHostTensor dtype = I32, shape = [1, 5], values = [11, 22, 33, 44, 55]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0
  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'basic_move_gpu_tensors_int32'
func @basic_move_gpu_tensors_int32() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_epoch "cpu"
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %gpu_handle_lhs = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [1, 5], values = [1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32] } : 1

  %cpu_handle_lhs = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%gpu_handle_lhs) : 1

  %gpu_handle_rhs = corert.executeop(%gpu) "tfrt_test.dht_to_gpu_tensor"(%cpu_handle_lhs) : 1

  %gpu_handle_result = corert.executeop(%gpu) "tfrt_test.add"(%gpu_handle_lhs, %gpu_handle_rhs) : 1

  %cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%gpu_handle_result) : 1
  // CHECK: DenseHostTensor dtype = I32, shape = [1, 5], values = [2, 4, 6, 8, 10]
  %ch_print_cpu = corert.executeop.seq(%cpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0
  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'basic_add_dense_gpu_tensors_f16'
func @basic_add_dense_gpu_tensors_f16() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_epoch "cpu"
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %cpu_handle_f32 = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [1, 5], values = [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32, 5.0 : f32] } : 1

  %cpu_handle_f16 = corert.executeop(%cpu)
    "tfrt_test.cast"(%cpu_handle_f32) { type = "f16" } : 1

  %gpu_handle_f16 = corert.executeop(%gpu) "tfrt_test.dht_to_gpu_tensor"(%cpu_handle_f16) : 1

  %gpu_handle_result_f16 = corert.executeop(%gpu) "tf.AddV2"(%gpu_handle_f16, %gpu_handle_f16) : 1

  %cpu_handle_result_f16 = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%gpu_handle_result_f16) : 1

  %cpu_handle_result_f32 = corert.executeop(%cpu)
    "tfrt_test.cast"(%cpu_handle_result_f16) { type = "f32" } : 1

  // CHECK: DenseHostTensor dtype = F32, shape = [1, 5], values = [2.000000e+00, 4.000000e+00, 6.000000e+00, 8.000000e+00, 1.000000e+01]
  %ch_print_cpu = corert.executeop.seq(%cpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result_f32) : 0
  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'return_multiple_results'
func @return_multiple_results() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %ch_cuda_init = cuda.init %ch_epoch
  %gpu = corert.get_op_handler %ch_cuda_init "gpu"

  %gpu_handles:2 = corert.executeop(%gpu) "tfrt_test.return_multiple_results"() : 2

  tfrt.return %ch_cuda_init : !tfrt.chain
}

// CHECK: --- Running 'return_multiple_results_with_error'
func @return_multiple_results_with_error() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %ch_cuda_init = cuda.init %ch_epoch
  %gpu = corert.get_op_handler %ch_cuda_init "gpu"

  // expected-error @+1 {{runtime error: error from ReturnMultipleResultsWithError op}}
  %gpu_handles:2 = corert.executeop(%gpu) "tfrt_test.return_multiple_results_with_error"() : 2

  tfrt.return %ch_cuda_init : !tfrt.chain
}

// CHECK: --- Running 'test_optional_args'
func @test_optional_args() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %ch_cuda_init = cuda.init %ch_epoch
  %gpu = corert.get_op_handler %ch_cuda_init "gpu"

  %gpu_handle1 = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [1, 1], values = [1.0 : f32] } : 1

  %gpu_handle2 = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [1, 1], values = [2.0 : f32] } : 1

  %gpu_handle3 = corert.executeop(%gpu) "tfrt_test.test_optional_arg"(%gpu_handle1) : 1
  %cpu_handle3 = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%gpu_handle3) : 1
  // CHECK: DenseHostTensor dtype = F32, shape = [1, 1], values = [1]
  %ch_print_cpu1 = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%cpu_handle3) : 0

  %gpu_handle4 = corert.executeop(%gpu) "tfrt_test.test_optional_arg"(%gpu_handle1, %gpu_handle2) : 1
  %cpu_handle4 = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%gpu_handle4) : 1
  // CHECK: DenseHostTensor dtype = F32, shape = [1, 1], values = [2]
  %ch_print_cpu2 = corert.executeop.seq(%gpu, %ch_print_cpu1) "tfrt_test.print"(%cpu_handle4) : 0

  tfrt.return %ch_print_cpu2 : !tfrt.chain
}

// CHECK: --- Running 'test_variadic_args'
func @test_variadic_args() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %ch_cuda_init = cuda.init %ch_epoch
  %gpu = corert.get_op_handler %ch_cuda_init "gpu"

  %gpu_handle1 = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [1, 1], values = [1.0 : f32] } : 1

  %gpu_handle2 = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [1, 1], values = [2.0 : f32] } : 1

  %gpu_handle3 = corert.executeop(%gpu) "tfrt_test.test_variadic_arg"(%gpu_handle1) : 1
  %cpu_handle3 = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%gpu_handle3) : 1
  // CHECK: DenseHostTensor dtype = F32, shape = [1, 1], values = [1]
  %ch_print_cpu1 = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%cpu_handle3) : 0

  %gpu_handle4 = corert.executeop(%gpu) "tfrt_test.test_variadic_arg"(%gpu_handle1, %gpu_handle2) : 1
  %cpu_handle4 = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%gpu_handle4) : 1
  // CHECK: DenseHostTensor dtype = F32, shape = [1, 1], values = [2]
  %ch_print_cpu2 = corert.executeop.seq(%gpu, %ch_print_cpu1) "tfrt_test.print"(%cpu_handle4) : 0

  tfrt.return %ch_print_cpu2 : !tfrt.chain
}
