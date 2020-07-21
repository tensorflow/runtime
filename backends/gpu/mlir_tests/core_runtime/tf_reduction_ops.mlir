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

// RUN: bef_executor -devices=gpu $(bef_name %s) | FileCheck %s --dump-input=always

// CHECK: --- Running 'mean_full_reduce_f32'
func @mean_full_reduce_f32() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %gpu_handle_input = corert.executeop(%gpu) "tfrt_test.create_dense_tensor"()
  { shape = [1, 5], values = [1.0 : f32, 2.0 : f32, 3.0 : f32, 4.0 : f32, 5.0 : f32] } : 1

  %gpu_handle_result = corert.executeop(%gpu) "_tf.Mean"(%gpu_handle_input)
  { reduction_indices = dense<[0, 1]> : tensor<2xi32> } : 1

  %cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%gpu_handle_result) : 1

  // CHECK: DenseHostTensor dtype = F32, shape = [], values = [3]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'mean_outer_reduce_f32'
func @mean_outer_reduce_f32() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %ch_cuda_init = cuda.init %ch_epoch
  %gpu = corert.get_op_handler %ch_cuda_init "gpu"

  %gpu_handle_input = corert.executeop(%gpu) "tfrt_test.create_dense_tensor"()
  { shape = [2, 2, 3, 2],
    values = [
      11.0 : f32, 12.0 : f32, 13.0 : f32, 14.0 : f32, 15.0 : f32, 16.0 : f32,
      21.0 : f32, 22.0 : f32, 23.0 : f32, 24.0 : f32, 25.0 : f32, 26.0 : f32,
      31.0 : f32, 32.0 : f32, 33.0 : f32, 34.0 : f32, 35.0 : f32, 36.0 : f32,
      41.0 : f32, 42.0 : f32, 43.0 : f32, 44.0 : f32, 45.0 : f32, 46.0 : f32
    ]
  } : 1

  %gpu_handle_result = corert.executeop(%gpu) "_tf.Mean"(%gpu_handle_input)
  { reduction_indices = dense<[0, 1]> : tensor<2xi32> } : 1

  %cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%gpu_handle_result) : 1

  // CHECK: DenseHostTensor dtype = F32, shape = [3, 2], values = [26, 27, 28, 29, 30, 31]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}

// CHECK: --- Running 'mean_inner_reduce_f32'
func @mean_inner_reduce_f32() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %ch_cuda_init = cuda.init %ch_epoch
  %gpu = corert.get_op_handler %ch_cuda_init "gpu"

  %gpu_handle_input = corert.executeop(%gpu) "tfrt_test.create_dense_tensor"()
  { shape = [2, 2, 3, 2],
    values = [
      11.0 : f32, 12.0 : f32, 13.0 : f32, 14.0 : f32, 15.0 : f32, 16.0 : f32,
      21.0 : f32, 22.0 : f32, 23.0 : f32, 24.0 : f32, 25.0 : f32, 26.0 : f32,
      31.0 : f32, 32.0 : f32, 33.0 : f32, 34.0 : f32, 35.0 : f32, 36.0 : f32,
      41.0 : f32, 42.0 : f32, 43.0 : f32, 44.0 : f32, 45.0 : f32, 46.0 : f32
    ]
  } : 1

  %gpu_handle_result = corert.executeop(%gpu) "_tf.Mean"(%gpu_handle_input)
  { reduction_indices = dense<[2, 3]> : tensor<2xi32> } : 1

  %cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%gpu_handle_result) : 1

  // CHECK: DenseHostTensor dtype = F32, shape = [2, 2], values = [13.5, 23.5, 33.5, 43.5]
  %ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%cpu_handle_result) : 0

  tfrt.return %ch_print_cpu : !tfrt.chain
}
