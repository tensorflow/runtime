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

// RUN: bef_executor -devices=gpu $(bef_name %s) | FileCheck %s --dump-input=fail

// Test case for b/148703930.
//
// This test case runs the following graph:
//
//            Epoch
//              |
//     +--------+--------+
//     |                 |
//     v                 v
//    H2D               H2D
//     |                 |
//     v                 v
//   Tanh              Tanh
//     |                 |
//     v                 v
//    D2H               D2H
//     |                 |
//     +--------+--------+
//              v
//          tfrt.return
//
// The left H2D copy blocks and theh left tf.Tanh calls Host::RunWhenReady (with
// closure "C") to enqueue work that would run when the left H2D finishes.  The
// GPU corert kernel for the right H2D copy is then executed while the closure
// "C" is still alive.
//
// This used to be a problem when "C" contained a GpuDispatchContext since
// GpuDispatchContext contains a gpu::stream::CurrentContext.  The right H2D
// copy would then try to CtxSetCurrent while the gpu::stream::CurrentContext
// from the left H2D copy was still alive.

// CHECK: --- Running 'current_context_lifetime'
func @current_context_lifetime() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %gpu = corert.get_op_handler %ch_epoch "gpu"

  %test0_operand = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [5], values = [-1.0 : f32, -0.5 : f32, 0.0 : f32, 0.5 : f32, 1.0 : f32] } : 1
  %test0_gpu_handle_result = corert.executeop(%gpu) "tf.Tanh"(%test0_operand) : 1
  %test0_cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%test0_gpu_handle_result) : 1
  %test0_ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%test0_cpu_handle_result) : 0
  // CHECK: DenseHostTensor dtype = F32, shape = [5], values = [-0.761594176, -0.462117136, 0, 0.462117136, 0.761594176]

  %test1_operand = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [5], values = [-1.0 : f32, -0.5 : f32, 0.0 : f32, 0.5 : f32, 1.0 : f32] } : 1
  %test1_gpu_handle_result = corert.executeop(%gpu) "tf.Tanh"(%test1_operand) : 1
  %test1_cpu_handle_result = corert.executeop(%gpu) "tfrt_test.gpu_tensor_to_host_tensor"(%test1_gpu_handle_result) : 1
  %test1_ch_print_cpu = corert.executeop.seq(%gpu, %ch_epoch) "tfrt_test.print"(%test0_cpu_handle_result) : 0
  // CHECK: DenseHostTensor dtype = F32, shape = [5], values = [-0.761594176, -0.462117136, 0, 0.462117136, 0.761594176]

  tfrt.return %test1_ch_print_cpu : !tfrt.chain
}
