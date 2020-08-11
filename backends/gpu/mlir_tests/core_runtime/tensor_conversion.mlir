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

// CHECK: --- Running 'round_trip_transfer'
func @round_trip_transfer() -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %gpu_handler = corert.get_op_handler %ch0 "gpu"
  %cpu_handler = corert.get_op_handler %ch0 "cpu"

  %th0_cpu = corert.executeop(%cpu_handler) "tf.Const"()
      { dtype = f32, value = dense<1.0> : tensor<1x1x2x2xf32> } : 1

  // Formats is a bitmask of Tensor::Subclass. 16 is DenseGpu.
  // DHT->DGT
  %th0_gpu = "corert.transfer"(%th0_cpu) {device="GPU:0", formats=16}
    : (!corert.tensorhandle) -> !corert.tensorhandle

  // CHECK: DenseGpuTensor<dtype=F32, shape=[1, 1, 2, 2], pointer={{0x[[:xdigit:]]*}} (CUDA)>
  %ch1 = "corert.print_tensorhandle"(%th0_gpu, %ch0)
    : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  // formats 1 is DenseCpu. DGT->DHT
  %th1_cpu = "corert.transfer"(%th0_cpu) {device="CPU:0", formats=1}
    : (!corert.tensorhandle) -> !corert.tensorhandle

  // CHECK: DenseHostTensor dtype = F32, shape = [1, 1, 2, 2], values = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
  %ch2 = "corert.print_tensorhandle"(%th1_cpu, %ch1)
    : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch2 : !tfrt.chain
}

// CHECK: --- Running 'invalid_transfer'
func @invalid_transfer() -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %gpu_handler = corert.get_op_handler %ch0 "gpu"
  %cpu_handler = corert.get_op_handler %ch0 "cpu"

  %th0_gpu = corert.executeop(%gpu_handler) "tf.Const"()
      { dtype = f32, value = dense<1.0> : tensor<1x1x2x2xf32> } : 1

  // Formats 2 is ScalarHost.
  // DGT->SHT is not supported yet
  // expected-error @+1 {{runtime error: does not support converting DenseGpuTensor to allowed_formats: 0x2}}
  %th0_cpu = "corert.transfer"(%th0_gpu) {device="CPU:0", formats=2}
    : (!corert.tensorhandle) -> !corert.tensorhandle

  // CHECK: Error TensorHandle: 'does not support converting DenseGpuTensor to allowed_formats: 0x2'
  %ch1 = "corert.print_tensorhandle"(%th0_cpu, %ch0)
    : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch1 : !tfrt.chain
}
