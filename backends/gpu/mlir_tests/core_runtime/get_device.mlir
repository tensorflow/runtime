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

// CHECK-LABEL: --- Running 'get_cpu_device'
func @get_cpu_device() -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"

  %handle = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [1, 1], values = [2.0 : f32] } : 1

  // CHECK: shape = [1, 1], values = [2.000000e+00]
  %ch1 = "corert.print_tensorhandle"(%handle, %ch0) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch1 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'get_gpu_device'
func @get_gpu_device() -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %gpu = corert.get_op_handler %ch0 "gpu"

  %handle = corert.executeop(%gpu)
    "tfrt_test.create_dense_tensor"() { shape = [1, 1], values = [2.0 : f32] } : 1

  // CHECK: DenseGpuTensor<dtype=F32, shape=[1, 1], pointer=0x{{[0-9a-f]+}} (CUDA)>
  %ch1 = "corert.print_tensorhandle"(%handle, %ch0) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch1 : !tfrt.chain
}
