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

// RUN: bef_executor --test_init_function=register_op_handlers_cpu_gpu %s.bef | FileCheck %s

func @register_op_handlers_cpu_gpu() {
  %null = "corert.create_null_op_handler"() : () -> !corert.ophandler

  %cpu = "corert.create_cpu_op_handler"(%null) : (!corert.ophandler) -> !corert.ophandler
  corert.register_op_handler %cpu "cpu"

  %gpu_ordinal = tfrt.constant.i32 0
  %gpu = "corert.create_gpu_op_handler" (%gpu_ordinal, %null) : (i32, !corert.ophandler) -> !corert.ophandler
  corert.register_op_handler %gpu "gpu"
  tfrt.return
}

// CHECK: --- Running 'round_trip_transfer'
func @round_trip_transfer() -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %gpu_handler = corert.get_op_handler %ch0 "gpu"
  %cpu_handler = corert.get_op_handler %ch0 "cpu"
  %gpu_device = "tfrt.get_device"(%ch0) {device_name="GPU:0"} : (!tfrt.chain) -> !tfrt.device
  %cpu_device = "tfrt.get_device"(%ch0) {device_name="CPU:0"} : (!tfrt.chain) -> !tfrt.device

  %th0_cpu = corert.executeop(%cpu_handler) "tf.Const"()
      { dtype = f32, value = dense<1.0> : tensor<1x1x2x2xf32> } : 1

  // DHT->DGT
  %th0_tensor_type = corert.get_dst_tensor_type %th0_cpu, %gpu_device
  %th0_gpu = corert.transfer %th0_cpu, %gpu_device, %th0_tensor_type

  // CHECK: DenseGpuTensor<dtype=f32, shape=[1, 1, 2, 2], pointer={{0x[[:xdigit:]]*}} (CUDA)>
  %ch1 = "corert.print_tensorhandle"(%th0_gpu, %ch0)
    : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  // DGT->DHT
  %th1_tensor_type = corert.get_dst_tensor_type %th0_cpu, %cpu_device
  %th1_cpu = corert.transfer %th0_cpu, %cpu_device, %th1_tensor_type

  // CHECK: DenseHostTensor dtype = f32, shape = [1, 1, 2, 2], values = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
  %ch2 = "corert.print_tensorhandle"(%th1_cpu, %ch1)
    : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch2 : !tfrt.chain
}

// CHECK: --- Running 'invalid_transfer'
func @invalid_transfer() -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %gpu_handler = corert.get_op_handler %ch0 "gpu"
  %cpu_handler = corert.get_op_handler %ch0 "cpu"
  %cpu_device = "tfrt.get_device"(%ch0) {device_name="CPU:0"} : (!tfrt.chain) -> !tfrt.device

  %th0_gpu = corert.executeop(%gpu_handler) "tf.Const"()
      { dtype = f32, value = dense<1.0> : tensor<1x1x2x2xf32> } : 1

  %th0_tensor_type = "tfrt_test.get_static_tensor_type"()
    { tensor_type = "StringHost" } : () -> !tfrt.tensor_type

  // DGT->SHT is not supported yet
  // expected-error @+1 {{runtime error: cannot find conversion function}}
  %th0_cpu = corert.transfer %th0_gpu, %cpu_device, %th0_tensor_type

  // CHECK: Error TensorHandle: 'cannot find conversion function for [DenseGpu]->[StringHost]'
  %ch1 = "corert.print_tensorhandle"(%th0_cpu, %ch0)
    : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch1 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'test_get_dst_tensor_type'
func @test_get_dst_tensor_type() -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"
  %cpu_handle = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    {shape = [1, 2], values = [0 : i64, 0 : i64] } : 1

  %cpu_device = "tfrt.get_device"(%ch0) {device_name="CPU:0"} : (!tfrt.chain) -> !tfrt.device
  %gpu_device = "tfrt.get_device"(%ch0) {device_name="GPU:0"} : (!tfrt.chain) -> !tfrt.device

  %result_1 = corert.get_dst_tensor_type %cpu_handle, %cpu_device
  // CHECK-NEXT: tensor_type = DenseHost
  %ch1 = "tfrt_test.print_tensor_type"(%result_1, %ch0) : (!tfrt.tensor_type, !tfrt.chain) -> (!tfrt.chain)

  %result_2 = corert.get_dst_tensor_type %cpu_handle, %gpu_device
  // CHECK-NEXT: tensor_type = DenseGpu
  %ch2 = "tfrt_test.print_tensor_type"(%result_2, %ch1) : (!tfrt.tensor_type, !tfrt.chain) -> (!tfrt.chain)

  tfrt.return %ch2 : !tfrt.chain
}
