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

// RUN: bef_executor --test_init_function=register_op_handlers_cpu %s.bef | FileCheck %s

func @register_op_handlers_cpu() {
  %null = "corert.create_null_op_handler"() : () -> !corert.ophandler
  %cpu = "corert.create_cpu_op_handler"(%null) : (!corert.ophandler) -> !corert.ophandler
  corert.register_op_handler %cpu "cpu"
  tfrt.return
}

// CHECK-LABEL: --- Running 'BM_FusedMatMulBiasRelu_256x256x256_f32'
func @BM_FusedMatMulBiasRelu_256x256x256_f32() {
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"

  %lhs = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [256, 256], values = [1.0 : f32] } : 1

  %rhs = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [256, 256], values = [1.0 : f32] } : 1

  %bias = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [256], values = [1.0 : f32] } : 1

  tfrt_test.benchmark "BM_FusedMatMulBiasRelu_256x256x256_f32"(
      %cpu     : !corert.ophandler,
      %lhs     : !corert.tensorhandle,
      %rhs     : !corert.tensorhandle,
      %bias    : !corert.tensorhandle
  )
  duration_secs = 5, max_count = 1000000, num_warmup_runs = 10
  {
    %result  = corert.executeop(%cpu)
      "tf._FusedMatMul"(%lhs, %rhs, %bias) {
        fused_ops = ["BiasAdd", "Relu"],
        transpose_a = false, transpose_b = false
      } : 1

    tfrt.return %result : !corert.tensorhandle
  }

  tfrt.return
}
