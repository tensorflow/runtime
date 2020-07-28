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

// RUN: bef_executor -devices=cpu $(bef_name %s) | FileCheck %s --dump-input=fail

// CHECK: --- Running 'zeroslike_f32'
func @zeroslike_f32() -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"

  %operand = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [2, 3], values = [-1.0 : f32, -0.5 : f32, 0.0 : f32, 0.5 : f32, 1.0 : f32, 1.5 : f32] } : 1

  %cpu_handle_result = corert.executeop(%cpu) "tf.ZerosLike"(%operand) { T = f32 } : 1

  // CHECK: ScalarHostTensor dtype = F32, shape = [2, 3], value = 0.000000e+00
  %ch_print_cpu = corert.executeop.seq(%cpu, %ch0) "tfrt_test.print"(%cpu_handle_result) : 0
  tfrt.return %ch_print_cpu : !tfrt.chain
}
