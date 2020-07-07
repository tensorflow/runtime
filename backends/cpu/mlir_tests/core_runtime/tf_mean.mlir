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

// RUN: tfrt_translate -mlir-to-bef %s | bef_executor -devices=cpu | FileCheck %s --dump-input=fail

// CHECK: --- Running 'mean'
func @mean() -> !hex.chain {
  %ch_1 = hex.new.chain
  %cpu = corert.get_device "cpu"

  %input_1 = corert.executeop(%cpu) "tf.Const"()
    { dtype = f32, value = dense<[[[[1.0], [2.0]], [[3.0], [4.0]]]]> : tensor<1x2x2x1xf32> } : 1
  %input_2 = corert.executeop(%cpu) "tf.Const"()
    { dtype = i32, value = dense<[1, 2]> : tensor<2xi32> } : 1
  %output = corert.executeop(%cpu) "tf.Mean"(%input_1, %input_2)
    { T = f32, Tidx = i32 } : 1

  // CHECK: DenseHostTensor dtype = F32, shape = [1, 1], values = [2.500000e+00]
  %ch_2 = corert.executeop.seq(%cpu, %ch_1) "tfrt_test.print"(%output) : 0
  hex.return %ch_2 : !hex.chain
}

