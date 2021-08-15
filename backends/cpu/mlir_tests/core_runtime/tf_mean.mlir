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

// CHECK: --- Running 'mean'
func @mean() -> !tfrt.chain {
  %ch_1 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_1 "cpu"

  %input_1 = corert.executeop(%cpu) "tf.Const"()
    { dtype = f32, value = dense<[[[[1.0], [2.0]], [[3.0], [4.0]]]]> : tensor<1x2x2x1xf32> } : 1
  %input_2 = corert.executeop(%cpu) "tf.Const"()
    { dtype = i32, value = dense<[1, 2]> : tensor<2xi32> } : 1
  %output = corert.executeop(%cpu) "tf.Mean"(%input_1, %input_2)
    { T = f32, Tidx = i32 } : 1

  // CHECK: DenseHostTensor dtype = f32, shape = [1, 1], values = [2.500000e+00]
  %ch_2 = corert.executeop.seq(%cpu, %ch_1) "tfrt_test.print"(%output) : 0
  tfrt.return %ch_2 : !tfrt.chain
}

// CHECK: --- Running 'mean_keep_dims'
func @mean_keep_dims() -> !tfrt.chain {
  %ch_1 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_1 "cpu"

  %input_1 = corert.const_dense_tensor dense<[[[[1.0], [2.0]], [[3.0], [4.0]]]]> : tensor<1x2x2x1xf32>
  %input_2 = corert.const_dense_tensor dense<[1, -2]> : tensor<2xi32>
  %output = corert.executeop(%cpu) "tf.Mean"(%input_1, %input_2)
    { T = f32, Tidx = i32, keep_dims = true } : 1

  // CHECK: DenseHostTensor dtype = f32, shape = [1, 1, 1, 1], values = [2.500000e+00]
  %ch_2 = corert.executeop.seq(%cpu, %ch_1) "tfrt_test.print"(%output) : 0
  tfrt.return %ch_2 : !tfrt.chain
}

// CHECK: --- Running 'mean_i32'
func @mean_i32() -> !tfrt.chain {
  %ch_1 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_1 "cpu"

  %input_1 = corert.executeop(%cpu) "tf.Const"()
    { dtype = i32, value = dense<[[[[1], [2]], [[3], [4]]]]> : tensor<1x2x2x1xi32> } : 1
  %input_2 = corert.executeop(%cpu) "tf.Const"()
    { dtype = i32, value = dense<[1, 2]> : tensor<2xi32> } : 1
  %output = corert.executeop(%cpu) "tf.Mean"(%input_1, %input_2)
    { T = i32, Tidx = i32 } : 1

  // CHECK: DenseHostTensor dtype = i32, shape = [1, 1], values = [2]
  %ch_2 = corert.executeop.seq(%cpu, %ch_1) "tfrt_test.print"(%output) : 0
  tfrt.return %ch_2 : !tfrt.chain
}
