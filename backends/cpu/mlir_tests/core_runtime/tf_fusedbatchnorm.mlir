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

// RUN: bef_executor --test_init_function=register_op_handlers_cpu %s.bef | FileCheck %s --dump-input=fail

func @register_op_handlers_cpu() {
  %null = "corert.create_null_op_handler"() : () -> !corert.ophandler
  %cpu = "corert.create_cpu_op_handler"(%null) : (!corert.ophandler) -> !corert.ophandler
  corert.register_op_handler %cpu "cpu"
  tfrt.return
}

// CHECK: --- Running 'fused_batch_norm_v3'
func @fused_batch_norm_v3() -> !tfrt.chain {
  %ch_epoch = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_epoch "cpu"

  // Test tf.FusedBatchNormV3.
  %input = corert.executeop(%cpu) "tf.Const"()
      { dtype = f32, value = dense<[[[[1.0], [-1.0]], [[-1.0], [1.0]]]]> : tensor<1x2x2x1xf32> } : 1
  %scale = corert.executeop(%cpu) "tf.Const"()
      { dtype = f32, value = dense<1.0> : tensor<1x1x1x1xf32> } : 1
  %bias = corert.executeop(%cpu) "tf.Const"()
      { dtype = f32, value = dense<0.0> : tensor<1x1x1x1xf32> } : 1
  %mean = corert.executeop(%cpu) "tf.Const"()
      { dtype = f32, value = dense<0.0> : tensor<1x1x1x1xf32> } : 1
  %variance = corert.executeop(%cpu) "tf.Const"()
      { dtype = f32, value = dense<1.0> : tensor<1x1x1x1xf32> } : 1
  %res: 6 = corert.executeop(%cpu) "tf.FusedBatchNormV3"(%input, %scale, %bias, %mean, %variance)
      { T = f32, U = f32, epsilon = 0.0 : f32, data_format = "NHWC", is_training = false } : 6

  // CHECK: DenseHostTensor dtype = f32, shape = [1, 2, 2, 1], values = [1.000000e+00, -1.000000e+00, -1.000000e+00, 1.000000e+00]
  %ch_print = corert.executeop.seq(%cpu, %ch_epoch) "tfrt_test.print"(%res#0) : 0
  tfrt.return %ch_print : !tfrt.chain
}
