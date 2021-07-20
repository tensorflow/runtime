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

// RUN: bef_executor %s.bef --test_init_function=register_op_handlers_cpu | FileCheck %s --dump-input=fail

func @register_op_handlers_cpu() {
  %null = "corert.create_null_op_handler"() : () -> !corert.ophandler
  %cpu = "corert.create_cpu_op_handler"(%null) : (!corert.ophandler) -> !corert.ophandler
  corert.register_op_handler %cpu "cpu"
  tfrt.return
}

func @mnist_compute(%cpu: !corert.ophandler,
                    %w1 : !corert.tensorhandle,
                    %b1 : !corert.tensorhandle,
                    %w2 : !corert.tensorhandle,
                    %b2 : !corert.tensorhandle,
                    %test_input_features : !corert.tensorhandle,
                    %test_input_labels : !corert.tensorhandle,
                    %ch0 : !tfrt.chain) -> (!tfrt.chain)
                    //!corert.tensorhandle)
{
  // broadcast_b1 = test.broadcast(b1)
  %broadcast_b1 = corert.executeop(%cpu)
    "tfrt_test.broadcast"(%b1) { shape = [100 : i64, 512 : i64] } : 1

  // a1 = test.matmul(test_input_feature, w1)
  %a1 = corert.executeop(%cpu) "tfrt_test.matmul"(%test_input_features, %w1)
    {transpose_a = false, transpose_b = false}: 1

  // z1 = test.add(a1, broadcast_b1)
  %z1 = corert.executeop(%cpu) "tfrt_test.add"(%a1, %broadcast_b1) : 1

  // activation1 = test.relu(z1)
  %activation1 = corert.executeop(%cpu) "tfrt_test.relu"(%z1) : 1

  // broadcast_b2 = test.broadcast(b2)
  %broadcast_b2 = corert.executeop(%cpu)
    "tfrt_test.broadcast"(%b2) { shape = [100 : i64, 10 : i64] } : 1

  // a2 = test.matmul(activation1, w2)
  %a2 = corert.executeop(%cpu) "tfrt_test.matmul"(%activation1, %w2)
    {transpose_a = false, transpose_b = false}: 1

  // z2 = test.add(a2, broadcast_b2)
  %z2 = corert.executeop(%cpu) "tfrt_test.add"(%a2, %broadcast_b2) : 1

  // activation2 = test.argmax(z2)
  %argmax_h2 = corert.executeop(%cpu)
    "tfrt_test.argmax"(%z2) {axis = 1 : i32} : 1

  // equal_i32 = test.equal(test_input_labels, argmax_h2)
  %equal_i32 = corert.executeop(%cpu)
    "tfrt_test.equal"(%test_input_labels, %argmax_h2) : 1

  // equal_f32 = test.cast(equal_i32)
  %equal_f32= corert.executeop(%cpu)
    "tfrt_test.cast"(%equal_i32) { type = "f32" } : 1

  // avg_accuracy = test.reduce_mean(equal_f32)
  %avg_accuracy = corert.executeop(%cpu)
    "tfrt_test.reduce_mean"(%equal_f32) { axis = 0 : i32 } : 1

  tfrt.return %ch0: !tfrt.chain
}

// CHECK-LABEL: --- Running 'bm_mnist'
func @bm_mnist() {
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"

  // w1
  %w1 = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [784 : i64, 512 : i64], values = [1.0 : f32] } : 1

  // b1
  %b1 = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [512 : i64], values = [1.0 : f32] } : 1

  // w2
  %w2 = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [512 : i64, 10 : i64], values = [1.0 : f32] } : 1

  // b2
  %b2 = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [10 : i64], values = [1.0 : f32] } : 1

  // test_input_features
  %test_input_features= corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [100 : i64, 784 : i64], values = [1.0 : f32] } : 1

  // test_input_labels
  %test_input_labels= corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [100 : i64], values = [1 : i32] } : 1


  tfrt_test.benchmark "bm_mnist"(
      %cpu: !corert.ophandler,
      %w1 : !corert.tensorhandle,
      %b1 : !corert.tensorhandle,
      %w2 : !corert.tensorhandle,
      %b2 : !corert.tensorhandle,
      %test_input_features : !corert.tensorhandle,
      %test_input_labels : !corert.tensorhandle,
      %ch0: !tfrt.chain)
      duration_secs = 10, max_count = 10000, num_warmup_runs = 10 {
      %avg_accuracy = tfrt.call @mnist_compute(%cpu, %w1, %b1, %w2, %b2, %test_input_features, %test_input_labels, %ch0)
       : (!corert.ophandler, !corert.tensorhandle, !corert.tensorhandle,
          !corert.tensorhandle, !corert.tensorhandle,
          !corert.tensorhandle, !corert.tensorhandle,
          !tfrt.chain) -> !tfrt.chain

      tfrt.return %avg_accuracy : !tfrt.chain
  }

  tfrt.return
}
