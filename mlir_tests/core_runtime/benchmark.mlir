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

// CHECK-LABEL: --- Running 'BM_corert.matmul'
func @BM_corert.matmul() {
  // CHECK: BM:BM_corert.matmul:Duration(ns):
  // CHECK: BM:BM_corert.matmul:Count: 1000
  // CHECK: BM:BM_corert.matmul:Time Min(ns):
  // CHECK: BM:BM_corert.matmul:Time 50%(ns):
  // CHECK: BM:BM_corert.matmul:Time 95%(ns):
  // CHECK: BM:BM_corert.matmul:Time 99%(ns):
  // CHECK: BM:BM_corert.matmul:CPU Min(ns):
  // CHECK: BM:BM_corert.matmul:CPU 50%(ns):
  // CHECK: BM:BM_corert.matmul:CPU 95%(ns):
  // CHECK: BM:BM_corert.matmul:CPU 99%(ns):
  // CHECK: BM:BM_corert.matmul:CPU utilization(percent):

  // Prepare input.
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"
  %a_handle = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [1, 1], values = [2.0 : f32] } : 1


  tfrt_test.benchmark "BM_corert.matmul"(%cpu : !corert.ophandler, %a_handle : !corert.tensorhandle, %ch0 : !tfrt.chain) duration_secs = 1, max_count = 1000
  {
    %result = corert.executeop(%cpu) "tfrt_test.matmul"(%a_handle, %a_handle)
      {transpose_a = false, transpose_b = false}: 1
    tfrt.return %result : !corert.tensorhandle
  }

  tfrt.return
}

func @matmul_sync(%cpu : !corert.ophandler, %a_handle : !corert.tensorhandle) -> () attributes {tfrt.sync} {
  %result = corert_sync.executeop(%cpu) "tfrt_test.matmul"(%a_handle, %a_handle)
    {transpose_a = false, transpose_b = false}: 1
  tfrt.return
}

// CHECK-LABEL: --- Running 'BM_corert.matmul_sync'
func @BM_corert.matmul_sync() attributes {tfrt.sync} {
  // CHECK: BM:matmul_sync:Duration(ns):
  // CHECK: BM:matmul_sync:Count: 1000
  // CHECK: BM:matmul_sync:Time Min(ns):
  // CHECK: BM:matmul_sync:Time 50%(ns):
  // CHECK: BM:matmul_sync:Time 95%(ns):
  // CHECK: BM:matmul_sync:Time 99%(ns):
  // CHECK: BM:matmul_sync:CPU Min(ns):
  // CHECK: BM:matmul_sync:CPU 50%(ns):
  // CHECK: BM:matmul_sync:CPU 95%(ns):
  // CHECK: BM:matmul_sync:CPU 99%(ns):
  // CHECK: BM:matmul_sync:CPU utilization(percent):

  // Prepare input.
  %cpu = corert_sync.get_op_handler "cpu"
  %a_handle = corert_sync.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [1, 1], values = [2.0 : f32] } : 1


  tfrt_test.sync_benchmark @matmul_sync(%cpu : !corert.ophandler, %a_handle : !corert.tensorhandle) duration_secs = 1, max_count = 1000

  tfrt.return
}
