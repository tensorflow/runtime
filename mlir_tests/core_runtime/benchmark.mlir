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

// A function to demonstrate the use of benchmark kernels.
// CHECK-LABEL: --- Running 'BM_corert.matmul'
func @BM_corert.matmul() {
  // CHECK: BM:BM_corert.matmul:Duration(us):
  // CHECK: BM:BM_corert.matmul:Count:
  // CHECK: BM:BM_corert.matmul:Time Min(us):
  // CHECK: BM:BM_corert.matmul:Time 50%(us):
  // CHECK: BM:BM_corert.matmul:Time 95%(us):
  // CHECK: BM:BM_corert.matmul:Time 99%(us):

  // Prepare input.
  %ch0 = hex.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"
  %a_handle = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [1, 1], values = [2.0 : f32] } : 1


  tfrt_test.benchmark "BM_corert.matmul"(%cpu : !corert.device, %a_handle : !corert.tensorhandle, %ch0 : !hex.chain) duration_secs = 1, max_count = 1000
  {
    %result = corert.executeop(%cpu) "tfrt_test.matmul"(%a_handle, %a_handle)
    {transpose_a = false, transpose_b = false}: 1
    hex.return %result : !corert.tensorhandle
  }

  hex.return
}
