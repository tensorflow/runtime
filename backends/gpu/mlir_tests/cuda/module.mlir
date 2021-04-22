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

// RUN: bef_executor $(bef_name %s) | FileCheck %s --dump-input=fail
// RUN: tfrt_gpu_opt %s | tfrt_gpu_opt

// CHECK-LABEL: --- Running 'function_test'
func @function_test() {
  %ch2 = tfrt.new.chain
  %index = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get %index, %ch2 { platform = 1 : i32 }
  %context = tfrt_gpu.context.create %device, %ch2

  // PTX for empty kernel.
  // Typically module loading should be done at initialization time.
  %func = tfrt_gpu.function.load %context, %ch2 {
    data = ".version 6.0\n.target sm_35\n.address_size 64\n.visible .entry Kernel() { ret; }\00",
    key = 0 : ui64,
    name = "Kernel\00"
  }

  tfrt.return
}

// CHECK-LABEL: --- Running 'function_bad_data_test'
func @function_bad_data_test() {
  %ch2 = tfrt.new.chain
  %index = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get %index, %ch2 { platform = 1 : i32 }
  %context = tfrt_gpu.context.create %device, %ch2

  // expected-error @+1 {{CUDA_ERROR_INVALID_IMAGE}}
  %func = tfrt_gpu.function.load %context, %ch2 {
    data = "invalid image\00",
    key = 0 : ui64,
    name = "Kernel\00"
  }

  tfrt.return
}

// CHECK-LABEL: --- Running 'function_bad_name_test'
func @function_bad_name_test() {
  %ch2 = tfrt.new.chain
  %index = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get %index, %ch2 { platform = 1 : i32 }
  %context = tfrt_gpu.context.create %device, %ch2

  // expected-error @+1 {{CUDA_ERROR_NOT_FOUND}}
  %func = tfrt_gpu.function.load %context, %ch2 {
    data = ".version 6.0\n.target sm_35\n.address_size 64\n.visible .entry Kernel() { ret; }\00",
    key = 0 : ui64,
    name = "Foo\00"
  }

  tfrt.return
}

// CHECK-LABEL: --- Running 'function_not_null_terminated_test'
func @function_not_null_terminated_test() {
  %ch2 = tfrt.new.chain
  %index = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get %index, %ch2 { platform = 1 : i32 }
  %context = tfrt_gpu.context.create %device, %ch2

  // expected-error @+1 {{data attribute must be null-terminated}}
  %func0 = tfrt_gpu.function.load %context, %ch2 {
    data = "not null-terminated",
    key = 0 : ui64,
    name = "\00"
  }

  // expected-error @+1 {{name attribute must be null-terminated}}
  %func1 = tfrt_gpu.function.load %context, %ch2 {
    data = "\00",
    key = 0 : ui64,
    name = "not null-terminated"
  }

  tfrt.return
}
