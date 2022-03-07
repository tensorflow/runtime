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

// RUN: bef_executor_lite %s.bef | FileCheck %s

// CHECK-LABEL: --- Running 'function_test'
func @function_test() {
  %ordinal = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get ROCm, %ordinal
  %context = tfrt_gpu.context.create %device

  // PTX for empty kernel.
  // Typically module loading should be done at initialization time.
  %module = tfrt_gpu.module.load %context {
    data = "extern \"C\" __global__ void Kernel() { return; }\00"
  }

  %func = tfrt_gpu.module.get_function %module { name = "Kernel" }

  tfrt.return
}

func @global_test() {
  %ordinal = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get ROCm, %ordinal
  %context = tfrt_gpu.context.create %device

  // PTX for a module with a global symbol.
  %module = tfrt_gpu.module.load %context {
    data = "__device__ unsigned int Global[128];\00"
  }

  %global = tfrt_gpu.module.get_global %module { name = "Global" }

  tfrt.return
}

// CHECK-LABEL: --- Running 'module_bad_data_test'
func @module_bad_data_test() {
  %ch2 = tfrt.new.chain
  %ordinal = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get ROCm, %ordinal
  %context = tfrt_gpu.context.create %device

  // expected-error @+1 {{hipErrorInvalidValue}}
  %func = tfrt_gpu.module.load %context {
    data = "invalid image\00"
  }

  tfrt.return
}

// CHECK-LABEL: --- Running 'function_bad_name_test'
func @function_bad_name_test() {
  %ch2 = tfrt.new.chain
  %ordinal = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get ROCm, %ordinal
  %context = tfrt_gpu.context.create %device

  %module = tfrt_gpu.module.load %context {
    data = "extern \"C\" __global__ void Kernel() { return; }\00"
  }

  // expected-error @+1 {{hipErrorNotFound}}
  %func = tfrt_gpu.module.get_function %module { name = "Foo\00" }

  tfrt.return
}

// CHECK-LABEL: --- Running 'module_not_null_terminated_test'
func @module_not_null_terminated_test() {
  %ch2 = tfrt.new.chain
  %ordinal = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get ROCm, %ordinal
  %context = tfrt_gpu.context.create %device

  // expected-error @+1 {{data attribute must be null-terminated}}
  %module = tfrt_gpu.module.load %context {
    data = "not null-terminated"
  }

  tfrt.return
}
