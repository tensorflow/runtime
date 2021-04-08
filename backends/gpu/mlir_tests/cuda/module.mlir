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

// CHECK-LABEL: --- Running 'module_load_static_test'
func @module_load_static_test() {
  %ch1 = tfrt.new.chain
  %ch2 = tfrt_cuda.init %ch1
  %index = tfrt.constant.i32 0
  %device = tfrt_cuda.device.get %index, %ch2
  %context, %ch4 = tfrt_cuda_test.context.get %device, %ch2

  // PTX for empty kernel.
  // Typically module loading should be done at initialization time.
  %ch5 = tfrt_cuda.module.load_static %context, %ch4
         { modules = [".version 6.0\n.target sm_35\n.address_size 64\n.visible .entry Kernel() { ret; }" ],
           funcs_per_module = [1 : i32],
           functions = [ "Kernel" ]
         }
  tfrt.return
}

// CHECK-LABEL: --- Running 'module_load_static_bad_ptx_test'
func @module_load_static_bad_ptx_test() {
  %ch1 = tfrt.new.chain
  %ch2 = tfrt_cuda.init %ch1
  %index = tfrt.constant.i32 0
  %device = tfrt_cuda.device.get %index, %ch2
  %context, %ch4 = tfrt_cuda_test.context.get %device, %ch2

  // expected-error @+1 {{CUDA_ERROR_INVALID_IMAGE}}
  %ch5 = tfrt_cuda.module.load_static %context, %ch4
         { modules = ["This is bad!" ],
           funcs_per_module = [1 : i32],
           functions = [ "f" ]
         }
  tfrt.return
}

// CHECK-LABEL: --- Running 'module_load_static_neg_func_count_test'
func @module_load_static_neg_func_count_test() {
  %ch1 = tfrt.new.chain
  %ch2 = tfrt_cuda.init %ch1
  %index = tfrt.constant.i32 0
  %device = tfrt_cuda.device.get %index, %ch2
  %context, %ch4 = tfrt_cuda_test.context.get %device, %ch2

  // expected-error @+1 {{CUDA module table spec is malformed; Invalid function count (-1) specified for module 0}}
  %ch5 = tfrt_cuda.module.load_static %context, %ch4
         { modules = [".version 6.0\n.target sm_35\n.address_size 64\n.visible .entry Kernel() { ret; }"],
           funcs_per_module = [-1 : i32],
           functions = [ "Kernel" ]
         }
  tfrt.return
}

// CHECK-LABEL: --- Running 'module_load_static_bad_func_count_test'
func @module_load_static_bad_func_count_test() {
  %ch1 = tfrt.new.chain
  %ch2 = tfrt_cuda.init %ch1
  %index = tfrt.constant.i32 0
  %device = tfrt_cuda.device.get %index, %ch2
  %context, %ch4 = tfrt_cuda_test.context.get %device, %ch2

  // expected-error @+1 {{CUDA module table spec is malformed; Number of entries in function count list doesn't match number of modules; 2 vs 1}}
  %ch5 = tfrt_cuda.module.load_static %context, %ch4
         { modules = [".version 6.0\n.target sm_35\n.address_size 64\n.visible .entry Kernel() { ret; }" ],
           funcs_per_module = [1 : i32, 2 : i32],
           functions = [ "Kernel" ]
         }
  tfrt.return
}

// CHECK-LABEL: --- Running 'module_load_static_twice_test'
func @module_load_static_twice_test() {
  %ch1 = tfrt.new.chain
  %ch2 = tfrt_cuda.init %ch1
  %index = tfrt.constant.i32 0
  %device = tfrt_cuda.device.get %index, %ch2
  %context, %ch4 = tfrt_cuda_test.context.get %device, %ch2

  %ch5 = tfrt_cuda.module.load_static %context, %ch4
         { modules = [".version 6.0\n.target sm_35\n.address_size 64\n.visible .entry Kernel() { ret; }" ],
           funcs_per_module = [1 : i32],
           functions = [ "Kernel" ]
         }

  // expected-error @+1 {{Unable to load CUDA module table. Table has already been created for device 0}}
  %ch6 = tfrt_cuda.module.load_static %context, %ch5
         { modules = [".version 6.0\n.target sm_35\n.address_size 64\n.visible .entry Kernel() { ret; }" ],
           funcs_per_module = [1 : i32],
           functions = [ "Kernel" ]
         }
  tfrt.return
}
