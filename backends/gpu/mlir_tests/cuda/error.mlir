// Copyright 2021 The TensorFlow Runtime Authors
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

// RUN: tfrt_gpu_opt -mlir-print-debuginfo \
// RUN:   -test-set-entry-point='platform=CUDA function_name=error' %s \
// RUN: | tfrt_gpu_translate -mlir-to-bef \
// RUN: | tfrt_gpu_executor

// RUN: bef_executor_lite %s.bef

func @error(
  %arg0 : !tfrt.chain,
  %arg1 : !tfrt_gpu.stream
) -> !tfrt.chain {
  %ordinal = tfrt.constant.i32 -1
  // expected-error@+1 {{CUDA_ERROR_INVALID_DEVICE}}
  %device = tfrt_gpu.device.get CUDA, %ordinal
  tfrt.return %arg0 : !tfrt.chain
}

func @main() {
  %ordinal = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get CUDA, %ordinal
  %context = tfrt_gpu.context.create %device
  %stream = tfrt_gpu.stream.create %context
  %ch0 = tfrt.new.chain
  %ch1 = tfrt.call @error(%ch0, %stream)
    : (!tfrt.chain, !tfrt_gpu.stream) -> (!tfrt.chain)
  tfrt.return
}
