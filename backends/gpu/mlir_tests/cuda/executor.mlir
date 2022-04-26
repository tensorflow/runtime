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
// RUN:   -tfrt-set-entry-point='platform=CUDA buffer_sizes=64' %s \
// RUN: | tfrt_gpu_translate -mlir-to-bef \
// RUN: | tfrt_gpu_executor \
// RUN: | FileCheck %s

func.func @main(
  %arg0 : !tfrt.chain,
  %arg1 : !tfrt_gpu.stream,
  %arg2 : !tfrt_gpu.buffer
) -> !tfrt.chain {
  %ch0 = tfrt_gpu.stream.synchronize %arg1, %arg0
  // CHECK: GpuBuffer<pointer={{0x[[:xdigit:]]+}} (CUDA), size=64>
  %ch1 = tfrt_gpu.mem.print_metadata %arg2, %ch0
  tfrt.return %ch1 : !tfrt.chain
}
