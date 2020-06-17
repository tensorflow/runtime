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

//===- parallel.benchmark.mlir --------------------------------------------===//
//
// These benchmarks are focused on testing hex parallel_for and parallel_call
// runtime overheads, and not on the actual workload.
//
//===----------------------------------------------------------------------===//

// RUN: tfrt_translate -mlir-to-bef %s             \
// RUN:   | bef_executor --work_queue_type=mstd:8  \
// RUN:   | FileCheck %s --dump-input=fail

func @native_sink(%a: i32, %b: i32) -> () attributes {hex.native}
func @native_async_sink(%a: i32, %b: i32) -> !hex.chain attributes {hex.native}

// CHECK-LABEL: --- Running 'parallel_for.async_body.benchmark'
func @parallel_for.async_body.benchmark() {
  %start      = hex.constant.i32    0
  %end        = hex.constant.i32 1000
  %block_size = hex.constant.i32    1

  tfrt_test.benchmark "parallel_for.async_body"(
    %start : i32, %end : i32, %block_size : i32
  ) duration_secs = 3, max_count = 1000000 {

    %done = hex.parallel_for.i32 %start to %end fixed %block_size {
      %0 = hex.add.i32 %start, %end
      %ch0 = hex.new.chain
      hex.return %ch0 :!hex.chain
    }

    hex.return %done : !hex.chain
  }

  hex.return
}

// CHECK-LABEL: --- Running 'parallel_for.sync_body.benchmark'
func @parallel_for.sync_body.benchmark() {
  %start      = hex.constant.i32    0
  %end        = hex.constant.i32 1000
  %block_size = hex.constant.i32    1

  tfrt_test.benchmark "parallel_for.sync_body"(
    %start : i32, %end : i32, %block_size : i32
  ) duration_secs = 3, max_count = 1000000 {

    %done = hex.parallel_for.i32 %start to %end fixed %block_size {
      hex.add.i32 %start, %end
      hex.return
    }

    hex.return %done : !hex.chain
  }

  hex.return
}

// CHECK-LABEL: --- Running 'parallel_call.async_fn.benchmark'
func @parallel_call.async_fn.benchmark() {
  %start      = hex.constant.i32    0
  %end        = hex.constant.i32 1000
  %block_size = hex.constant.i32    1

  tfrt_test.benchmark "parallel_call.async_fn"(
    %start : i32, %end : i32, %block_size : i32
  ) duration_secs = 3, max_count = 1000000 {

    %done = hex.parallel_call.i32 %start to %end fixed %block_size
            @native_async_sink()

    hex.return %done : !hex.chain
  }

  hex.return
}

// CHECK-LABEL: --- Running 'parallel_call.sync_fn.benchmark'
func @parallel_call.sync_fn.benchmark() {
  %start      = hex.constant.i32    0
  %end        = hex.constant.i32 1000
  %block_size = hex.constant.i32    1

  tfrt_test.benchmark "parallel_call.sync_fn"(
    %start : i32, %end : i32, %block_size : i32
  ) duration_secs = 3, max_count = 1000000 {

    %done = hex.parallel_call.i32 %start to %end fixed %block_size
            @native_sink()

    hex.return %done : !hex.chain
  }

  hex.return
}
