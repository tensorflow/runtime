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
// These benchmarks are focused on testing tfrt parallel_for and parallel_call
// runtime overheads, and not on the actual workload.
//
//===----------------------------------------------------------------------===//

// RUN: bef_executor --work_queue_type=mstd:8 %s.bef | FileCheck %s --dump-input=fail

func private @native_sink(%a: i32, %b: i32) -> () attributes {tfrt.native}
func private @native_async_sink(%a: i32, %b: i32) -> !tfrt.chain attributes {tfrt.native}

// CHECK-LABEL: --- Running 'parallel_for.async_body.benchmark'
func @parallel_for.async_body.benchmark() {
  %start      = tfrt.constant.i32    0
  %end        = tfrt.constant.i32 1000
  %block_size = tfrt.constant.i32    1

  tfrt_test.benchmark "parallel_for.async_body"(
    %start : i32, %end : i32, %block_size : i32
  ) duration_secs = 3, max_count = 1000000 {

    %done = tfrt.parallel_for.i32 %start to %end fixed %block_size {
      %0 = tfrt.add.i32 %start, %end
      %ch0 = tfrt.new.chain
      tfrt.return %ch0 :!tfrt.chain
    }

    tfrt.return %done : !tfrt.chain
  }

  tfrt.return
}

// CHECK-LABEL: --- Running 'parallel_for.sync_body.benchmark'
func @parallel_for.sync_body.benchmark() {
  %start      = tfrt.constant.i32    0
  %end        = tfrt.constant.i32 1000
  %block_size = tfrt.constant.i32    1

  tfrt_test.benchmark "parallel_for.sync_body"(
    %start : i32, %end : i32, %block_size : i32
  ) duration_secs = 3, max_count = 1000000 {

    %done = tfrt.parallel_for.i32 %start to %end fixed %block_size {
      tfrt.add.i32 %start, %end
      tfrt.return
    }

    tfrt.return %done : !tfrt.chain
  }

  tfrt.return
}

// CHECK-LABEL: --- Running 'parallel_call.async_fn.benchmark'
func @parallel_call.async_fn.benchmark() {
  %start      = tfrt.constant.i32    0
  %end        = tfrt.constant.i32 1000
  %block_size = tfrt.constant.i32    1

  tfrt_test.benchmark "parallel_call.async_fn"(
    %start : i32, %end : i32, %block_size : i32
  ) duration_secs = 3, max_count = 1000000 {

    %done = tfrt.parallel_call.i32 %start to %end fixed %block_size
            @native_async_sink()

    tfrt.return %done : !tfrt.chain
  }

  tfrt.return
}

// CHECK-LABEL: --- Running 'parallel_call.sync_fn.benchmark'
func @parallel_call.sync_fn.benchmark() {
  %start      = tfrt.constant.i32    0
  %end        = tfrt.constant.i32 1000
  %block_size = tfrt.constant.i32    1

  tfrt_test.benchmark "parallel_call.sync_fn"(
    %start : i32, %end : i32, %block_size : i32
  ) duration_secs = 3, max_count = 1000000 {

    %done = tfrt.parallel_call.i32 %start to %end fixed %block_size
            @native_sink()

    tfrt.return %done : !tfrt.chain
  }

  tfrt.return
}
