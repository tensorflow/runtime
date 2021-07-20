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

// RUN: bef_executor %s.bef | FileCheck %s --dump-input=fail
// RUN: bef_executor -work_queue_type=mstd %s.bef | FileCheck %s --dump-input=fail

// CHECK-LABEL: --- Running 'parallel_for.fixed_block_size.async'
func @parallel_for.fixed_block_size.async() -> !tfrt.chain {
  %start      = tfrt.constant.i32 0
  %end        = tfrt.constant.i32 10
  %block_size = tfrt.constant.i32 1

  %cnt0 = "tfrt_test.atomic.create.i32"() : () -> !test.atomic.i32
  %cnt1 = "tfrt_test.atomic.create.i32"() : () -> !test.atomic.i32

  %done = tfrt.parallel_for.i32 %start to %end fixed %block_size, %cnt0, %cnt1
          : !test.atomic.i32, !test.atomic.i32 {
    %ch0 = tfrt.new.chain

    %ch1 = "tfrt_test.atomic.add.i32"(%cnt0, %start, %ch0)
           : (!test.atomic.i32, i32, !tfrt.chain) -> !tfrt.chain

    %ch2 = "tfrt_test.atomic.add.i32"(%cnt1, %end, %ch1)
           : (!test.atomic.i32, i32, !tfrt.chain) -> !tfrt.chain

    tfrt.return %ch2 : !tfrt.chain
  }

  %v0, %ch0 = "tfrt_test.atomic.get.i32"(%cnt0, %done)
     : (!test.atomic.i32, !tfrt.chain) -> (i32, !tfrt.chain)
  %v1, %ch1 = "tfrt_test.atomic.get.i32"(%cnt1, %ch0)
     : (!test.atomic.i32, !tfrt.chain) -> (i32, !tfrt.chain)

  // CHECK: int32 = 45
  %ch2 = tfrt.print.i32 %v0, %ch1
  // CHECK: int32 = 55
  %ch3 = tfrt.print.i32 %v1, %ch2

  tfrt.return %ch3 : !tfrt.chain
}

// Asynchronous function signals its completion using result chain.
func @async_fn(%start : i32, %end : i32,
               %cnt0 : !test.atomic.i32,
               %cnt1 : !test.atomic.i32) -> !tfrt.chain {
    %ch0 = tfrt.new.chain

    %ch1 = "tfrt_test.atomic.add.i32"(%cnt0, %start, %ch0)
           : (!test.atomic.i32, i32, !tfrt.chain) -> !tfrt.chain

    %ch2 = "tfrt_test.atomic.add.i32"(%cnt1, %end, %ch1)
           : (!test.atomic.i32, i32, !tfrt.chain) -> !tfrt.chain

    tfrt.return %ch2 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'parallel_call.fixed_block_size.async'
func @parallel_call.fixed_block_size.async() -> !tfrt.chain {
  %start      = tfrt.constant.i32 0
  %end        = tfrt.constant.i32 10
  %block_size = tfrt.constant.i32 1

  %cnt0 = "tfrt_test.atomic.create.i32"() : () -> !test.atomic.i32
  %cnt1 = "tfrt_test.atomic.create.i32"() : () -> !test.atomic.i32

  %done = tfrt.parallel_call.i32 %start to %end fixed %block_size
          @async_fn(%cnt0, %cnt1) : !test.atomic.i32, !test.atomic.i32

  %v0, %ch0 = "tfrt_test.atomic.get.i32"(%cnt0, %done)
     : (!test.atomic.i32, !tfrt.chain) -> (i32, !tfrt.chain)
  %v1, %ch1 = "tfrt_test.atomic.get.i32"(%cnt1, %ch0)
     : (!test.atomic.i32, !tfrt.chain) -> (i32, !tfrt.chain)

  // CHECK: int32 = 45
  %ch2 = tfrt.print.i32 %v0, %ch1
  // CHECK: int32 = 55
  %ch3 = tfrt.print.i32 %v1, %ch2

  tfrt.return %ch3 : !tfrt.chain
}
