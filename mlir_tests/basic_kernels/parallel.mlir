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

// RUN: tfrt_translate --mlir-to-bef %s | bef_executor | FileCheck %s --dump-input=fail
// RUN: tfrt_translate --mlir-to-bef %s | bef_executor -work_queue_type=mstd | FileCheck %s --dump-input=fail

// CHECK-LABEL: --- Running 'parallel_for.fixed_block_size'
func @parallel_for.fixed_block_size() -> !hex.chain {
  %start      = hex.constant.i32 0
  %end        = hex.constant.i32 10
  %block_size = hex.constant.i32 1

  %cnt0 = "tfrt_test.atomic.create.i32"() : () -> !test.atomic.i32
  %cnt1 = "tfrt_test.atomic.create.i32"() : () -> !test.atomic.i32

  %done = hex.parallel_for.i32 %start to %end fixed %block_size, %cnt0, %cnt1
          : !test.atomic.i32, !test.atomic.i32 {
    %ch0 = hex.new.chain

    %ch1 = "tfrt_test.atomic.add.i32"(%cnt0, %start, %ch0)
           : (!test.atomic.i32, i32, !hex.chain) -> !hex.chain

    %ch2 = "tfrt_test.atomic.add.i32"(%cnt1, %end, %ch1)
           : (!test.atomic.i32, i32, !hex.chain) -> !hex.chain

    hex.return %ch2 : !hex.chain
  }

  %v0, %ch0 = "tfrt_test.atomic.get.i32"(%cnt0, %done)
     : (!test.atomic.i32, !hex.chain) -> (i32, !hex.chain)
  %v1, %ch1 = "tfrt_test.atomic.get.i32"(%cnt1, %ch0)
     : (!test.atomic.i32, !hex.chain) -> (i32, !hex.chain)

  // CHECK: int32 = 45
  %ch2 = hex.print.i32 %v0, %ch1
  // CHECK: int32 = 55
  %ch3 = hex.print.i32 %v1, %ch2

  hex.return %ch3 : !hex.chain
}
