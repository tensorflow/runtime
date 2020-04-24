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

// Note: the program has a specific implementation, meant to match exactly the
// operations made in similar benchmarks. So the divisions and equality tests
// should not be changed.

func @fizzbuzz_compute(%n : i32) -> (i32, i32, i32)
{
  %zero = hex.constant.i32 0
  %one = hex.constant.i32 1
  %two = hex.constant.i32 2
  %three = hex.constant.i32 3
  %six = hex.constant.i32 6

  // TODO(b/146014117): Don't compute constant.i32 0 multiple times.
  %fizzbuzz0 = hex.constant.i32 0
  %buzz0 = hex.constant.i32 0
  %fizz0 = hex.constant.i32 0

  %i0 = hex.constant.i32 0

  %i2, %fizzbuzz3, %buzz4, %fizz5, %zero1, %one1, %two1, %three1, %six1 =
      hex.repeat.i32
      %n, %i0, %fizzbuzz0, %buzz0, %fizz0, %zero, %one, %two, %three, %six :
      i32, i32, i32, i32, i32, i32, i32, i32, i32 {
    %i1 = hex.add.i32 %i0, %one

    %quot, %rem = hex.div.i32 %i1, %six
    %isfizzbuzz = hex.equal.i32 %rem, %zero

    %fizzbuzz2, %buzz3, %fizz4 = hex.if
        %isfizzbuzz, %i1, %fizzbuzz0, %buzz0, %fizz0, %zero, %one, %two, %three :
        (i32, i32, i32, i32, i32, i32, i32, i32) -> (i32, i32, i32) {

        %fizzbuzz1 = hex.add.i32 %fizzbuzz0, %one
        hex.return %fizzbuzz1, %buzz0, %fizz0 : i32, i32, i32

    } else {

      %quot, %rem = hex.div.i32 %i1, %three
      %isfizz = hex.equal.i32 %rem, %zero

      %buzz2, %fizz3 = hex.if
          %isfizz, %i1, %buzz0, %fizz0, %zero, %one, %two :
          (i32, i32, i32, i32, i32, i32) -> (i32, i32) {

        %buzz1 = hex.add.i32 %buzz0, %one
        hex.return %buzz1, %fizz0 : i32, i32

      } else {

        %quot, %rem = hex.div.i32 %i1, %two
        %isbuzz = hex.equal.i32 %rem, %zero

        %fizz2 = hex.if
            %isbuzz, %i1, %fizz0, %one :
            (i32, i32, i32) -> (i32) {
          %fizz1 = hex.add.i32 %fizz0, %one
          hex.return %fizz1 : i32
        } else {
          hex.return %fizz0 : i32
        }

        hex.return %buzz0, %fizz2 : i32, i32
      }

      hex.return %fizzbuzz0, %buzz2, %fizz3 : i32, i32, i32
    }

    hex.return %i1, %fizzbuzz2, %buzz3, %fizz4, %zero, %one, %two, %three, %six :
        i32, i32, i32, i32, i32, i32, i32, i32, i32
  }

  hex.return %fizzbuzz3, %buzz4, %fizz5 : i32, i32, i32
}


// CHECK-LABEL: --- Running 'fizzbuzz'
func @fizzbuzz() -> !hex.chain {
  %ch0 = hex.new.chain

  %n = hex.constant.i32 1000

  %fizzbuzz0, %fizz0, %buzz0 =
      hex.call @fizzbuzz_compute(%n) : (i32) -> (i32, i32, i32)

  // 1000 // 6
  // CHECK: int32 = 166
  %ch1 = hex.print.i32 %fizzbuzz0, %ch0
  // 1000 // 3 - 1000 // 6
  // CHECK: int32 = 167
  %ch2 = hex.print.i32 %fizz0, %ch1
  // 1000 // 2 - 1000 // 6
  // CHECK: int32 = 334
  %ch3 = hex.print.i32 %buzz0, %ch2

  hex.return %ch3 : !hex.chain
}

// CHECK-LABEL: --- Running 'bm_fizzbuzz'
func @bm_fizzbuzz() {
  %n = hex.constant.i32 1000

  tfrt_test.benchmark "bm_fizzbuzz"(%n : i32)
      duration_secs = 4, max_count = 10000, num_warmup_runs = 10 {
      %fizzbuzz0, %fizz0, %buzz0 =
          hex.call @fizzbuzz_compute(%n) : (i32) -> (i32, i32, i32)
      hex.return %n : i32
  }

  hex.return
}
