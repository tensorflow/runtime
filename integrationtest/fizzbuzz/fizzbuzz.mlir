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

// RUN: bef_executor %s.bef | FileCheck %s

// Note: the program has a specific implementation, meant to match exactly the
// operations made in similar benchmarks. So the divisions and equality tests
// should not be changed.

func @fizzbuzz_compute(%n : i32) -> (i32, i32, i32)
{
  %zero = tfrt.constant.i32 0
  %one = tfrt.constant.i32 1
  %two = tfrt.constant.i32 2
  %three = tfrt.constant.i32 3
  %six = tfrt.constant.i32 6

  // TODO(b/146014117): Don't compute constant.i32 0 multiple times.
  %fizzbuzz0 = tfrt.constant.i32 0
  %buzz0 = tfrt.constant.i32 0
  %fizz0 = tfrt.constant.i32 0

  %i0 = tfrt.constant.i32 0

  %i2, %fizzbuzz3, %buzz4, %fizz5, %zero1, %one1, %two1, %three1, %six1 =
      tfrt.repeat.i32
      %n, %i0, %fizzbuzz0, %buzz0, %fizz0, %zero, %one, %two, %three, %six :
      i32, i32, i32, i32, i32, i32, i32, i32, i32 {
    %i1 = tfrt.add.i32 %i0, %one

    %quot, %rem = tfrt.div.i32 %i1, %six
    %isfizzbuzz = tfrt.equal.i32 %rem, %zero

    %fizzbuzz2, %buzz3, %fizz4 = tfrt.if
        %isfizzbuzz, %i1, %fizzbuzz0, %buzz0, %fizz0, %zero, %one, %two, %three :
        (i32, i32, i32, i32, i32, i32, i32, i32) -> (i32, i32, i32) {

        %fizzbuzz1 = tfrt.add.i32 %fizzbuzz0, %one
        tfrt.return %fizzbuzz1, %buzz0, %fizz0 : i32, i32, i32

    } else {

      %quot, %rem = tfrt.div.i32 %i1, %three
      %isfizz = tfrt.equal.i32 %rem, %zero

      %buzz2, %fizz3 = tfrt.if
          %isfizz, %i1, %buzz0, %fizz0, %zero, %one, %two :
          (i32, i32, i32, i32, i32, i32) -> (i32, i32) {

        %buzz1 = tfrt.add.i32 %buzz0, %one
        tfrt.return %buzz1, %fizz0 : i32, i32

      } else {

        %quot, %rem = tfrt.div.i32 %i1, %two
        %isbuzz = tfrt.equal.i32 %rem, %zero

        %fizz2 = tfrt.if
            %isbuzz, %i1, %fizz0, %one :
            (i32, i32, i32) -> (i32) {
          %fizz1 = tfrt.add.i32 %fizz0, %one
          tfrt.return %fizz1 : i32
        } else {
          tfrt.return %fizz0 : i32
        }

        tfrt.return %buzz0, %fizz2 : i32, i32
      }

      tfrt.return %fizzbuzz0, %buzz2, %fizz3 : i32, i32, i32
    }

    tfrt.return %i1, %fizzbuzz2, %buzz3, %fizz4, %zero, %one, %two, %three, %six :
        i32, i32, i32, i32, i32, i32, i32, i32, i32
  }

  tfrt.return %fizzbuzz3, %buzz4, %fizz5 : i32, i32, i32
}


// CHECK-LABEL: --- Running 'fizzbuzz'
func @fizzbuzz() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %n = tfrt.constant.i32 1000

  %fizzbuzz0, %fizz0, %buzz0 =
      tfrt.call @fizzbuzz_compute(%n) : (i32) -> (i32, i32, i32)

  // 1000 // 6
  // CHECK: int32 = 166
  %ch1 = tfrt.print.i32 %fizzbuzz0, %ch0
  // 1000 // 3 - 1000 // 6
  // CHECK: int32 = 167
  %ch2 = tfrt.print.i32 %fizz0, %ch1
  // 1000 // 2 - 1000 // 6
  // CHECK: int32 = 334
  %ch3 = tfrt.print.i32 %buzz0, %ch2

  tfrt.return %ch3 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'bm_fizzbuzz'
func @bm_fizzbuzz() {
  %n = tfrt.constant.i32 1000

  tfrt_test.benchmark "bm_fizzbuzz"(%n : i32)
      duration_secs = 4, max_count = 10000, num_warmup_runs = 10 {
      %fizzbuzz0, %fizz0, %buzz0 =
          tfrt.call @fizzbuzz_compute(%n) : (i32) -> (i32, i32, i32)
      tfrt.return %n : i32
  }

  tfrt.return
}
