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

// RUN: cat %s.bef > /dev/null

func @trivial() {
  "simple.kernel"() : () -> ()
  tfrt.return
}

func @args_results(%a: i32, %b: f32) -> (i8, i32) {
  %c = "simple.kernel"(%a, %b) : (i32, f32) -> i8
  tfrt.return %c, %a : i8, i32
}
