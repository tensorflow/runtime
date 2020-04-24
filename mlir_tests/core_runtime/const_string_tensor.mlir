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

// RUN: tfrt_translate -mlir-to-bef %s | bef_executor | FileCheck %s --dump-input=fail

// CHECK-LABEL: --- Running 'basic'
func @basic() {
  %ch0 = hex.new.chain

  %a = corert.const_string_tensor {shape = [2], value = ["string", "tensor"]}

  // CHECK: shape = [2], values = ["string", "tensor"]
  %ch5 = "corert.print_tensorhandle"(%a, %ch0) : (!corert.tensorhandle, !hex.chain) -> !hex.chain

  hex.return
}
