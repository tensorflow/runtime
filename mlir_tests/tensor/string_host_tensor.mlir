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

// RUN: bef_executor $(bef_name %s) | FileCheck %s --dump-input=fail

// CHECK-LABEL: --- Running 'basic'
func @basic() {
  %c0 = tfrt.new.chain

  %a = "tfrt_sht.create_tensor"()
    {shape = [2], values = ["string", "tensor"]} : () -> !t.tensor

  // CHECK: shape = [2], values = ["string", "tensor"]
  %c1 = tfrt_dht.print_tensor %a, %c0

  tfrt.return
}

func @sync_basic() attributes {tfrt.sync} {
  %a = "tfrt_sht_sync.create_tensor"()
    {shape = [2], values = ["string", "tensor"]} : () -> !t.tensor

  // CHECK: shape = [2], values = ["string", "tensor"]
  tfrt_dht_sync.print_tensor %a

  %b = "tfrt_sht_sync.create_uninitialized_tensor"()
    {shape = [2]} : () -> !t.tensor

  // CHECK: shape = [2], values = ["", ""]
  tfrt_dht_sync.print_tensor %b

  tfrt.return
}
