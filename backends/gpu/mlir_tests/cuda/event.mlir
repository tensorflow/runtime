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
// RUN: tfrt_gpu_opt %s | tfrt_gpu_opt

// CHECK-LABEL: --- Running 'event_create_test'
func @event_create_test() {
  %ch2 = tfrt.new.chain
  %index = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get %index, %ch2 { platform = 1 : i32 }
  %context = tfrt_gpu.context.create %device, %ch2
  %event = tfrt_gpu.event.create %context, %ch2

  tfrt.return
}

// CHECK-LABEL: --- Running 'event_record_and_poll_test'
func @event_record_and_poll_test() {
  %ch2 = tfrt.new.chain
  %index = tfrt.constant.i32 0
  %device = tfrt_gpu.device.get %index, %ch2 { platform = 1 : i32 }
  %context = tfrt_gpu.context.create %device, %ch2
  %stream = tfrt_gpu.stream.create %context, %ch2

  %event = tfrt_gpu.event.create %context, %ch2
  %ch7 = tfrt_gpu.event.record %event, %stream, %ch2
  %ch8 = tfrt_gpu.event.synchronize %event, %ch7

  tfrt.return
}
