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
// RUN: tfrt_opt %s | tfrt_opt

// CHECK-LABEL: --- Running 'event_create_test'
func @event_create_test() {
  %ch1 = tfrt.new.chain
  %ch2 = cuda.init %ch1
  %index = tfrt.constant.i32 0
  %device, %ch3 = cuda.device.get %index, %ch2
  %context, %ch4 = cuda_test.context.get %device, %ch2
  %event, %ch5 = cuda.event.create %context, %ch2

  tfrt.return
}

// CHECK-LABEL: --- Running 'event_record_and_poll_test'
func @event_record_and_poll_test() {
  %ch1 = tfrt.new.chain
  %ch2 = cuda.init %ch1
  %index = tfrt.constant.i32 0
  %device, %ch3 = cuda.device.get %index, %ch2
  %context, %ch4 = cuda_test.context.get %device, %ch2
  %stream, %ch5 = cuda.stream.create %context, %ch2

  %event, %ch6 = cuda.event.create %context, %ch2
  %ch7 = cuda.event.record %event, %stream, %ch2
  %ch8 = cuda.event.poll %event, %ch7

  tfrt.return
}
