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

// RUN: bef_executor $(bef_name %s) 2>&1 | FileCheck %s --dump-input=fail

// CHECK: --- Running 'basic_f64'
func @basic_f64() {
  %ch0 = tfrt.new.chain
  %value_1 = tfrt.constant.f64 156.0
  %value_2 = tfrt.constant.f64 100.0

  %value_add = tfrt.add.f64 %value_1, %value_2
  // %value_div = tfrt.div.f64 %value_1, %value_2
  %value_div = "tfrt.div.f64"(%value_1, %value_2) : (f64, f64) -> f64
  %value_min = "tfrt.minimum.f64"(%value_1, %value_2) : (f64, f64) -> f64
  %value_mul = "tfrt.multiply.f64"(%value_1, %value_2) : (f64, f64) -> f64

  // CHECK-NEXT: f64 = 256.000000
  %ch1 = tfrt.print.f64 %value_add, %ch0
  // CHECK-NEXT: f64 = 1.560000
  %ch2 = tfrt.print.f64 %value_div, %ch1
  // CHECK-NEXT: f64 = 100.000000
  %ch3 = tfrt.print.f64 %value_min, %ch2
  // CHECK-NEXT: f64 = 15600.000000
  %ch4 = tfrt.print.f64 %value_mul, %ch3

  tfrt.return
}

// CHECK: --- Running 'basic_f32'
func @basic_f32() {
  %ch0 = tfrt.new.chain
  %value_1 = tfrt.constant.f32 156.0
  %value_2 = tfrt.constant.f32 100.0

  %value_add = tfrt.add.f32 %value_1, %value_2
  // %value_div = tfrt.div.f32 %value_1, %value_2
  %value_div = "tfrt.div.f32"(%value_1, %value_2) : (f32, f32) -> f32
  %value_min = "tfrt.minimum.f32"(%value_1, %value_2) : (f32, f32) -> f32
  %value_mul = "tfrt.multiply.f32"(%value_1, %value_2) : (f32, f32) -> f32

  // CHECK-NEXT: f32 = 256.000000
  %ch1 = tfrt.print.f32 %value_add, %ch0
  // CHECK-NEXT: f32 = 1.560000
  %ch2 = tfrt.print.f32 %value_div, %ch1
  // CHECK-NEXT: f32 = 100.000000
  %ch3 = tfrt.print.f32 %value_min, %ch2
  // CHECK-NEXT: f32 = 15600.000000
  %ch4 = tfrt.print.f32 %value_mul, %ch3

  tfrt.return
}

// CHECK: --- Running 'cast_between_i64_f32'
func @cast_between_i64_f32() {
  %ch0 = tfrt.new.chain
  %resize_min = tfrt.constant.f32 256.0
  %height = tfrt.constant.i64 737
  %width = tfrt.constant.i64 899

  %height_float = "tfrt.cast.i64_to_f32"(%height) : (i64) -> f32
  %width_float = "tfrt.cast.i64_to_f32"(%width) : (i64) -> f32
  %smaller_dim = "tfrt.minimum.f32"(%height_float, %width_float) : (f32, f32) -> f32
  %scale_ratio = "tfrt.div.f32"(%resize_min, %smaller_dim) : (f32, f32) -> f32
  %new_height_float = "tfrt.multiply.f32"(%height_float, %scale_ratio) : (f32, f32) -> f32
  %new_width_float = "tfrt.multiply.f32"(%width_float, %scale_ratio) : (f32, f32) -> f32
  %new_height = "tfrt.cast.f32_to_i64"(%new_height_float) : (f32) -> i64
  %new_width = "tfrt.cast.f32_to_i64"(%new_width_float) : (f32) -> i64

  // CHECK-NEXT: f32 = 737.000000
  %ch1 = tfrt.print.f32 %height_float, %ch0
  // CHECK-NEXT: f32 = 899.000000
  %ch2 = tfrt.print.f32 %width_float, %ch1
  // CHECK-NEXT: f32 = 737.000000
  %ch3 = tfrt.print.f32 %smaller_dim, %ch2
  // CHECK-NEXT: f32 = 0.347354
  %ch4 = tfrt.print.f32 %scale_ratio, %ch3
  // CHECK-NEXT: f32 = 256.000000
  %ch5 = tfrt.print.f32 %new_height_float, %ch4
  // CHECK-NEXT: f32 = 312.271362
  %ch6 = tfrt.print.f32 %new_width_float, %ch5
  // CHECK-NEXT: int64 = 256
  %ch7 = tfrt.print.i64 %new_height, %ch6
  // CHECK-NEXT: int64 = 312
  %ch8 = tfrt.print.i64 %new_width, %ch7

  tfrt.return
}

// CHECK: --- Running 'cast_between_i64_f64'
func @cast_between_i64_f64() {
  %ch0 = tfrt.new.chain
  %height = tfrt.constant.i64 737
  %width = tfrt.constant.i64 899
  %resize_min = tfrt.constant.f64 256.0

  %height_float = "tfrt.cast.i64_to_f64"(%height) : (i64) -> f64
  %width_float = "tfrt.cast.i64_to_f64"(%width) : (i64) -> f64
  %smaller_dim = "tfrt.minimum.f64"(%height_float, %width_float) : (f64, f64) -> f64
  %scale_ratio = "tfrt.div.f64"(%resize_min, %smaller_dim) : (f64, f64) -> f64
  %new_height_float = "tfrt.multiply.f64"(%height_float, %scale_ratio) : (f64, f64) -> f64
  %new_width_float = "tfrt.multiply.f64"(%width_float, %scale_ratio) : (f64, f64) -> f64
  %new_height = "tfrt.cast.f64_to_i64"(%new_height_float) : (f64) -> i64
  %new_width = "tfrt.cast.f64_to_i64"(%new_width_float) : (f64) -> i64

  // CHECK-NEXT: f64 = 737.000000
  %ch1 = tfrt.print.f64 %height_float, %ch0
  // CHECK-NEXT: f64 = 899.000000
  %ch2 = tfrt.print.f64 %width_float, %ch1
  // CHECK-NEXT: f64 = 737.000000
  %ch3 = tfrt.print.f64 %smaller_dim, %ch2
  // CHECK-NEXT: f64 = 0.347354
  %ch4 = tfrt.print.f64 %scale_ratio, %ch3
  // CHECK-NEXT: f64 = 256.000000
  %ch5 = tfrt.print.f64 %new_height_float, %ch4
  // CHECK-NEXT: f64 = 312.271370
  %ch6 = tfrt.print.f64 %new_width_float, %ch5
  // CHECK-NEXT: int64 = 255
  %ch7 = tfrt.print.i64 %new_height, %ch6
  // CHECK-NEXT: int64 = 312
  %ch8 = tfrt.print.i64 %new_width, %ch7

  tfrt.return
}
