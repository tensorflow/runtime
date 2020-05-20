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

// CHECK-LABEL: --- Running 'simple_test'
func @simple_test() -> !hex.chain {
  %ch0 = hex.new.chain

  %attrs_a = "corert.create_op_attrs"() : () -> !corert.opattrs

  // CHECK: OpAttrs is empty
  %ch1 = "tfrt_test.corert.op_attrs_print"(%attrs_a, %ch0)
    : (!corert.opattrs, !hex.chain) -> !hex.chain

  %ch2 = "corert.op_attrs_set_array.i32"(%attrs_a, %ch1) {key="shape", value= [1 : i32, 65536: i32]}
    : (!corert.opattrs, !hex.chain) -> (!hex.chain)

  // CHECK: OpAttrs contains 1 entries:
  // CHECK-NEXT: 'shape' type=I32 value=[1, 65536]
  %ch3 = "tfrt_test.corert.op_attrs_print"(%attrs_a, %ch2)
    : (!corert.opattrs, !hex.chain) -> !hex.chain

  %ch4 = "corert.op_attrs_set.f32"(%attrs_a, %ch3) {key="value", value= 1.0 : f32}
    : (!corert.opattrs, !hex.chain) -> (!hex.chain)

  %ch5 = "corert.op_attrs_set.dtype"(%attrs_a, %ch4) {key="type", value=f32}
    : (!corert.opattrs, !hex.chain) -> (!hex.chain)

  %ch6 = "corert.op_attrs_set.dense"(%attrs_a, %ch5) {key="dense", value=dense<[[1,2],[3,4]]> : tensor<2x2xi32>}
    : (!corert.opattrs, !hex.chain) -> (!hex.chain)

  // CHECK: OpAttrs contains 4 entries:
  // CHECK-NEXT: 'dense' type=DENSE value=dtype=I32, rank=2, elt_count=4
  // CHECK-NEXT: 'shape' type=I32 value=[1, 65536]
  // CHECK-NEXT: 'type' type=DTYPE value=F32
  // CHECK-NEXT: 'value' type=F32 value=1.0
  %ch7 = "tfrt_test.corert.op_attrs_print"(%attrs_a, %ch6)
    : (!corert.opattrs, !hex.chain) -> (!hex.chain)

  hex.return %ch7 : !hex.chain
}

// CHECK-LABEL: --- Running 'bool_test'
func @bool_test() -> !hex.chain {
  %ch0 = hex.new.chain

  %attrs_a = "corert.create_op_attrs"() : () -> !corert.opattrs

  %ch1 = "corert.op_attrs_set.bool"(%attrs_a, %ch0) {key="value", value=false}
    : (!corert.opattrs, !hex.chain) -> (!hex.chain)

  // CHECK: OpAttrs contains 1 entries:
  // CHECK-NEXT: 'value' type=BOOL value=0
  %ch2 = "tfrt_test.corert.op_attrs_print"(%attrs_a, %ch1)
    : (!corert.opattrs, !hex.chain) -> !hex.chain

  hex.return %ch2 : !hex.chain
}

// CHECK-LABEL: --- Running 'lots_of_attrs'
func @lots_of_attrs() -> !hex.chain {
  %ch0 = hex.new.chain

  %attrs_a = "corert.create_op_attrs"() : () -> !corert.opattrs

  %ch1 = "corert.op_attrs_set.i32"(%attrs_a, %ch0) {key="a", value = 1: i32}
    : (!corert.opattrs, !hex.chain) -> (!hex.chain)

  %ch2 = "corert.op_attrs_set.i32"(%attrs_a, %ch1) {key="b", value = 2: i32}
    : (!corert.opattrs, !hex.chain) -> (!hex.chain)

  %ch3 = "corert.op_attrs_set.i32"(%attrs_a, %ch2) {key="c", value = 3: i32}
    : (!corert.opattrs, !hex.chain) -> (!hex.chain)

  %ch4 = "corert.op_attrs_set.i32"(%attrs_a, %ch3) {key="d", value = 4: i32}
  : (!corert.opattrs, !hex.chain) -> (!hex.chain)

  %ch5 = "corert.op_attrs_set.i32"(%attrs_a, %ch4) {key="e", value = 5: i32}
  : (!corert.opattrs, !hex.chain) -> (!hex.chain)

  %ch6 = "corert.op_attrs_set.i32"(%attrs_a, %ch5) {key="f", value = 6: i32}
  : (!corert.opattrs, !hex.chain) -> (!hex.chain)

  %ch7 = "corert.op_attrs_set.i32"(%attrs_a, %ch6) {key="g", value = 7: i32}
  : (!corert.opattrs, !hex.chain) -> (!hex.chain)

  %ch8 = "corert.op_attrs_set.i32"(%attrs_a, %ch7) {key="h", value = 8: i32}
  : (!corert.opattrs, !hex.chain) -> (!hex.chain)

  %ch9 = "corert.op_attrs_set.i32"(%attrs_a, %ch8) {key="i", value = 9: i32}
  : (!corert.opattrs, !hex.chain) -> (!hex.chain)

  %ch10 = "corert.op_attrs_set.i32"(%attrs_a, %ch9) {key="j", value = 10: i32}
  : (!corert.opattrs, !hex.chain) -> (!hex.chain)

  // CHECK: OpAttrs contains 10 entries:
  // CHECK-NEXT: 'a' type=I32 value=1
  // CHECK-NEXT: 'b' type=I32 value=2
  // CHECK-NEXT: 'c' type=I32 value=3
  // CHECK-NEXT: 'd' type=I32 value=4
  // CHECK-NEXT: 'e' type=I32 value=5
  // CHECK-NEXT: 'f' type=I32 value=6
  // CHECK-NEXT: 'g' type=I32 value=7
  // CHECK-NEXT: 'h' type=I32 value=8
  // CHECK-NEXT: 'i' type=I32 value=9
  // CHECK-NEXT: 'j' type=I32 value=10
  %ch13 = "tfrt_test.corert.op_attrs_print"(%attrs_a, %ch10) : (!corert.opattrs, !hex.chain) -> !hex.chain

  hex.return %ch13 : !hex.chain
}

// CHECK-LABEL: --- Running 'freezing'
func @freezing() -> !hex.chain {
  %ch0 = hex.new.chain

  %attrs_a = "corert.create_op_attrs"() : () -> !corert.opattrs

  %ch1 = "corert.op_attrs_set.i32"(%attrs_a, %ch0) {key="a", value = 1: i32}
    : (!corert.opattrs, !hex.chain) -> (!hex.chain)

  %ch2 = "corert.op_attrs_set.i32"(%attrs_a, %ch1) {key="b", value = 2: i32}
    : (!corert.opattrs, !hex.chain) -> (!hex.chain)

  %ch3 = "corert.op_attrs_set.i32"(%attrs_a, %ch2) {key="c", value = 3: i32}
    : (!corert.opattrs, !hex.chain) -> (!hex.chain)

  %ch4 = "corert.op_attrs_set.i32"(%attrs_a, %ch3) {key="d", value = 4: i32}
  : (!corert.opattrs, !hex.chain) -> (!hex.chain)

  %ch5 = "corert.op_attrs_set.i32"(%attrs_a, %ch4) {key="e", value = 5: i32}
  : (!corert.opattrs, !hex.chain) -> (!hex.chain)

  %attr_ref, %ch5a = "tfrt_test.corert.op_attrs_freeze"(%attrs_a, %ch5)
    : (!corert.opattrs, !hex.chain) -> (!corert.opattrs_ref, !hex.chain)

  // CHECK: OpAttrs contains 5 entries:
  // CHECK-NEXT: 'a' type=I32 value=1
  // CHECK-NEXT: 'b' type=I32 value=2
  // CHECK-NEXT: 'c' type=I32 value=3
  // CHECK-NEXT: 'd' type=I32 value=4
  // CHECK-NEXT: 'e' type=I32 value=5
  %ch5b = "tfrt_test.corert.op_attrs_ref_print"(%attr_ref, %ch5a)
    : (!corert.opattrs_ref, !hex.chain) -> !hex.chain

  %ch6 = "corert.op_attrs_set.i32"(%attrs_a, %ch5b) {key="f", value = 6: i32}
  : (!corert.opattrs, !hex.chain) -> (!hex.chain)

  %ch7 = "corert.op_attrs_set_array.i32"(%attrs_a, %ch6) {key="g", value= [1 : i32, 1: i32]}
  : (!corert.opattrs, !hex.chain) -> (!hex.chain)

  %ch8 = "corert.op_attrs_set_array.i32"(%attrs_a, %ch7) {key="h", value= [1 : i32, 2: i32, 3: i32, 4: i32, 5: i32, 6: i32, 7: i32, 8: i32]}
  : (!corert.opattrs, !hex.chain) -> (!hex.chain)

  %ch9 = "corert.op_attrs_set.i32"(%attrs_a, %ch8) {key="i", value = 9: i32}
  : (!corert.opattrs, !hex.chain) -> (!hex.chain)

  %ch10 = "corert.op_attrs_set.i32"(%attrs_a, %ch9) {key="j", value = 10: i32}
  : (!corert.opattrs, !hex.chain) -> (!hex.chain)


  %attr_ref2, %ch11 = "tfrt_test.corert.op_attrs_freeze"(%attrs_a, %ch10)
    : (!corert.opattrs, !hex.chain) -> (!corert.opattrs_ref, !hex.chain)

  // CHECK: OpAttrs contains 10 entries:
  // CHECK-NEXT: 'a' type=I32 value=1
  // CHECK-NEXT: 'b' type=I32 value=2
  // CHECK-NEXT: 'c' type=I32 value=3
  // CHECK-NEXT: 'd' type=I32 value=4
  // CHECK-NEXT: 'e' type=I32 value=5
  // CHECK-NEXT: 'f' type=I32 value=6
  // CHECK-NEXT: 'g' type=I32 value=[1, 1]
  // CHECK-NEXT: 'h' type=I32 value=[1, 2, 3, 4, 5...]
  // CHECK-NEXT: 'i' type=I32 value=9
  // CHECK-NEXT: 'j' type=I32 value=10
  %ch12 = "tfrt_test.corert.op_attrs_ref_print"(%attr_ref2, %ch11)
    : (!corert.opattrs_ref, !hex.chain) -> !hex.chain

  // CHECK: OpAttrs contains 10 entries:
  // CHECK-NEXT: 'a' type=I32 value=1
  // CHECK-NEXT: 'b' type=I32 value=2
  // CHECK-NEXT: 'c' type=I32 value=3
  // CHECK-NEXT: 'd' type=I32 value=4
  // CHECK-NEXT: 'e' type=I32 value=5
  // CHECK-NEXT: 'f' type=I32 value=6
  // CHECK-NEXT: 'g' type=I32 value=[1, 1]
  // CHECK-NEXT: 'h' type=I32 value=[1, 2, 3, 4, 5...]
  // CHECK-NEXT: 'i' type=I32 value=9
  // CHECK-NEXT: 'j' type=I32 value=10
  %ch13 = "tfrt_test.corert.op_attrs_print"(%attrs_a, %ch12) : (!corert.opattrs, !hex.chain) -> !hex.chain

  hex.return %ch13 : !hex.chain
}

// CHECK-LABEL: --- Running 'aggregate_attr_test'
func @aggregate_attr_test() -> !hex.chain {
  %ch0 = hex.new.chain

  %attrs = "corert.create_op_attrs"() : () -> !corert.opattrs

  %ch1 = "corert.op_attrs_set.aggregate"(%attrs, %ch0)
    {key="aggregate", value=[dense<[[1,2],[3,4]]> : tensor<2x2xi32>, dense<1> : tensor<i64>]}
    : (!corert.opattrs, !hex.chain) -> (!hex.chain)

  // CHECK: OpAttrs contains 1 entries:
  // CHECK-NEXT: 'aggregate' type=AGGREGATE value=elt_count=2 [{dtype=I32, rank=2, elt_count=4}, {dtype=I64, rank=0, elt_count=1}]
  %ch2 = "tfrt_test.corert.op_attrs_print"(%attrs, %ch1)
    : (!corert.opattrs, !hex.chain) -> (!hex.chain)

  hex.return %ch2 : !hex.chain
}

// CHECK-LABEL: --- Running 'shape_attr_test'
func @shape_attr_test() -> !hex.chain {
  %ch0 = hex.new.chain

  %attrs = "corert.create_op_attrs"() : () -> !corert.opattrs

  %ch1 = "corert.op_attrs_set.shape"(%attrs, %ch0)
    {key="shape", value=#corert.shape<1x2x3>}
    : (!corert.opattrs, !hex.chain) -> (!hex.chain)

  // CHECK: OpAttrs contains 1 entries:
  // CHECK-NEXT: 'shape' type=SHAPE value=<1x2x3>
  %ch2 = "tfrt_test.corert.op_attrs_print"(%attrs, %ch1)
    : (!corert.opattrs, !hex.chain) -> (!hex.chain)

  hex.return %ch2 : !hex.chain
}

// CHECK-LABEL: --- Running 'string_type_attr_test'
func @string_type_attr_test() -> !hex.chain {
  %ch0 = hex.new.chain

  %attrs = "corert.create_op_attrs"() : () -> !corert.opattrs

  %ch1 = "corert.op_attrs_set.dtype"(%attrs, %ch0)
    {key="type", value=!corert.string}
    : (!corert.opattrs, !hex.chain) -> (!hex.chain)

  // CHECK: OpAttrs contains 1 entries:
  // CHECK-NEXT: 'type' type=DTYPE value=CHAR
  %ch2 = "tfrt_test.corert.op_attrs_print"(%attrs, %ch1)
    : (!corert.opattrs, !hex.chain) -> (!hex.chain)

  hex.return %ch2 : !hex.chain
}
