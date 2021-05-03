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

// CHECK-LABEL: --- Running 'simple_test'
func @simple_test() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %attrs_a = "corert.create_op_attrs"() : () -> !corert.opattrs

  // CHECK: OpAttrs is empty
  %ch1 = "tfrt_test.corert.op_attrs_print"(%attrs_a, %ch0)
    : (!corert.opattrs, !tfrt.chain) -> !tfrt.chain

  %ch2 = "corert.op_attrs_set_array.i32"(%attrs_a, %ch1) {key="shape", value= [1 : i32, 65536: i32]}
    : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)

  // CHECK: OpAttrs contains 1 entries:
  // CHECK-NEXT: 'shape' type=I32 value=[1, 65536]
  %ch3 = "tfrt_test.corert.op_attrs_print"(%attrs_a, %ch2)
    : (!corert.opattrs, !tfrt.chain) -> !tfrt.chain

  %ch4 = "corert.op_attrs_set.f32"(%attrs_a, %ch3) {key="value", value= 1.0 : f32}
    : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)

  %ch5 = "corert.op_attrs_set.dtype"(%attrs_a, %ch4) {key="type", value=f32}
    : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)

  %ch6 = "corert.op_attrs_set.dense"(%attrs_a, %ch5) {key="dense", value=dense<[[1,2],[3,4]]> : tensor<2x2xi32>}
    : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)

  // CHECK: OpAttrs contains 4 entries:
  // CHECK-NEXT: 'dense' type=DENSE value=dtype=I32, rank=2, elt_count=4
  // CHECK-NEXT: 'shape' type=I32 value=[1, 65536]
  // CHECK-NEXT: 'type' type=DTYPE value=F32
  // CHECK-NEXT: 'value' type=F32 value=1.0
  %ch7 = "tfrt_test.corert.op_attrs_print"(%attrs_a, %ch6)
    : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)

  tfrt.return %ch7 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'bool_test'
func @bool_test() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %attrs_a = "corert.create_op_attrs"() : () -> !corert.opattrs

  %ch1 = "corert.op_attrs_set.bool"(%attrs_a, %ch0) {key="value", value=false}
    : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)

  // CHECK: OpAttrs contains 1 entries:
  // CHECK-NEXT: 'value' type=BOOL value=0
  %ch2 = "tfrt_test.corert.op_attrs_print"(%attrs_a, %ch1)
    : (!corert.opattrs, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch2 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'lots_of_attrs'
func @lots_of_attrs() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %attrs_a = "corert.create_op_attrs"() : () -> !corert.opattrs

  %ch1 = "corert.op_attrs_set.i32"(%attrs_a, %ch0) {key="a", value = 1: i32}
    : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)

  %ch2 = "corert.op_attrs_set.i32"(%attrs_a, %ch1) {key="b", value = 2: i32}
    : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)

  %ch3 = "corert.op_attrs_set.i32"(%attrs_a, %ch2) {key="c", value = 3: i32}
    : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)

  %ch4 = "corert.op_attrs_set.i32"(%attrs_a, %ch3) {key="d", value = 4: i32}
  : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)

  %ch5 = "corert.op_attrs_set.i32"(%attrs_a, %ch4) {key="e", value = 5: i32}
  : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)

  %ch6 = "corert.op_attrs_set.i32"(%attrs_a, %ch5) {key="f", value = 6: i32}
  : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)

  %ch7 = "corert.op_attrs_set.i32"(%attrs_a, %ch6) {key="g", value = 7: i32}
  : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)

  %ch8 = "corert.op_attrs_set.i32"(%attrs_a, %ch7) {key="h", value = 8: i32}
  : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)

  %ch9 = "corert.op_attrs_set.i32"(%attrs_a, %ch8) {key="i", value = 9: i32}
  : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)

  %ch10 = "corert.op_attrs_set.i32"(%attrs_a, %ch9) {key="j", value = 10: i32}
  : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)

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
  %ch13 = "tfrt_test.corert.op_attrs_print"(%attrs_a, %ch10) : (!corert.opattrs, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch13 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'freezing'
func @freezing() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %attrs_a = "corert.create_op_attrs"() : () -> !corert.opattrs

  %ch1 = "corert.op_attrs_set.i32"(%attrs_a, %ch0) {key="a", value = 1: i32}
    : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)

  %ch2 = "corert.op_attrs_set.i32"(%attrs_a, %ch1) {key="b", value = 2: i32}
    : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)

  %ch3 = "corert.op_attrs_set.i32"(%attrs_a, %ch2) {key="c", value = 3: i32}
    : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)

  %ch4 = "corert.op_attrs_set.i32"(%attrs_a, %ch3) {key="d", value = 4: i32}
  : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)

  %ch5 = "corert.op_attrs_set.i32"(%attrs_a, %ch4) {key="e", value = 5: i32}
  : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)

  %attr_ref, %ch5a = "tfrt_test.corert.op_attrs_freeze"(%attrs_a, %ch5)
    : (!corert.opattrs, !tfrt.chain) -> (!corert.opattrs_ref, !tfrt.chain)

  // CHECK: OpAttrs contains 5 entries:
  // CHECK-NEXT: 'a' type=I32 value=1
  // CHECK-NEXT: 'b' type=I32 value=2
  // CHECK-NEXT: 'c' type=I32 value=3
  // CHECK-NEXT: 'd' type=I32 value=4
  // CHECK-NEXT: 'e' type=I32 value=5
  %ch5b = "tfrt_test.corert.op_attrs_ref_print"(%attr_ref, %ch5a)
    : (!corert.opattrs_ref, !tfrt.chain) -> !tfrt.chain

  %ch6 = "corert.op_attrs_set.i32"(%attrs_a, %ch5b) {key="f", value = 6: i32}
  : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)

  %ch7 = "corert.op_attrs_set_array.i32"(%attrs_a, %ch6) {key="g", value= [1 : i32, 1: i32]}
  : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)

  %ch8 = "corert.op_attrs_set_array.i32"(%attrs_a, %ch7) {key="h", value= [1 : i32, 2: i32, 3: i32, 4: i32, 5: i32, 6: i32, 7: i32, 8: i32]}
  : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)

  %ch9 = "corert.op_attrs_set.i32"(%attrs_a, %ch8) {key="i", value = 9: i32}
  : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)

  %ch10 = "corert.op_attrs_set.i32"(%attrs_a, %ch9) {key="j", value = 10: i32}
  : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)


  %attr_ref2, %ch11 = "tfrt_test.corert.op_attrs_freeze"(%attrs_a, %ch10)
    : (!corert.opattrs, !tfrt.chain) -> (!corert.opattrs_ref, !tfrt.chain)

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
    : (!corert.opattrs_ref, !tfrt.chain) -> !tfrt.chain

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
  %ch13 = "tfrt_test.corert.op_attrs_print"(%attrs_a, %ch12) : (!corert.opattrs, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch13 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'aggregate_attr_test'
func @aggregate_attr_test() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %attrs = "corert.create_op_attrs"() : () -> !corert.opattrs

  %ch1 = "corert.op_attrs_set.aggregate"(%attrs, %ch0)
    {key="aggregate", value=[dense<[[1,2],[3,4]]> : tensor<2x2xi32>, dense<1> : tensor<i64>]}
    : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)

  // CHECK: OpAttrs contains 1 entries:
  // CHECK-NEXT: 'aggregate' type=AGGREGATE value=elt_count=2 [{dtype=I32, rank=2, elt_count=4}, {dtype=I64, rank=0, elt_count=1}]
  %ch2 = "tfrt_test.corert.op_attrs_print"(%attrs, %ch1)
    : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)

  tfrt.return %ch2 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'shape_attr_test'
func @shape_attr_test() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %attrs = "corert.create_op_attrs"() : () -> !corert.opattrs

  %ch1 = "corert.op_attrs_set.shape"(%attrs, %ch0)
    {key="shape", value=#corert.shape<1x2x3>}
    : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)

  // CHECK: OpAttrs contains 1 entries:
  // CHECK-NEXT: 'shape' type=SHAPE value=<1x2x3>
  %ch2 = "tfrt_test.corert.op_attrs_print"(%attrs, %ch1)
    : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)

  tfrt.return %ch2 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'type_attr_test'
func @type_attr_test() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %attrs = "corert.create_op_attrs"() : () -> !corert.opattrs

  %ch1 = "corert.op_attrs_set.dtype"(%attrs, %ch0)
    {key="f16", value=f16}
    : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)
  %ch2 = "corert.op_attrs_set.dtype"(%attrs, %ch1)
    {key="f32", value=f32}
    : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)
  %ch3 = "corert.op_attrs_set.dtype"(%attrs, %ch2)
    {key="f64", value=f64}
    : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)
  %ch4 = "corert.op_attrs_set.dtype"(%attrs, %ch3)
    {key="i8", value=i8}
    : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)
  %ch5 = "corert.op_attrs_set.dtype"(%attrs, %ch4)
    {key="i16", value=i16}
    : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)
  %ch6 = "corert.op_attrs_set.dtype"(%attrs, %ch5)
    {key="i32", value=i32}
    : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)
  %ch7 = "corert.op_attrs_set.dtype"(%attrs, %ch6)
    {key="i64", value=i64}
    : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)
  %ch8 = "corert.op_attrs_set.dtype"(%attrs, %ch7)
    {key="string", value=!corert.string}
    : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)
  %ch9 = "corert.op_attrs_set.dtype"(%attrs, %ch8)
    {key="ui8", value=ui8}
    : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)
  %ch10 = "corert.op_attrs_set.dtype"(%attrs, %ch9)
    {key="qui8", value=!corert.quint8}
    : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)
  %ch11 = "corert.op_attrs_set.dtype"(%attrs, %ch10)
    {key="qui16", value=!corert.quint16}
    : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)
  %ch12 = "corert.op_attrs_set.dtype"(%attrs, %ch11)
    {key="qi8", value=!corert.qint8}
    : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)
  %ch13 = "corert.op_attrs_set.dtype"(%attrs, %ch12)
    {key="qi16", value=!corert.qint16}
    : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)
  %ch14 = "corert.op_attrs_set.dtype"(%attrs, %ch13)
    {key="qi32", value=!corert.qint32}
    : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)

  // CHECK: OpAttrs contains 14 entries:
  // CHECK-NEXT: 'f16' type=DTYPE value=F16
  // CHECK-NEXT: 'f32' type=DTYPE value=F32
  // CHECK-NEXT: 'f64' type=DTYPE value=F64
  // CHECK-NEXT: 'i16' type=DTYPE value=I16
  // CHECK-NEXT: 'i32' type=DTYPE value=I32
  // CHECK-NEXT: 'i64' type=DTYPE value=I64
  // CHECK-NEXT: 'i8' type=DTYPE value=I8
  // CHECK-NEXT: 'qi16' type=DTYPE value=QI16
  // CHECK-NEXT: 'qi32' type=DTYPE value=QI32
  // CHECK-NEXT: 'qi8' type=DTYPE value=QI8
  // CHECK-NEXT: 'qui16' type=DTYPE value=QUI16
  // CHECK-NEXT: 'qui8' type=DTYPE value=QUI8
  // CHECK-NEXT: 'string' type=DTYPE value=CHAR
  // CHECK-NEXT: 'ui8' type=DTYPE value=UI8
  %ch15 = "tfrt_test.corert.op_attrs_print"(%attrs, %ch14)
    : (!corert.opattrs, !tfrt.chain) -> (!tfrt.chain)

  tfrt.return %ch15 : !tfrt.chain
}
