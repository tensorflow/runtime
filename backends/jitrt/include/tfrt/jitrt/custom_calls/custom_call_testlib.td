// Copyright 2022 The TensorFlow Runtime Authors
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

//===- custom_call_testlib.td ---------------------------------------------===//
//
// Dialect, attributes and structs definitions for testing custom calls.
//
//===----------------------------------------------------------------------===//

#ifndef CUSTOM_CALL_TESTLIB
#define CUSTOM_CALL_TESTLIB

include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// JitRt custom calls Testlib dialect definitions.
//===----------------------------------------------------------------------===//

def TestlibDialect : Dialect {
  let name = "testlib";

  let description = [{
    JitRt custom call test dialect.
  }];

  let cppNamespace = "::tfrt::jitrt";

  let emitAccessorPrefix = kEmitAccessorPrefix_Raw;
}

include "custom_call_testlib_attrs.td"
include "custom_call_testlib_enums.td"

#endif // CUSTOM_CALL_TESTLIB
