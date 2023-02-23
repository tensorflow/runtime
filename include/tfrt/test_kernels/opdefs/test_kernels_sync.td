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

//===- test_kernels_sync.td ----------------------------------------------------===//
//
// Operation definitions for test_kernels_sync.
//
//===----------------------------------------------------------------------===//

#ifdef TEST_SYNC_OPS
#else
#define TEST_SYNC_OPS

include "tfrt/tfrt_op_base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// "tfrt_test_sync" dialect
def Test_Sync_Dialect : Dialect {
  let name = "tfrt_test_sync";

  let description = [{
    The synchronous Test dialect.

    This dialect contains common test operations.
  }];

  let cppNamespace = "::tfrt::test_sync";
}

#endif  // TEST_SYNC_OPS
