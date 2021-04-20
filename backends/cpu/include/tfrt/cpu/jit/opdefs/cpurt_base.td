// Copyright 2021 The TensorFlow Runtime Authors
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

//===- cpurt_base.td ------------------------------------------------------===//
//
// Dialect and type definitions for CPU Runtime dialect.
//
//===----------------------------------------------------------------------===//

#ifdef CPURT_BASE
#else
#define CPURT_BASE

//===----------------------------------------------------------------------===//
// TFRT CPU Runtime dialect definitions
//===----------------------------------------------------------------------===//

def TFRT_CPURT_Dialect : Dialect {
  let name = "cpurt";

  let description = [{
    The CPU runtime dialect.

    This dialect contains common CPU runtime operations.
  }];

  let cppNamespace = "::tfrt::cpu::jit";
}

def JitExecutableType : OpaqueType<"cpurt", "jit_executable",
                                   "!cpurt.jit_executable type">;


#endif // CPURT_BASE
