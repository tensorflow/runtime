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

//===- jitrt_base.td ------------------------------------------------------===//
//
// Dialect and type definitions for JIT Runtime dialect.
//
//===----------------------------------------------------------------------===//

#ifdef JITRT_BASE
#else
#define JITRT_BASE

//===----------------------------------------------------------------------===//
// TFRT JIT Runtime dialect definitions
//===----------------------------------------------------------------------===//

def JitRuntimeDialect : Dialect {
  let name = "jitrt";

  let description = [{
    The CPU runtime dialect.

    This dialect contains common CPU runtime operations.
  }];

  let cppNamespace = "::tfrt::jitrt";
}

def JitExecutableType : OpaqueType<"jitrt", "jit_executable",
                                   "!jitrt.jit_executable type">;


#endif // JITRT_BASE
