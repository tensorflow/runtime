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

//===- tensor_shape_base.td -----------------------------------------------===//
//
// Base type definitions for tensor shape ops.
//
//===----------------------------------------------------------------------===//

#ifdef TS_OPS_BASE
#else
#define TS_OPS_BASE

// Type definition.
def TS_Shape : OpaqueType<"ts", "shape", "!ts.shape type">;

#endif  // TS_OPS_BASE
