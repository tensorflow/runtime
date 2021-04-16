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

//===- cuda_ops_base.td ---------------------------------------------------===//
//
// GPU dialect and type definitions.
//
//===----------------------------------------------------------------------===//

#ifdef CUDA_OPS_BASE
#else
#define CUDA_OPS_BASE

// "tfrt_gpu" dialect
def GPU_Dialect : Dialect {
  let name = "tfrt_gpu";

  let description = [{
    The GPU dialect.

    This dialect contains common GPU operations for CUDA and ROCm.
  }];

  let cppNamespace = "tfrt::gpu";
}

// Base class for GPU dialect ops.
class GPU_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<GPU_Dialect, mnemonic, traits> {

  let assemblyFormat = "operands attr-dict";
}

// Type definitions.

class GPU_Type<string name> :
    OpaqueType<"tfrt_gpu", name, !strconcat("!tfrt_gpu.", name, " type")> {}

def GPU_AllocatorType : GPU_Type<"allocator">;
def GPU_BufferType : GPU_Type<"buffer">;
def GPU_ContextType : GPU_Type<"context">;
def GPU_DeviceType : GPU_Type<"device">;
def GPU_EventType : GPU_Type<"event">;
def GPU_FunctionType : GPU_Type<"function">;
def GPU_StreamType : GPU_Type<"stream">;

#endif  // CUDA_OPS_BASE
