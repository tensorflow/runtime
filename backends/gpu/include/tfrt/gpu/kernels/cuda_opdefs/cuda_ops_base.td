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
// CUDA type definitions.
//
//===----------------------------------------------------------------------===//

#ifdef CUDA_OPS_BASE
#else
#define CUDA_OPS_BASE

// Type definitions.

// Device is the physical CUDA device.
def DeviceType : OpaqueType<"cuda", "device", "!cuda.device type">;

// Context holds state that provides the context for executing kernels,
// allocating memory, and other CUDA operations.
// This Context is analogous to CUcontext.
def ContextType : OpaqueType<"cuda", "context", "!cuda.context type">;

// Stream represents a CUDA Stream.
def StreamType : OpaqueType<"cuda", "stream", "!cuda.stream type">;

def EventType : OpaqueType<"cuda", "event", "!cuda.event type">;

def AllocatorType : OpaqueType<"cuda", "allocator", "!cuda.allocator type">;

// Buffer represents a CUDA device memory buffer.
def BufferType : OpaqueType<"cuda", "buffer", "!cuda.buffer type">;

def CublasHandleType : OpaqueType<"cuda", "blas_handle", "!cuda.blas_handle type">;

// "cuda" dialect
def CUDA_Dialect : Dialect {
  let name = "cuda";

  let description = [{
    The CUDA dialect.

    This dialect contains common CUDA operations.
  }];

  let cppNamespace = "cuda";
}

// Base class for CUDA dialect ops.
class CUDA_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<CUDA_Dialect, mnemonic, traits> {

  let assemblyFormat = "operands attr-dict";
}

#endif  // CUDA_OPS_BASE
