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

//===- gpu_ops_base.td ---------------------------------------------------===//
//
// GPU dialect and type definitions.
//
//===----------------------------------------------------------------------===//

#ifdef GPU_OPS_BASE
#else
#define GPU_OPS_BASE

include "tfrt/tfrt_op_base.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

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
  let summary = !strconcat("tfrt_gpu.", mnemonic, " operation");
  let assemblyFormat = "operands attr-dict";
}

// Type definitions.

class GPU_Type<string name, list<Trait> traits = []> :
    TypeDef<GPU_Dialect, name, traits> {
  let summary = !strconcat("!tfrt_gpu.", mnemonic, " type");
}

def GPU_AllocatorType : GPU_Type<"Allocator"> { let mnemonic = "allocator"; }
def GPU_BufferType : GPU_Type<"Buffer"> { let mnemonic = "buffer"; }
def GPU_ContextType : GPU_Type<"Context"> { let mnemonic = "context"; }
def GPU_StreamType : GPU_Type<"Stream"> { let mnemonic = "stream"; }
// Specific to driver, but included here for export.
def GPU_ModuleType : GPU_Type<"Module"> { let mnemonic = "module"; }
// Specific to CCL, but included here for export.
def GPU_CclHandleType : GPU_Type<"CclHandle"> { let mnemonic = "ccl.handle"; }

// Attribute definitions.

class GPU_WrapperAttr<string name> : DialectAttr<
    GPU_Dialect, CPred<"$_self.isa<::tfrt::gpu::" # name # "Attr>()">> {
  let storageType = "::tfrt::gpu::" # name # "Attr";
  let returnType = "::tfrt::gpu::wrapper::" # name;
  let constBuilderCall = storageType # "::get($_builder.getContext(), $0)";
}

def GPU_PlatformAttr : GPU_WrapperAttr<"Platform">;
// Specific to CCL, but included here for export.
def GPU_CclDataTypeAttr : GPU_WrapperAttr<"CclDataType"> {
  let returnType = "::ncclDataType_t";
}

#endif  // GPU_OPS_BASE
