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

//===- cuda_test_ops.td ---------------------------------------------------===//
//
// Operation definitions for testing CUDA.
//
//===----------------------------------------------------------------------===//

#ifdef CUDA_TEST_OPS
#else
#define CUDA_TEST_OPS

include "tfrt/tfrt_op_base.td"
include "tfrt/tensor/opdefs/tensor_shape_base.td"
include "tfrt/gpu/kernels/cuda_opdefs/cuda_ops_base.td"

// "cuda_test" dialect
def CUDA_Test_Dialect : Dialect {
  let name = "cuda_test";

  let description = [{
    The CUDA dialect.

    This dialect contains common CUDA operations.
  }];

  let cppNamespace = "cuda";
}

// Base class for CUDA test dialect ops.
class CUDA_Test_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<CUDA_Test_Dialect, mnemonic, traits> {

  // Each registered op needs to provide a printer, a parser and a verifier.
  let printer = [{ return tfrt::cuda::print(p, *this); }];
  let parser = [{ return tfrt::cuda::parse$cppClass(parser, result); }];
}


def TestContextGetOp : CUDA_Test_Op<"context.get"> {
  let summary = "cuda_test context.get operation";
  let description = [{
    cuda_test.context.get returns a CUDA context for the given device.

    The returned context does not need to be destroyed or released and user
    can assume that it will outlive the function where it is used.

    Example:
      %ch1 = tfrt.new.chain
      %ch2 = cuda.init %ch1
      %index = tfrt.constant.i32 0
      %device, %ch3 = cuda.device.get %index, %ch2
      %ctx, %ch4 = cuda_test.context.get %device, %ch2
  }];
  let arguments = (ins DeviceType, TFRT_ChainType);
  let results = (outs ContextType, TFRT_ChainType);
  let assemblyFormat = "operands attr-dict";
}

#endif  // CUDA_TEST_OPS
