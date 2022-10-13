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

//===- gpu_fft_ops.td ----------------------------------------------------===//
//
// CuFFT based CUDA operation definitions.
// We separate FFTs from the other cuda library calls in cuDNN because cuFFT
// is a separate library.
//
//===----------------------------------------------------------------------===//

#ifdef TFRTGPU_FFT_OPS
#else
#define TFRTGPU_FFT_OPS

include "tfrt/gpu/kernels/gpu_ops_base.td"

//===----------------------------------------------------------------------===//
// Fast Fourier Transform Type enum definitions.
//===----------------------------------------------------------------------===//
def TFRTGPU_FftHandleType : TFRTGPU_Type<"OwningFftHandle"> { let mnemonic = "fft.handle"; }

def TFRTGPU_FftTypeAttr : TFRTGPU_WrapperAttr<"FftType">;
def TFRTGPU_FftDirectionAttr : TFRTGPU_WrapperAttr<"FftDirection">;

def TFRTGPU_FftCreateOp : TFRTGPU_Op<"fft.create"> {
  let description = [{
    tfrt_gpu.fft.create returns a tfrt_gpu.fft.handle for the FFT operation
    that is described by the supplied parameters.

    Example:
      %fft = tfrt_gpu.fft.create %ctx, CUFFT_R2C, 1, [4, 4], [16, 4, 1], [16, 4, 1]

    'dims' are the FFT dimensions, with rank one smaller than the two strides.
  }];
  // Note: these attributes will have to become values to support dynamic
  // shapes. But it's unclear at the moment what type they would be and XLIR
  // as the only user is operating on static shapes still.
  let arguments = (ins TFRTGPU_ContextType:$context,
    TFRTGPU_FftTypeAttr:$type, I64Attr:$batch, I64ArrayAttr:$dims,
    I64ArrayAttr:$in_strides, I64ArrayAttr:$out_strides);
  let results = (outs TFRTGPU_FftHandleType:$handle);
  let assemblyFormat = [{
    $context`,` custom<Enum>($type)`,` $batch`,` $dims`,`
    $in_strides`,` $out_strides attr-dict
  }];
  let hasVerifier = 1;
}

def TFRTGPU_FftGetWorkspaceSizeOp : TFRTGPU_Op<"fft.get_workspace_size", [Pure]> {
  let description = [{
    tfrt_gpu.fft.create returns workspace size in bytes required to execute
    the FFT operation.
  }];
  let arguments = (ins TFRTGPU_FftHandleType:$handle);
  let results = (outs I64:$workspace_size);
  let assemblyFormat = [{$handle attr-dict}];
}

def TFRTGPU_FftExecuteOp : TFRTGPU_Op<"fft.execute"> {
  let description = [{
    tfrt_gpu.fft.execute runs the FFT according to the supplied parameters.
  }];
  let arguments = (ins
    TFRTGPU_StreamType:$stream,
    TFRTGPU_FftHandleType:$handle,
    TFRTGPU_BufferType:$input,
    TFRTGPU_BufferType:$output,
    TFRTGPU_BufferType:$workspace,
    TFRTGPU_FftDirectionAttr:$direction,
    TFRT_ChainType:$chain
  );
  let results = (outs TFRT_ChainType);
  let assemblyFormat = [{
    $stream`,` $handle`,` $input`,` $output`,` $workspace`,`
    custom<Enum>($direction)`,` $chain attr-dict
  }];
}

#endif  // TFRTGPU_FFT_OPS
