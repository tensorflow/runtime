/*
 * Copyright 2020 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Forward-declares CUDA API types used in platform-agnostic wrapper headers.
#ifndef TFRT_GPU_WRAPPER_CUDA_FORWARDS_H_
#define TFRT_GPU_WRAPPER_CUDA_FORWARDS_H_

// Forward declaration of CUDA driver types.
using CUdevice = int;
using CUcontext = struct CUctx_st *;
using CUmodule = struct CUmod_st *;
using CUstream = struct CUstream_st *;
using CUevent = struct CUevent_st *;
using CUfunction = struct CUfunc_st *;

// Enums for corresponding #defines in the CUDA headers.
enum CUmemhostalloc_flags_enum : int {
  CU_MEMHOSTALLOC_DEFAULT = 0x0,
  // CU_MEMHOSTALLOC_PORTABLE = 0x1,
  // CU_MEMHOSTALLOC_DEVICEMAP = 0x2,
  // CU_MEMHOSTALLOC_WRITECOMBINED = 0x4,
};
using CUmemhostalloc_flags = CUmemhostalloc_flags_enum;
enum CUmemhostregister_flags_enum : int {
  CU_MEMHOSTREGISTER_DEFAULT = 0x0,
  // CU_MEMHOSTREGISTER_PORTABLE = 0x1,
  // CU_MEMHOSTREGISTER_DEVICEMAP = 0x2,
  // CU_MEMHOSTREGISTER_IOMEMORY = 0x4,
};
using CUmemhostregister_flags = CUmemhostregister_flags_enum;

// Forward declaration of CUDA runtime types.
using cudaStream_t = struct CUstream_st *;

// Forward declaration of cuDNN types.
struct cudnnRuntimeTag_t;

using cudnnHandle_t = struct cudnnContext *;
using cudnnTensorDescriptor_t = struct cudnnTensorStruct *;
using cudnnConvolutionDescriptor_t = struct cudnnConvolutionStruct *;
using cudnnPoolingDescriptor_t = struct cudnnPoolingStruct *;
using cudnnFilterDescriptor_t = struct cudnnFilterStruct *;
using cudnnLRNDescriptor_t = struct cudnnLRNStruct *;
using cudnnActivationDescriptor_t = struct cudnnActivationStruct *;
using cudnnSpatialTransformerDescriptor_t =
    struct cudnnSpatialTransformerStruct *;
using cudnnOpTensorDescriptor_t = struct cudnnOpTensorStruct *;
using cudnnReduceTensorDescriptor_t = struct cudnnReduceTensorStruct *;
using cudnnCTCLossDescriptor_t = struct cudnnCTCLossStruct *;
using cudnnTensorTransformDescriptor_t = struct cudnnTensorTransformStruct *;
using cudnnDropoutDescriptor_t = struct cudnnDropoutStruct *;
using cudnnRNNDescriptor_t = struct cudnnRNNStruct *;
using cudnnPersistentRNNPlan_t = struct cudnnPersistentRNNPlan *;
using cudnnRNNDataDescriptor_t = struct cudnnRNNDataStruct *;
using cudnnAlgorithmDescriptor_t = struct cudnnAlgorithmStruct *;
using cudnnAlgorithmPerformance_t = struct cudnnAlgorithmPerformanceStruct *;
using cudnnSeqDataDescriptor_t = struct cudnnSeqDataStruct *;
using cudnnAttnDescriptor_t = struct cudnnAttnStruct *;
using cudnnFusedOpsConstParamPack_t = struct cudnnFusedOpsConstParamStruct *;
using cudnnFusedOpsVariantParamPack_t =
    struct cudnnFusedOpsVariantParamStruct *;
using cudnnFusedOpsPlan_t = struct cudnnFusedOpsPlanStruct *;

// Forward declaration of cuBLAS types.
using cublasHandle_t = struct cublasContext *;

// Forward declaration of cuSOLVER types.
using cusolverDnHandle_t = struct cusolverDnContext *;
using syevjInfo_t = struct syevjInfo *;
using gesvdjInfo_t = struct gesvdjInfo *;

// Forward declaration of cuFFT types.
using cufftHandle = int;

#endif  // TFRT_GPU_WRAPPER_CUDA_FORWARDS_H_
