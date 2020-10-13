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

//===- hip_forwards.h -------------------------------------------*- C++ -*-===//
//
// Forward-declares HIP API types used in platform-agnostic wrapper headers.
//
//===----------------------------------------------------------------------===//
#ifndef TFRT_GPU_STREAM_HIP_FORWARDS_H_
#define TFRT_GPU_STREAM_HIP_FORWARDS_H_

// Forward declaration of HIP types.
using hipDevice_t = int;
using hipCtx_t = struct ihipCtx_t *;
using hipModule_t = struct ihipModule_t *;
using hipStream_t = struct ihipStream_t *;
using hipEvent_t = struct ihipEvent_t *;
using hipFunction_t = struct ihipModuleSymbol_t *;

// Forward declaration of MIOpen types.
using miopenHandle_t = struct miopenHandle *;
using miopenTensorDescriptor_t = struct miopenTensorStruct *;
using miopenConvolutionDescriptor_t = struct miopenConvolutionStruct *;
using miopenPoolingDescriptor_t = struct miopenPoolingStruct *;
using miopenActivationDescriptor_t = struct miopenActivationStruct *;
using miopenFilterDescriptor_t = struct miopenFilterStruct *;
using miopenDropoutDescriptor_t = struct miopenDropoutStruct *;
using miopenRNNDescriptor_t = struct miopenRNNStruct *;
using miopenPersistentRNNPlan_t = struct miopenPersistentRNNPlan *;
using miopenOpTensorDescriptor_t = struct miopenOpTensorDescriptor *;
using miopenTensorTransformDescriptor_t =
    struct miopenTensorTransformDescriptor *;
using miopenReduceTensorDescriptor_t = struct miopenReduceTensorDescriptor *;
using miopenReduceTensorDescriptor_t = struct miopenReduceTensorDescriptor *;
using miopenRuntimeTag_t = struct miopenRuntimeTag *;
using miopenStatus_t = struct miopenStatus *;
using miopenConvolutionFwdAlgo_t = struct miopenConvolutionFwdAlgo *;
using miopenConvolutionBwdFilterAlgo_t =
    struct miopenConvolutionBwdFilterAlgo *;
using miopenConvolutionBwdDataAlgo_t = struct miopenConvolutionBwdDataAlgo *;
using miopenConvolutionFwdAlgoPerf_t = struct mioopenConvolutionFwdAlgoPerf *;
using miopenConvolutionBwdFilterAlgoPerf_t =
    struct miopenConvolutionBwdFilterAlgoPerf *;
using miopenConvolutionBwdDataAlgoPerf_t =
    struct miopenConvolutionBwdDataAlgoPerf *;

// Forward declaration of rocBLAS types.
using rocblas_handle = struct _rocblas_handle *;

// Forward declaration of rocSOLVER types.
using rocsolver_handle = rocblas_handle;

#endif  // TFRT_GPU_STREAM_HIP_FORWARDS_H_
