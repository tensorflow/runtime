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

//===- gpu_dnn_ops.td ----------------------------------------------------===//
//
// CUDA based CUDA operation definitions.
//
// The same ops should be implementable with a ROCm backend as well.
// Current doc strings refer to CUDA only.
//
//===----------------------------------------------------------------------===//

#ifdef GPU_DNN_OPS
#else
#define GPU_DNN_OPS

include "tfrt/gpu/kernels/gpu_ops_base.td"

def GPU_DnnActivationDescriptorType : GPU_Type<"dnn_activation_descriptor">;
def GPU_DnnConvolutionDescriptorType : GPU_Type<"dnn_convolution_descriptor">;
def GPU_DnnFilterDescriptorType : GPU_Type<"dnn_filter_descriptor">;
def GPU_DnnHandleType : GPU_Type<"dnn_handle">;
def GPU_DnnPoolingDescriptorType : GPU_Type<"dnn_pooling_descriptor">;
def GPU_DnnTensorDescriptorType : GPU_Type<"dnn_tensor_descriptor">;


def GPU_DnnCreateOp : GPU_Op<"dnn.create"> {
  let description = [{
    tfrt_gpu.dnn.create initializes the DNN library and creates a handle to an
    opaque structure holding the DNN library context. It allocates hardware
    resources on the host and device and must be called prior to making any
    other DNN library calls.

    The DNN library handle is tied to the provided CUDA device (context).

    Example:
    %cudnn, %ch5 = tfrt_gpu.dnn.create %context, %ch1

  }];
  let arguments = (ins GPU_StreamType, TFRT_ChainType);
  let results = (outs GPU_DnnHandleType);
}

def GPU_DnnCreatePoolingDescriptorOp : GPU_Op<"dnn.create_pooling_descriptor"> {
  let description = [{
    tfrt_gpu.dnn.create_pooling_descriptor creates a pooling descriptor object
    by allocating the memory needed to hold its opaque structure then it
    initializes created pooling descriptor object.

    mode is an UI32 value with the following meaning:
    0 -> PoolingMax,
    1 -> PoolingAverageCountIncludePadding,
    2 -> PoolingAverageCountExcludePadding,
    3 -> PoolingMaxDeterministic,
    any other value -> Error

    nan_propagation is an UI32 value with the following meaning:
    0 -> NotPropagateNan,
    1 -> PropagateNan,
    any other value -> Error

    Example:
    %dim0 = tfrt.constant.i32 3
    %dim1 = tfrt.constant.i32 3
    %window_dimensions = tfrt_dht.create_uninitialized_tensor.i32.1 [2 : i64]
    %ch15 = "tfrt_dht.set_tensor_with_values.i32"(%window_dimensions, %ch14, %dim0, %dim1):(!t.tensor, !tfrt.chain, i32, i32) -> !tfrt.chain
    %p0 = tfrt.constant.i32 0
    %p1 = tfrt.constant.i32 0
    %paddings = tfrt_dht.create_uninitialized_tensor.i32.1 [2 : i64]
    %ch16 = "tfrt_dht.set_tensor_with_values.i32"(%paddings, %ch15, %p0, %p1):(!t.tensor, !tfrt.chain, i32, i32) -> !tfrt.chain
    %s0 = tfrt.constant.i32 1
    %s1 = tfrt.constant.i32 1
    %strides = tfrt_dht.create_uninitialized_tensor.i32.1 [2 : i64]
    %ch17 = "tfrt_dht.set_tensor_with_values.i32"(%strides, %ch16, %s0, %s1):(!t.tensor, !tfrt.chain, i32, i32) -> !tfrt.chain
    %mode = tfrt.constant.ui32 0
    %nan_propagation = tfrt.constant.ui32 0
    %pooling_desc, %ch18 = tfrt_gpu.dnn.create_pooling_descriptor %context, %mode, %nan_propagation, %window_dimensions, %paddings, %strides, %ch17

  }];
  let arguments = (ins GPU_ContextType, UI32, UI32, TensorType, TensorType,
                   TensorType, TFRT_ChainType);
  let results = (outs GPU_DnnPoolingDescriptorType);
}

def GPU_DnnCreateTensorDescriptorOp : GPU_Op<"dnn.create_tensor_descriptor"> {
  let description = [{
    tfrt_gpu.dnn.create_tensor_descriptor creates a generic tensor
    descriptor object by allocating the memory needed to hold its opaque
    structure. The data is initialized to provided values.

    data_type is an UI32 value with the following meaning:
    0 -> Float,
    1 -> Double,
    2 -> Half,
    3 -> Int8,
    4 -> Int32,
    5 -> Int8x4,
    6 -> Uint8,
    7 -> Uint8x4,
    8 -> Uint8x32
    any other value -> Error

    Example:
    %dout0 = tfrt.constant.i32 2
    %dout1 = tfrt.constant.i32 2
    %dout2 = tfrt.constant.i32 8
    %dout3 = tfrt.constant.i32 8
    %dim_out = tfrt_dht.create_uninitialized_tensor.i32.1 [4 : i64]
    %ch23 = "tfrt_dht.set_tensor_with_values.i32"(%dim_out, %ch22, %dout0, %dout1, %dout2, %dout3):(!t.tensor, !tfrt.chain, i32, i32, i32, i32) -> !tfrt.chain
    %sout0 = tfrt.constant.i32 128
    %sout1 = tfrt.constant.i32 64
    %sout2 = tfrt.constant.i32 8
    %sout3 = tfrt.constant.i32 1
    %stride_out = tfrt_dht.create_uninitialized_tensor.i32.1 [4 : i64]
    %ch24 = "tfrt_dht.set_tensor_with_values.i32"(%stride_out, %ch23, %sout0, %sout1, %sout2, %sout3):(!t.tensor, !tfrt.chain, i32, i32, i32, i32) -> !tfrt.chain
    %data_type_out = tfrt.constant.ui32 0
    %out_desc, %ch25 = tfrt_gpu.dnn.set_tensor_descriptor %context, %data_type_out, %dim_out, %stride_out, %ch24

  }];
  let arguments = (ins GPU_ContextType, UI32, TensorType, TensorType, TFRT_ChainType);
  let results = (outs GPU_DnnTensorDescriptorType);
}

def GPU_DnnPoolingForwardOp : GPU_Op<"dnn.pooling_forward"> {
  let description = [{
    tfrt_gpu.dnn.pooling_forward computes pooling of input values (meaning,
    the maximum or average of several adjacent values) to produce an
    output with smaller height and/or width.

    Example:
    %ch33 = tfrt_gpu.dnn.pooling_forward %context, %cudnn, %pooling_desc, %alpha, %in_desc, %input_device_buffer, %beta, %out_desc, %output_device_buffer, %ch32

}];
  let arguments = (ins
                   GPU_DnnHandleType,
                   GPU_DnnPoolingDescriptorType,
                   F32,
                   GPU_DnnTensorDescriptorType,
                   GPU_BufferType,
                   F32,
                   GPU_DnnTensorDescriptorType,
                   GPU_BufferType,
                   TFRT_ChainType
                  );
  let results = (outs TFRT_ChainType);
}

def GPU_DnnPoolingBackwardOp : GPU_Op<"dnn.pooling_backward"> {
  let description = [{
    tfrt_gpu.dnn.pooling_backward computes the gradient of a pooling operation.

    Example:
    %ch42 = tfrt_gpu.dnn.pooling_backward %context, %cudnn, %pooling_desc, %alpha, %out_desc, %output_device_buffer, %out_desc, %output_device_buffer, %in_desc, %input_device_buffer, %beta, %in_desc, %in_grad_device_buffer, %ch41

  }];
  let arguments = (ins GPU_DnnHandleType,
                   GPU_DnnPoolingDescriptorType,
                   F32,
                   GPU_DnnTensorDescriptorType,
                   GPU_BufferType,
                   GPU_DnnTensorDescriptorType,
                   GPU_BufferType,
                   GPU_DnnTensorDescriptorType,
                   GPU_BufferType,
                   F32,
                   GPU_DnnTensorDescriptorType,
                   GPU_BufferType,
                   TFRT_ChainType
                  );
  let results = (outs TFRT_ChainType);
}

// TODO(b/184675727): Remove this Op once the tfrt_gpu dialect moves from
// async to sync Ops.
def GPU_DnnConvolutionForwardOp : GPU_Op<"dnn.convolution_forward"> {
  let description = [{
    tfrt_gpu.dnn.convolution_forward executes convolutions or
                                      cross-correlations over x using filters
                                      specified with w, returning results in y.
                                      Scaling factors alpha and beta can be
                                      used to scale the input tensor and the
                                      output tensor respectively.
  }];
  let arguments = (ins GPU_DnnHandleType:$handle,
                   GPU_DnnTensorDescriptorType:$x_desc,
                   GPU_BufferType:$x,
                   GPU_DnnFilterDescriptorType:$w_desc,
                   GPU_BufferType:$w,
                   GPU_DnnConvolutionDescriptorType:$conv_desc,
                   UI64:$algo,
                   GPU_BufferType:$work_space,
                   GPU_DnnTensorDescriptorType:$y_desc,
                   GPU_BufferType:$y);
  let results = (outs TFRT_ChainType);
}

// TODO(b/184675727): Remove this Op once the tfrt_gpu dialect moves from
// async to sync Ops.
def GPU_DnnConvolutionBackwardDataOp : GPU_Op<"dnn.convolution_backward_data"> {
  let description = [{
    tfrt_gpu.dnn.convolution_backward_data computes the convolution data
                                            gradient of the tensor dy, where y
                                            is the output of the forward
                                            convolution in
                                            tfrt_gpu.dnn.convolution_forward.
                                            It uses the specified algo, and
                                            returns the results in the output
                                            tensor dx.
}];
  let arguments = (ins GPU_DnnHandleType:$handle,
                   GPU_DnnFilterDescriptorType:$w_desc,
                   GPU_BufferType:$w,
                   GPU_DnnTensorDescriptorType:$dy_desc,
                   GPU_BufferType:$dy,
                   GPU_DnnConvolutionDescriptorType:$conv_desc,
                   UI64:$algo,
                   GPU_BufferType:$work_space,
                   GPU_DnnTensorDescriptorType:$dx_desc,
                   GPU_BufferType:$dx);
  let results = (outs TFRT_ChainType);
}

// TODO(b/184675727): Remove this Op once the tfrt_gpu dialect moves from
// async to sync Ops.
def GPU_DnnConvolutionBackwardFilterOp : GPU_Op<"dnn.convolution_backward_filter"> {
  let description = [{
    tfrt_gpu.dnn.convolution_backward_filter computes the convolution weight
                                              (filter) gradient of the tensor
                                              dy, where y is the output of the
                                              forward convolution in
                                              tfrt_gpu.dnn.convolution_forward.
                                              It uses the specified algo, and
                                              returns the results in the output
                                              tensor dw.
}];
  let arguments = (ins GPU_DnnHandleType:$handle,
                   GPU_DnnTensorDescriptorType:$x_desc,
                   GPU_BufferType:$x,
                   GPU_DnnTensorDescriptorType:$dy_desc,
                   GPU_BufferType:$dy,
                   GPU_DnnConvolutionDescriptorType:$conv_desc,
                   UI64:$algo,
                   GPU_BufferType:$work_space,
                   GPU_DnnFilterDescriptorType:$dw_desc,
                   GPU_BufferType:$dw);
  let results = (outs TFRT_ChainType);
}

// TODO(b/184675727): Remove this Op once the tfrt_gpu dialect moves from
// async to sync Ops.
def GPU_DnnConvolutionBiasActivationForwardOp :
             GPU_Op<"dnn.convolution_bias_activation_forward"> {
  let description = [{
    tfrt_gpu.dnn.convolution_bias_activation_forward applies a bias and then
                        an activation to the convolutions or cross-correlations
                        of tfrt_gpu.dnn.convolution_forward, returning results
                        in y. The full computation follows the equation
                        y = act ( alpha1 * conv(x) + alpha2 * z + bias ).
  }];
  let arguments = (ins GPU_DnnHandleType:$handle,
                   GPU_BufferType:$alpha1,
                   GPU_DnnTensorDescriptorType:$x_desc,
                   GPU_BufferType:$x,
                   GPU_DnnFilterDescriptorType:$w_desc,
                   GPU_BufferType:$w,
                   GPU_DnnConvolutionDescriptorType:$conv_desc,
                   UI64:$algo,
                   GPU_BufferType:$work_space,
                   GPU_BufferType:$alpha2,
                   GPU_DnnTensorDescriptorType:$z_desc,
                   GPU_BufferType:$z,
                   GPU_DnnTensorDescriptorType:$bias_desc,
                   GPU_BufferType:$bias,
                   GPU_DnnActivationDescriptorType:$activation_desc,
                   GPU_DnnTensorDescriptorType:$y_desc,
                   GPU_BufferType:$y);
  let results = (outs TFRT_ChainType);
}


#endif  // GPU_DNN_OPS
