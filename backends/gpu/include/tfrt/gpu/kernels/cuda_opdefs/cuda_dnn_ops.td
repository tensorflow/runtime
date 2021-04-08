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

//===- cuda_dnn_ops.td ----------------------------------------------------===//
//
// CUDA based CUDA operation definitions.
//
// The same ops should be implementable with a ROCm backend as well.
// Current doc strings refer to CUDA only.
//
//===----------------------------------------------------------------------===//

#ifdef CUDA_DNN_OPS
#else
#define CUDA_DNN_OPS

include "tfrt/tfrt_op_base.td"
include "tfrt/tensor/opdefs/tensor_shape_base.td"
include "tfrt/gpu/kernels/cuda_opdefs/cuda_ops_base.td"

def DnnActivationDescriptorType : OpaqueType<"tfrt_cuda", "dnn_activation_descriptor", "!trft_cuda.dnn_activation_descriptor type">;
def DnnConvolutionDescriptorType : OpaqueType<"tfrt_cuda", "dnn_convolution_descriptor", "!trft_cuda.dnn_convolution_descriptor type">;
def DnnFilterDescriptorType : OpaqueType<"tfrt_cuda", "dnn_filter_descriptor", "!trft_cuda.dnn_filter_descriptor type">;
def DnnHandleType : OpaqueType<"tfrt_cuda", "dnn_handle", "!trft_cuda.dnn_handle type">;
def DnnPoolingDescriptorType : OpaqueType<"tfrt_cuda", "dnn_pooling_descriptor", "!trft_cuda.dnn_pooling_descriptor type">;
def DnnTensorDescriptorType : OpaqueType<"tfrt_cuda", "dnn_tensor_descriptor", "!trft_cuda.dnn_tensor_descriptor type">;


def DnnCreateOp : CUDA_Op<"dnn.create"> {
  let summary = "tfrt_cuda dnn.create operation";
  let description = [{
    tfrt_cuda.dnn.create initializes the DNN library and creates a handle to an
    opaque structure holding the DNN library context. It allocates hardware
    resources on the host and device and must be called prior to making any
    other DNN library calls.

    The DNN library handle is tied to the provided CUDA device (context).

    Example:
    %ch1 = tfrt.new.chain
    %ch2 = tfrt_cuda.init %ch1
    %index = tfrt.constant.i32 0
    %device = tfrt_cuda.device.get %index, %ch1
    %context, %ch4 = tfrt_cuda_test.context.get %device, %ch1
    %cudnn, %ch5 = tfrt_cuda.dnn.create %context, %ch1

  }];
  let arguments = (ins
                   ContextType,
                   TFRT_ChainType
                  );
  let results = (outs
                 DnnHandleType,
                 TFRT_ChainType
                );
}

def DnnSetStream : CUDA_Op<"dnn.set_stream"> {
  let summary = "tfrt_cuda.dnn.set_stream operation";
  let description = [{
    tfrt_cuda.dnn.set_stream sets the user's stream in the DNN handle.
    The new stream will be used to launch DNN GPU kernels or to synchronize
    to this stream when DNN kernels are launched in the internal streams.
    If the DNN library stream is not set, all kernels use the default (NULL)
    stream. Setting the user stream in the DNN handle guarantees the
    issue-order execution of DNN calls and other GPU kernels launched in the
    same stream.

    Example:
    %ch2 = tfrt_cuda.dnn.set_stream %dnn_handle, %stream, %ch1

  }];
  let arguments = (ins
                   DnnHandleType,
                   StreamType,
                   TFRT_ChainType
                  );
  let results = (outs
                 TFRT_ChainType
                );
}

def DnnGetStream : CUDA_Op<"dnn.get_stream"> {
  let summary = "tfrt_cuda.dnn.get_stream operation";
  let description = [{
    tfrt_cuda.dnn.get_stream retrieves the user stream programmed in the DNN
    handle. When the user's stream was not set in the DNN handle, this function
    reports the null-stream.

    Example:
    %stream, %ch2 = tfrt_cuda.dnn.get_stream %dnn_handle, %ch1

  }];
  let arguments = (ins
                   DnnHandleType,
                   TFRT_ChainType
                  );
  let results = (outs
                 StreamType,
                 TFRT_ChainType
                );
}

def DnnDestroyOp : CUDA_Op<"dnn.destroy"> {
  let summary = "tfrt_cuda dnn.destroy operation";
  let description = [{
    tfrt_cuda.dnn.destroy releases the resources used by the DNN handle.
    This function is usually the last call with a particular handle to the DNN
    handle.

    Example:
    %ch3 = tfrt_cuda.dnn.destroy %dnn_handle, %ch2

  }];
  let arguments = (ins
                   DnnHandleType,
                   TFRT_ChainType
                  );
  let results = (outs
                 TFRT_ChainType
                );
}

def DnnCreatePoolingDescriptorOp : CUDA_Op<"dnn.create_pooling_descriptor"> {
  let summary = "tfrt_cuda dnn.create_pooling_descriptor operation";
  let description = [{
    tfrt_cuda.dnn.create_pooling_descriptor creates a pooling descriptor object
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
    %pooling_desc, %ch18 = tfrt_cuda.dnn.create_pooling_descriptor %context, %mode, %nan_propagation, %window_dimensions, %paddings, %strides, %ch17

  }];
  let arguments = (ins
                   ContextType,
                   UI32,
                   UI32,
                   TensorType,
                   TensorType,
                   TensorType,
                   TFRT_ChainType
                   );
  let results = (outs
                 DnnPoolingDescriptorType,
                 TFRT_ChainType
                );
}

def DnnDestroyPoolingDescriptorOp : CUDA_Op<"dnn.destroy_pooling_descriptor"> {
  let summary = "tfrt_cuda dnn.destroy_pooling_descriptor operation";
  let description = [{
    tfrt_cuda.dnn.destroy_pooling_descriptor destroys a previously created
    pooling descriptor object.

    Example:
    %ch5 = tfrt_cuda.dnn.destroy_pooling_descriptor %dnn_pooling_descriptor, %ch4

  }];
  let arguments = (ins
                   DnnPoolingDescriptorType,
                   TFRT_ChainType
                  );
  let results = (outs
                 TFRT_ChainType
                );
}

def DnnCreateTensorDescriptorOp : CUDA_Op<"dnn.create_tensor_descriptor"> {
  let summary = "tfrt_cuda dnn.create_tensor_descriptor operation";
  let description = [{
    tfrt_cuda.dnn.create_tensor_descriptor creates a generic tensor
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
    %out_desc, %ch25 = tfrt_cuda.dnn.set_tensor_descriptor %context, %data_type_out, %dim_out, %stride_out, %ch24

  }];
  let arguments = (ins
                   ContextType,
                   UI32,
                   TensorType,
                   TensorType,
                   TFRT_ChainType
                  );
  let results = (outs
                 DnnTensorDescriptorType,
                 TFRT_ChainType
                );
}

def DnnDestroyTensorDescriptorOp : CUDA_Op<"dnn.destroy_tensor_descriptor"> {
  let summary = "tfrt_cuda dnn.destroy_tensor_descriptor operation";
  let description = [{
    tfrt_cuda.dnn.destroy_tensor_descriptor destroys a previously created tensor
    descriptor object.

    Example:
    %ch3 = tfrt_cuda.dnn.destroy_tensor_descriptor %dnn_tensor_descriptor, %ch4

  }];
  let arguments = (ins
                   DnnTensorDescriptorType,
                   TFRT_ChainType
                  );
  let results = (outs
                 TFRT_ChainType
                );
}

def DnnPoolingForwardOp : CUDA_Op<"dnn.pooling_forward"> {
  let summary = "tfrt_cuda dnn.pooling_forward operation";
  let description = [{
    tfrt_cuda.dnn.pooling_forward computes pooling of input values (meaning,
    the maximum or average of several adjacent values) to produce an
    output with smaller height and/or width.

    Example:
    %ch33 = tfrt_cuda.dnn.pooling_forward %context, %cudnn, %pooling_desc, %alpha, %in_desc, %input_device_buffer, %beta, %out_desc, %output_device_buffer, %ch32

}];
  let arguments = (ins
                   ContextType,
                   DnnHandleType,
                   DnnPoolingDescriptorType,
                   F32,
                   DnnTensorDescriptorType,
                   BufferType,
                   F32,
                   DnnTensorDescriptorType,
                   BufferType,
                   TFRT_ChainType
                  );
  let results = (outs TFRT_ChainType);
}

def DnnPoolingBackwardOp : CUDA_Op<"dnn.pooling_backward"> {
  let summary = "tfrt_cuda dnn.pooling_backward operation";
  let description = [{
    tfrt_cuda.dnn.pooling_backward computes the gradient of a pooling operation.

    Example:
    %ch42 = tfrt_cuda.dnn.pooling_backward %context, %cudnn, %pooling_desc, %alpha, %out_desc, %output_device_buffer, %out_desc, %output_device_buffer, %in_desc, %input_device_buffer, %beta, %in_desc, %in_grad_device_buffer, %ch41

  }];
  let arguments = (ins ContextType,
                   DnnHandleType,
                   DnnPoolingDescriptorType,
                   F32,
                   DnnTensorDescriptorType,
                   BufferType,
                   DnnTensorDescriptorType,
                   BufferType,
                   DnnTensorDescriptorType,
                   BufferType,
                   F32,
                   DnnTensorDescriptorType,
                   BufferType,
                   TFRT_ChainType
                  );
  let results = (outs TFRT_ChainType);
}

// TODO(b/184675727): Remove this Op once the tfrt_cuda dialect moves from
// async to sync Ops.
def DnnConvolutionForwardOp : CUDA_Op<"dnn.convolution_forward"> {
  let summary = "tfrt_cuda dnn.convolution_forward operation";
  let description = [{
    tfrt_cuda.dnn.convolution_forward executes convolutions or
                                      cross-correlations over x using filters
                                      specified with w, returning results in y.
                                      Scaling factors alpha and beta can be
                                      used to scale the input tensor and the
                                      output tensor respectively.
  }];
  let arguments = (ins ContextType:$context,
                   DnnHandleType:$handle,
                   DnnTensorDescriptorType:$x_desc,
                   BufferType:$x,
                   DnnFilterDescriptorType:$w_desc,
                   BufferType:$w,
                   DnnConvolutionDescriptorType:$conv_desc,
                   UI64:$algo,
                   BufferType:$work_space,
                   DnnTensorDescriptorType:$y_desc,
                   BufferType:$y);
}

// TODO(b/184675727): Remove this Op once the tfrt_cuda dialect moves from
// async to sync Ops.
def DnnConvolutionBackwardDataOp : CUDA_Op<"dnn.convolution_backward_data"> {
  let summary = "tfrt_cuda dnn.convolution_backward_data operation";
  let description = [{
    tfrt_cuda.dnn.convolution_backward_data computes the convolution data
                                            gradient of the tensor dy, where y
                                            is the output of the forward
                                            convolution in
                                            tfrt_cuda.dnn.convolution_forward.
                                            It uses the specified algo, and
                                            returns the results in the output
                                            tensor dx.
}];
  let arguments = (ins ContextType:$context,
                   DnnHandleType:$handle,
                   DnnFilterDescriptorType:$w_desc,
                   BufferType:$w,
                   DnnTensorDescriptorType:$dy_desc,
                   BufferType:$dy,
                   DnnConvolutionDescriptorType:$conv_desc,
                   UI64:$algo,
                   BufferType:$work_space,
                   DnnTensorDescriptorType:$dx_desc,
                   BufferType:$dx);
}

// TODO(b/184675727): Remove this Op once the tfrt_cuda dialect moves from
// async to sync Ops.
def DnnConvolutionBackwardFilterOp : CUDA_Op<"dnn.convolution_backward_filter"> {
  let summary = "tfrt_cuda dnn.convolution_backward_filter operation";
  let description = [{
    tfrt_cuda.dnn.convolution_backward_filter computes the convolution weight
                                              (filter) gradient of the tensor
                                              dy, where y is the output of the
                                              forward convolution in
                                              tfrt_cuda.dnn.convolution_forward.
                                              It uses the specified algo, and
                                              returns the results in the output
                                              tensor dw.
}];
  let arguments = (ins ContextType:$context,
                   DnnHandleType:$handle,
                   DnnTensorDescriptorType:$x_desc,
                   BufferType:$x,
                   DnnTensorDescriptorType:$dy_desc,
                   BufferType:$dy,
                   DnnConvolutionDescriptorType:$conv_desc,
                   UI64:$algo,
                   BufferType:$work_space,
                   DnnFilterDescriptorType:$dw_desc,
                   BufferType:$dw);
}

// TODO(b/184675727): Remove this Op once the tfrt_cuda dialect moves from
// async to sync Ops.
def CudnnConvolutionBiasActivationForwardOp :
             CUDA_Op<"dnn.convolution_bias_activation_forward"> {
  let summary =
    "tfrt_cuda dnn.convolution_bias_activation_forward operation";
  let description = [{
    tfrt_cuda.dnn.convolution_bias_activation_forward applies a bias and then
                        an activation to the convolutions or cross-correlations
                        of tfrt_cuda.dnn.convolution_forward, returning results
                        in y. The full computation follows the equation
                        y = act ( alpha1 * conv(x) + alpha2 * z + bias ).
  }];
  let arguments = (ins ContextType:$context,
                   DnnHandleType:$handle,
                   BufferType:$alpha1,
                   DnnTensorDescriptorType:$x_desc,
                   BufferType:$x,
                   DnnFilterDescriptorType:$w_desc,
                   BufferType:$w,
                   DnnConvolutionDescriptorType:$conv_desc,
                   UI64:$algo,
                   BufferType:$work_space,
                   BufferType:$alpha2,
                   DnnTensorDescriptorType:$z_desc,
                   BufferType:$z,
                   DnnTensorDescriptorType:$bias_desc,
                   BufferType:$bias,
                   DnnActivationDescriptorType:$activation_desc,
                   DnnTensorDescriptorType:$y_desc,
                   BufferType:$y);
}


#endif  // CUDA_DNN_OPS
