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

def GPU_DnnHandleType : GPU_Type<"DnnHandle"> { let mnemonic = "dnn.handle"; }
def GPU_DnnActivationDescriptorType : GPU_Type<"DnnActivationDescriptor"> {
  let mnemonic = "dnn.activation.descriptor";
}
def GPU_DnnConvolutionDescriptorType : GPU_Type<"DnnConvolutionDescriptor"> {
  let mnemonic = "dnn.convolution.descriptor";
}
def GPU_DnnFilterDescriptorType : GPU_Type<"DnnFilterDescriptor"> {
  let mnemonic = "dnn.filter.descriptor";
}
def GPU_DnnPoolingDescriptorType : GPU_Type<"DnnPoolingDescriptor"> {
  let mnemonic = "dnn.pooling.descriptor";
}
def GPU_DnnTensorDescriptorType : GPU_Type<"DnnTensorDescriptor"> {
  let mnemonic = "dnn.tensor.descriptor";
}

def GPU_DnnDataTypeAttr : GPU_WrapperAttr<"DnnDataType">;
def GPU_DnnConvolutionModeAttr : GPU_WrapperAttr<"DnnConvolutionMode">;

def GPU_DnnCreateOp : GPU_Op<"dnn.create"> {
  let description = [{
    tfrt_gpu.dnn.create initializes the DNN library and creates a handle to an
    opaque structure holding the DNN library context. It allocates hardware
    resources on the host and device and must be called prior to making any
    other DNN library calls.

    The DNN library handle is tied to the provided CUDA device (context).

    Example:
    %cudnn = tfrt_gpu.dnn.create %stream

  }];
  let arguments = (ins GPU_StreamType);
  let results = (outs GPU_DnnHandleType);
}

// TODO(hanbinyoon): Consider removing the input chain for the
// DnnCreate*DescriptorOp(s).
def GPU_DnnCreateConvolutionDescriptorOp : GPU_Op<"dnn.create_convolution_descriptor"> {
  let description = [{
    tfrt_gpu.dnn.create_convolution_descriptor creates a convolution descriptor
    object by allocating the memory needed to hold its opaque structure. The
    data is initialized to provided values.

    Example:
    %desc = tfrt_gpu.dnn.create_convolution_descriptor CUDNN_DATA_FLOAT,
      CUDNN_CROSS_CORRELATION, [0 : i32, 0 : i32], [1 : i32, 1 : i32],
      [1 : i32, 1 : i32], %ch0
  }];
  let arguments = (ins GPU_DnnDataTypeAttr:$compute_type,
                   GPU_DnnConvolutionModeAttr:$conv_mode, I32ArrayAttr:$pad,
                   I32ArrayAttr:$filter_stride, I32ArrayAttr:$dilation,
                   TFRT_ChainType:$chain);
  let results = (outs GPU_DnnConvolutionDescriptorType);
  let assemblyFormat = [{
    custom<Enum>($compute_type)`,` custom<Enum>($conv_mode)`,` $pad`,`
    $filter_stride`,` $dilation`,` $chain attr-dict
  }];
}

def GPU_DnnCreateFilterDescriptorOp : GPU_Op<"dnn.create_filter_descriptor"> {
  let description = [{
    tfrt_gpu.dnn.create_filter_descriptor creates a filter descriptor object by
    allocating the memory needed to hold its opaque structure. The data is
    initialized to provided values.

    Example:
    %desc = tfrt_gpu.dnn.create_filter_descriptor CUDNN_DATA_FLOAT,
      [3 : i32, 3 : i32], %ch0
  }];
  let arguments = (ins GPU_DnnDataTypeAttr:$data_type,
                   I32Attr:$tensor_format,
                   I32ArrayAttr:$dimensions, TFRT_ChainType:$chain);
  let results = (outs GPU_DnnFilterDescriptorType);
  let assemblyFormat = [{
    custom<Enum>($data_type)`,` $tensor_format`,` $dimensions`,` $chain
    attr-dict
  }];
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
    %mode = tfrt.constant.ui32 0
    %nan_propagation = tfrt.constant.ui32 0
    %pooling_desc = tfrt_gpu.dnn.create_pooling_descriptor %context, %mode,
      %nan_propagation, [3 : i32, 3 : i32], [0 : i32, 0 : i32],
      [1 : i32, 1 : i32], %ch0
  }];
  let arguments = (ins GPU_ContextType:$context, UI32:$mode,
                   UI32:$nan_propagation, I32ArrayAttr:$window_dimensions,
                   I32ArrayAttr:$paddings, I32ArrayAttr:$strides,
                   TFRT_ChainType:$chain);
  let results = (outs GPU_DnnPoolingDescriptorType);
  let assemblyFormat = [{
    $context`,` $mode`,` $nan_propagation`,` $window_dimensions`,` $paddings`,`
    $strides`,` $chain attr-dict
  }];
}

def GPU_DnnCreateTensorDescriptorOp : GPU_Op<"dnn.create_tensor_descriptor"> {
  let description = [{
    tfrt_gpu.dnn.create_tensor_descriptor creates a generic tensor
    descriptor object by allocating the memory needed to hold its opaque
    structure. The data is initialized to provided values.

    Example:
    %desc = tfrt_gpu.dnn.create_tensor_descriptor CUDNN_DATA_FLOAT,
      [3 : i32, 3 : i32], [1 : i32, 1 : i32], %ch0
  }];
  let arguments = (ins GPU_DnnDataTypeAttr:$data_type, I32ArrayAttr:$dimensions,
                   I32ArrayAttr:$strides, TFRT_ChainType:$chain);
  let results = (outs GPU_DnnTensorDescriptorType);
  let assemblyFormat = [{
    custom<Enum>($data_type)`,` $dimensions`,` $strides`,` $chain attr-dict
  }];
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

def GPU_DnnConvolutionForwardOp : GPU_Op<"dnn.convolution_forward"> {
  let description = [{
    tfrt_gpu.dnn.convolution_forward executes convolutions or
                                      cross-correlations over x using filters
                                      specified with w, returning results in y.
  }];
  let arguments = (ins GPU_DnnHandleType:$handle,
                   GPU_DnnDataTypeAttr:$compute_type,
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
  let assemblyFormat = [{
    $handle`,` custom<Enum>($compute_type)`,` $x_desc`,` $x`,` $w_desc`,` $w`,`
    $conv_desc`,` $algo`,` $work_space`,` $y_desc`,` $y attr-dict
  }];
}

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
                   GPU_DnnDataTypeAttr:$compute_type,
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
  let assemblyFormat = [{
    $handle`,` custom<Enum>($compute_type)`,` $w_desc`,` $w`,` $dy_desc`,`
    $dy`,` $conv_desc`,` $algo`,` $work_space`,` $dx_desc`,` $dx attr-dict
  }];
}

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
                   GPU_DnnDataTypeAttr:$compute_type,
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
  let assemblyFormat = [{
    $handle`,` custom<Enum>($compute_type)`,` $x_desc`,` $x`,` $dy_desc`,`
    $dy`,` $conv_desc`,` $algo`,` $work_space`,` $dw_desc`,` $dw attr-dict
  }];
}

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
