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

// This file implements the tfrt_gpu.dnn kernels.
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "tfrt/gpu/gpu_types.h"
#include "tfrt/gpu/kernels/kernels_detail.h"
#include "tfrt/gpu/wrapper/cudnn_wrapper.h"
#include "tfrt/gpu/wrapper/driver_wrapper.h"
#include "tfrt/gpu/wrapper/miopen_wrapper.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/tensor_shape.h"
#include "tfrt/tensor/tensor_type_registration.h"

namespace tfrt {
namespace gpu {

// A helper class to set env-vars and choose options for dnn-related algorithms.
template <typename EnvVar>
class DnnEnvVar {
 public:
  static bool IsEnabled() {
    static bool is_enabled = IsEnabledImpl();
    return is_enabled;
  }

 private:
  static bool IsEnabledImpl() {
    const char* tf_env_var_val = getenv(EnvVar::kName);
    if (tf_env_var_val != nullptr) {
      tfrt::string_view tf_env_var_val_str(tf_env_var_val);
      if (tf_env_var_val_str == "0") {
        return false;
      }
      return true;
    }
    return EnvVar::kDefaultFlag;
  }
};

// A helper struct to decide whether to use FP32 as the internal compute type
// for convolution when the input data type is FP16. By default it is turned on,
// users can explicitly disable them (choose to use FP16 as the internal compute
// type) through an env-var "TF_FP16_CONV_USE_FP32_COMPUTE=0".
struct ConvDoFP32ComputationFP16Input {
  static constexpr const char* kName = "TF_FP16_CONV_USE_FP32_COMPUTE";
  // Using FP16 as the internal compute type for convolution when the input data
  // type is FP16 is only supported on architectures with true fp16 support
  // (compute capability 5.3 and 6.0). Setting this to false in an unsupported
  // architecture will cause internal errors.
  static constexpr bool kDefaultFlag = true;
};

static Expected<GpuDnnHandle> DnnCreate(Argument<GpuContext> context) {
  auto current = wrapper::CtxSetCurrent(context->get());
  if (!current) return current.takeError();
  auto handle = wrapper::DnnCreate(*current);
  if (!handle) return handle.takeError();
  return GpuDnnHandle(context.ValueRef(), std::move(*handle));
}

static Expected<wrapper::OwningDnnConvolutionDescriptor>
DnnCreateConvolutionDescriptor(
    // Needs to be sorted alphabetically by attribute name!
    Attribute<int32_t> compute_type_attr, Attribute<int32_t> conv_mode_attr,
    ArrayAttr dilation, ArrayAttr filter_stride,
    Attribute<int32_t> math_type_attr, ArrayAttr pad) {
  auto conv_mode =
      wrapper::DnnConvolutionMode::FromOpaqueValue(*conv_mode_attr);
  auto compute_type = wrapper::DnnDataType::FromOpaqueValue(*compute_type_attr);
  auto platform = compute_type.platform();
  auto accumulator_type = GetConvAccumulatorType(
      compute_type, DnnEnvVar<ConvDoFP32ComputationFP16Input>::IsEnabled());
  auto descriptor = wrapper::DnnCreateConvolutionDescriptor(platform);
  if (!descriptor) return descriptor.takeError();
  if (auto error = wrapper::DnnSetConvolutionDescriptor(
          descriptor->get(), pad.GetValue<int32_t>(),
          filter_stride.GetValue<int32_t>(), dilation.GetValue<int32_t>(),
          conv_mode, accumulator_type))
    return std::move(error);
  if (platform == wrapper::Platform::CUDA) {
    auto math_type = wrapper::DnnMathType::FromOpaqueValue(*math_type_attr);
    if (auto error =
            wrapper::CudnnSetConvolutionMathType(descriptor->get(), math_type))
      return std::move(error);
  }
  return std::move(*descriptor);
}

static Expected<wrapper::OwningDnnFilterDescriptor> DnnCreateFilterDescriptor(
    // Needs to be sorted alphabetically by attribute name!
    Attribute<int32_t> data_type_attr, ArrayAttr dimensions,
    Attribute<int32_t> tensor_format_attr) {
  auto data_type = wrapper::DnnDataType::FromOpaqueValue(*data_type_attr);
  auto descriptor = wrapper::DnnCreateFilterDescriptor(data_type.platform());
  if (!descriptor) return descriptor.takeError();
  if (auto error = wrapper::DnnSetFilterDescriptor(
          descriptor->get(), data_type, *tensor_format_attr,
          dimensions.GetValue<int32_t>()))
    return std::move(error);
  return std::move(*descriptor);
}

static Expected<wrapper::OwningDnnPoolingDescriptor> DnnCreatePoolingDescriptor(
    const GpuContext& context, uint32_t mode, uint32_t nan_propagation,
    // Needs to be sorted alphabetically by attribute name!
    ArrayAttr paddings, ArrayAttr strides, ArrayAttr window_dimensions) {
  auto current = wrapper::CtxSetCurrent(context.get());
  auto descriptor = wrapper::DnnCreatePoolingDescriptor(context->platform());
  if (!descriptor) return descriptor.takeError();
  wrapper::DnnNanPropagation cuda_nan_propagation;
  if (current->platform() == wrapper::Platform::CUDA)
    cuda_nan_propagation = static_cast<cudnnNanPropagation_t>(nan_propagation);
  if (auto error = wrapper::DnnSetPoolingDescriptor(
          descriptor->get(), static_cast<wrapper::DnnPoolingMode>(mode),
          cuda_nan_propagation, window_dimensions.GetValue<int32_t>(),
          paddings.GetValue<int32_t>(), strides.GetValue<int32_t>()))
    return std::move(error);
  return std::move(*descriptor);
}

static Expected<GpuDnnTensorDesc> DnnCreateTensorDescriptor(
    // Needs to be sorted alphabetically by attribute name!
    Attribute<int32_t> data_type_attr, ArrayAttr dimensions,
    ArrayAttr strides) {
  auto data_type = wrapper::DnnDataType::FromOpaqueValue(*data_type_attr);
  auto descriptor = wrapper::DnnCreateTensorDescriptor(data_type.platform());
  if (!descriptor) return descriptor.takeError();
  if (auto error = wrapper::DnnSetTensorDescriptor(
          descriptor->get(), data_type, dimensions.GetValue<int32_t>(),
          strides.GetValue<int32_t>()))
    return std::move(error);
  return std::move(*descriptor);
}

static Expected<wrapper::OwningDnnActivationDescriptor>
DnnCreateActivationDescriptor(
    const double coefficient,
    // Needs to be sorted alphabetically by attribute name!
    Attribute<int32_t> mode_attr, Attribute<bool> nan_propagation_attr) {
  auto mode = wrapper::DnnActivationMode::FromOpaqueValue(*mode_attr);
  auto descriptor = wrapper::DnnCreateActivationDescriptor(mode.platform());
  if (!descriptor) return descriptor.takeError();
  if (auto error = wrapper::DnnSetActivationDescriptor(
          descriptor->get(), mode, *nan_propagation_attr, coefficient))
    return std::move(error);
  return std::move(*descriptor);
}

static Error DnnPoolingForward(
    const GpuDnnHandle& handle, const GpuStream& stream,
    const wrapper::OwningDnnPoolingDescriptor& pooling_desc, float alpha,
    const GpuDnnTensorDesc& x_desc, const GpuBuffer& x, float beta,
    const GpuDnnTensorDesc& y_desc, const GpuBuffer& y) {
  auto current = wrapper::CtxSetCurrent(handle.context()->get());
  if (!current) return current.takeError();

  if (auto error = wrapper::DnnSetStream(handle.get(), stream.get()))
    return error;

  wrapper::Pointer<const void> alpha_ptr(&alpha, handle->platform());
  wrapper::Pointer<const void> beta_ptr(&beta, handle->platform());

  return wrapper::DnnPoolingForward(*current, handle.get(), pooling_desc.get(),
                                    alpha_ptr, x_desc.get(), x.pointer(),
                                    beta_ptr, y_desc.get(), y.pointer());
}

static Error DnnPoolingBackward(
    const GpuDnnHandle& handle, const GpuStream& stream,
    const wrapper::OwningDnnPoolingDescriptor& pooling_desc, float alpha,
    const GpuDnnTensorDesc& y_desc, const GpuBuffer& y,
    const GpuDnnTensorDesc& dy_desc, const GpuBuffer& dy,
    const GpuDnnTensorDesc& x_desc, const GpuBuffer& x, float beta,
    const GpuDnnTensorDesc& dx_desc, const GpuBuffer& dx) {
  auto current = wrapper::CtxSetCurrent(handle.context()->get());
  if (!current) return current.takeError();

  if (auto error = wrapper::DnnSetStream(handle.get(), stream.get()))
    return error;

  auto platform = handle->platform();
  wrapper::Pointer<const void> alpha_ptr(&alpha, platform);
  wrapper::Pointer<const void> beta_ptr(&beta, platform);

  switch (platform) {
    case wrapper::Platform::CUDA:
      return wrapper::CudnnPoolingBackward(
          *current, handle.get(), pooling_desc.get(), alpha_ptr, y_desc.get(),
          y.pointer(), dy_desc.get(), dy.pointer(), x_desc.get(), x.pointer(),
          beta_ptr, dx_desc.get(), dx.pointer());
    case wrapper::Platform::ROCm: {
      auto workspace_size_bytes =
          MiopenPoolingGetWorkSpaceSize(pooling_desc.get(), y_desc.get());
      if (!workspace_size_bytes) return workspace_size_bytes.takeError();
      auto workspace = wrapper::MemAlloc(*current, *workspace_size_bytes);
      if (!workspace) return workspace.takeError();
      return wrapper::MiopenPoolingBackward(
          *current, handle.get(), pooling_desc.get(), alpha_ptr, y_desc.get(),
          y.pointer(), dy_desc.get(), dy.pointer(), x_desc.get(), x.pointer(),
          beta_ptr, dx_desc.get(), dx.pointer(), workspace->get());
    }
    default:
      return MakeStringError("Unknown platform.");
  }
}

static uint64_t DnnConvolutionAlgorithm(Attribute<uint64_t> algo) {
  return *algo;
}

static Error DnnConvolutionForward(
    const GpuDnnHandle& handle, const GpuStream& stream,
    const GpuDnnTensorDesc& x_desc, const GpuBuffer& x,
    const wrapper::OwningDnnFilterDescriptor& w_desc, const GpuBuffer& w,
    const wrapper::OwningDnnConvolutionDescriptor& conv_desc, uint64_t algo,
    const GpuBuffer& work_space, const GpuDnnTensorDesc& y_desc,
    const GpuBuffer& y,
    // Needs to be sorted alphabetically by attribute name!
    Attribute<int32_t> scale_type_attr) {
  auto current = wrapper::CtxSetCurrent(handle.context()->get());
  if (!current) return current.takeError();

  if (auto error = wrapper::DnnSetStream(handle.get(), stream.get()))
    return error;

  auto scale_type = wrapper::DnnDataType::FromOpaqueValue(*scale_type_attr);
  auto algo_dnn = wrapper::DnnConvFwdAlgo::FromOpaqueValue(algo);
  return wrapper::DnnConvolutionForward(
      *current, handle.get(), scale_type, x_desc.get(), x.pointer(),
      w_desc.get(), w.pointer(), conv_desc.get(), algo_dnn,
      work_space.pointer(), work_space.size(), y_desc.get(), y.pointer());
}

static Error DnnConvolutionBackwardData(
    const GpuDnnHandle& handle, const GpuStream& stream,
    const wrapper::OwningDnnFilterDescriptor& w_desc, const GpuBuffer& w,
    const GpuDnnTensorDesc& dy_desc, const GpuBuffer& dy,
    const wrapper::OwningDnnConvolutionDescriptor& conv_desc, uint64_t algo,
    const GpuBuffer& work_space, const GpuDnnTensorDesc& dx_desc,
    const GpuBuffer& dx,
    // Needs to be sorted alphabetically by attribute name!
    Attribute<int32_t> scale_type_attr) {
  auto current = wrapper::CtxSetCurrent(handle.context()->get());
  if (!current) return current.takeError();

  if (auto error = wrapper::DnnSetStream(handle.get(), stream.get()))
    return error;

  auto scale_type = wrapper::DnnDataType::FromOpaqueValue(*scale_type_attr);
  auto algo_dnn = wrapper::DnnConvBwdDataAlgo::FromOpaqueValue(algo);
  return wrapper::DnnConvolutionBackwardData(
      *current, handle.get(), scale_type, w_desc.get(), w.pointer(),
      dy_desc.get(), dy.pointer(), conv_desc.get(), algo_dnn,
      work_space.pointer(), work_space.size(), dx_desc.get(), dx.pointer());
}

static Error DnnConvolutionBackwardFilter(
    const GpuDnnHandle& handle, const GpuStream& stream,
    const GpuDnnTensorDesc& x_desc, const GpuBuffer& x,
    const GpuDnnTensorDesc& dy_desc, const GpuBuffer& dy,
    const wrapper::OwningDnnConvolutionDescriptor& conv_desc, uint64_t algo,
    const GpuBuffer& work_space,
    const wrapper::OwningDnnFilterDescriptor& dw_desc, const GpuBuffer& dw,
    // Needs to be sorted alphabetically by attribute name!
    Attribute<int32_t> scale_type_attr) {
  auto current = wrapper::CtxSetCurrent(handle.context()->get());
  if (!current) return current.takeError();

  if (auto error = wrapper::DnnSetStream(handle.get(), stream.get()))
    return error;

  auto scale_type = wrapper::DnnDataType::FromOpaqueValue(*scale_type_attr);
  auto algo_dnn = wrapper::DnnConvBwdFilterAlgo::FromOpaqueValue(algo);
  return wrapper::DnnConvolutionBackwardFilter(
      *current, handle.get(), scale_type, x_desc.get(), x.pointer(),
      dy_desc.get(), dy.pointer(), conv_desc.get(), algo_dnn,
      work_space.pointer(), work_space.size(), dw_desc.get(), dw.pointer());
}

// This is CUDA specific kernel, there is no ROCm counterpart.
static Error CudnnConvolutionBiasActivationForward(
    const GpuDnnHandle& handle, const GpuStream& stream, AsyncValue* alpha1,
    const GpuDnnTensorDesc& x_desc, const GpuBuffer& x,
    const wrapper::OwningDnnFilterDescriptor& w_desc, const GpuBuffer& w,
    const wrapper::OwningDnnConvolutionDescriptor& conv_desc, uint64_t algo,
    const GpuBuffer& work_space, AsyncValue* alpha2,
    const GpuDnnTensorDesc& z_desc, const GpuBuffer& z,
    const GpuDnnTensorDesc& bias_desc, const GpuBuffer& bias,
    const wrapper::OwningDnnActivationDescriptor& activation_desc,
    const GpuDnnTensorDesc& y_desc, const GpuBuffer& y,
    // Needs to be sorted alphabetically by attribute name!
    Attribute<int32_t> scale_type_attr) {
  auto current = wrapper::CtxSetCurrent(handle.context()->get());
  if (!current) return current.takeError();

  if (auto error = wrapper::DnnSetStream(handle.get(), stream.get()))
    return error;

  auto algo_dnn = wrapper::DnnConvFwdAlgo::FromOpaqueValue(algo);
  auto scale_type = wrapper::DnnDataType::FromOpaqueValue(*scale_type_attr);
  if (wrapper::GetDnnDataTypeId(scale_type) == mlir::TypeID::get<double>()) {
    return wrapper::CudnnConvolutionBiasActivationForward(
        *current, handle.get(), &alpha1->get<double>(), x_desc.get(),
        x.pointer(), w_desc.get(), w.pointer(), conv_desc.get(), algo_dnn,
        work_space.pointer(), work_space.size(), &alpha2->get<double>(),
        z_desc.get(), z.pointer(), bias_desc.get(), bias.pointer(),
        activation_desc.get(), y_desc.get(), y.pointer());
  } else {
    return wrapper::CudnnConvolutionBiasActivationForward(
        *current, handle.get(), &alpha1->get<float>(), x_desc.get(),
        x.pointer(), w_desc.get(), w.pointer(), conv_desc.get(), algo_dnn,
        work_space.pointer(), work_space.size(), &alpha2->get<float>(),
        z_desc.get(), z.pointer(), bias_desc.get(), bias.pointer(),
        activation_desc.get(), y_desc.get(), y.pointer());
  }
}

static Expected<cudnn_frontend::Tensor> BuildTensor(
    wrapper::DnnDataType data_type, int64_t ndim, int64_t const* dim,
    int64_t const* strides, char id, wrapper::DnnDataType unvectorized_type,
    bool set_virtual = false) {
  int vector_size, vector_dim;
  std::tie(vector_size, vector_dim) =
      wrapper::GetTensorVectorizedSizeAndDim(data_type);
  if (vector_size == 32) {
    return MakeStringError(
        "cuDNN frontend doesn't support int8x32 at the moment.");
  }

  auto tensor = cudnn_frontend::TensorBuilder()
                    .setDim(ndim, dim)
                    .setStrides(ndim, strides)
                    .setVirtual(set_virtual)
                    .setId(id)
                    .setAlignment(32)
                    .setDataType(unvectorized_type)
                    .setVectorCountAndDimension(vector_size, vector_dim)
                    .build();
  if (tensor.get_status())
    return wrapper::MakeError(tensor.get_status(), tensor.describe().c_str());
  return std::move(tensor);
}

static Expected<cudnn_frontend::ExecutionPlan> BuildExecutionPlan(
    const GpuDnnHandle& handle, cudnn_frontend::OperationGraph op_graph,
    int32_t engine_id, ArrayRef<int64_t> tuning_knob_ids,
    ArrayRef<int64_t> tuning_knob_values) {
  if (tuning_knob_ids.size() != tuning_knob_values.size()) {
    return MakeStringError(
        "Mismatched number of tuning knob types and choices.");
  }

  // Errors encountered when building a cuDNN operation graph are surfaced in an
  // unprecedented and innovative way: they're written into a field of the
  // contained engine object, but then clobbered by the object's move
  // constructor which makes more cuDNN API calls and encounters further errors.
  // The only way to get the actual errors is to peek at them via the returned
  // rvalue reference before actually moving the object to finish its
  // initialization.
  cudnn_frontend::EngineBuilder engine_builder;
  engine_builder.setOperationGraph(op_graph).setGlobalEngineIdx(engine_id);
  auto&& unmoved = engine_builder.build();
  if (unmoved.get_status())
    return wrapper::MakeError(unmoved.get_status(), unmoved.describe().c_str());
  cudnn_frontend::Engine engine = std::move(unmoved);
  if (engine.get_status())
    return wrapper::MakeError(engine.get_status(), engine.describe().c_str());

  std::unordered_map<int64_t, int64_t> tuning_knobs;
  for (int i = 0; i < tuning_knob_ids.size(); ++i) {
    tuning_knobs[tuning_knob_ids[i]] = tuning_knob_values[i];
  }

  for (auto& knob : engine.getSupportedKnobs()) {
    const auto it = tuning_knobs.find(static_cast<int64_t>(knob.getKnobType()));
    if (it != tuning_knobs.end()) {
      knob.setChoice(it->second);
    }
  }

  auto engine_config =
      cudnn_frontend::EngineConfigBuilder().setEngine(engine).build();
  if (engine_config.get_status())
    return wrapper::MakeError(engine_config.get_status(),
                              engine_config.describe().c_str());

  auto execution_plan = cudnn_frontend::ExecutionPlanBuilder()
                            .setHandle(handle.get())
                            .setEngineConfig(engine_config)
                            .build();
  if (execution_plan.get_status())
    return wrapper::MakeError(execution_plan.get_status(),
                              execution_plan.describe().c_str());
  return std::move(execution_plan);
}

Expected<cudnn_frontend::ExecutionPlan> DnnBuildConvolution(
    const GpuDnnHandle& handle,
    // Needs to be sorted alphabetically by attribute name!
    Attribute<int32_t> backend_type_attr, ArrayAttr conv_dilations,
    Attribute<int32_t> conv_dim_attr, Attribute<int32_t> conv_mode_attr,
    ArrayAttr conv_padding, ArrayAttr conv_strides,
    Attribute<int32_t> engine_id, ArrayAttr filter_dims_attr,
    ArrayAttr filter_strides, ArrayAttr input_dims_attr,
    ArrayAttr input_strides, Attribute<int32_t> input_type_attr,
    ArrayAttr output_dims_attr, ArrayAttr output_strides,
    Attribute<int32_t> output_type_attr, ArrayAttr tuning_knob_ids,
    ArrayAttr tuning_knob_values) {
  auto input_type = wrapper::DnnDataType::FromOpaqueValue(*input_type_attr);
  auto output_type = wrapper::DnnDataType::FromOpaqueValue(*output_type_attr);

  auto unvectorized_input_type =
      wrapper::GetUnvectorizedDnnDataType(input_type);
  auto unvectorized_output_type =
      wrapper::GetUnvectorizedDnnDataType(output_type);

  auto accumulator_type = GetConvAccumulatorType(
      unvectorized_input_type,
      DnnEnvVar<ConvDoFP32ComputationFP16Input>::IsEnabled());

  // x tensor.
  auto input_dims = input_dims_attr.GetValue<int64_t>();
  auto tensor_x = BuildTensor(input_type, input_dims.size(), input_dims.data(),
                              input_strides.GetValue<int64_t>().data(), 'x',
                              unvectorized_input_type);
  if (!tensor_x) return tensor_x.takeError();

  // y tensor.
  auto output_dims = output_dims_attr.GetValue<int64_t>();
  auto tensor_y = BuildTensor(
      output_type, output_dims.size(), output_dims.data(),
      output_strides.GetValue<int64_t>().data(), 'y', unvectorized_output_type);
  if (!tensor_y) return tensor_y.takeError();

  // w tensor.
  auto filter_dims = filter_dims_attr.GetValue<int64_t>();
  auto tensor_w = BuildTensor(
      input_type, filter_dims.size(), filter_dims.data(),
      filter_strides.GetValue<int64_t>().data(), 'w', unvectorized_input_type);
  if (!tensor_w) return tensor_w.takeError();

  // conv_desc.
  cudnnConvolutionMode_t conv_mode =
      wrapper::DnnConvolutionMode::FromOpaqueValue(*conv_mode_attr);
  int conv_dim = *conv_dim_attr;
  auto conv_desc =
      cudnn_frontend::ConvDescBuilder()
          .setComputePrecision(accumulator_type)
          .setMathMode(conv_mode)
          .setNDims(conv_dim)
          .setStrides(conv_dim, conv_strides.GetValue<int64_t>().data())
          .setPrePadding(conv_dim, conv_padding.GetValue<int64_t>().data())
          .setPostPadding(conv_dim, conv_padding.GetValue<int64_t>().data())
          .setDilation(conv_dim, conv_dilations.GetValue<int64_t>().data())
          .build();
  if (conv_desc.get_status())
    return wrapper::MakeError(conv_desc.get_status(),
                              conv_desc.describe().c_str());

  double alpha = 1.0;
  double beta = 0.0;

  // CUDNN Operation
  auto backend_type =
      static_cast<cudnnBackendDescriptorType_t>(*backend_type_attr);
  auto op = cudnn_frontend::OperationBuilder(backend_type)
                .setxDesc(*tensor_x)
                .setyDesc(*tensor_y)
                .setwDesc(*tensor_w)
                .setcDesc(conv_desc)
                .setAlpha(alpha)
                .setBeta(beta)
                .build();
  if (op.get_status())
    return wrapper::MakeError(op.get_status(), op.describe().c_str());

  // CUDNN OperationGraph
  std::array<cudnn_frontend::Operation const*, 1> ops = {&op};
  auto op_graph = cudnn_frontend::OperationGraphBuilder()
                      .setHandle(handle.get())
                      .setOperationGraph(ops.size(), ops.data())
                      .build();
  if (op_graph.get_status())
    return wrapper::MakeError(op_graph.get_status(),
                              op_graph.describe().c_str());
  return BuildExecutionPlan(handle, std::move(op_graph), *engine_id,
                            tuning_knob_ids.GetValue<int64_t>(),
                            tuning_knob_values.GetValue<int64_t>());
}

Error DnnRunConvolution(const GpuDnnHandle& handle, const GpuStream& stream,
                        const cudnn_frontend::ExecutionPlan& execution_plan,
                        const GpuBuffer& input, const GpuBuffer& output,
                        const GpuBuffer& filter, const GpuBuffer& workspace) {
  auto current = wrapper::CtxSetCurrent(handle.context()->get());
  if (!current) return current.takeError();

  if (auto error = wrapper::DnnSetStream(handle.get(), stream.get()))
    return error;

  auto platform = handle->platform();
  void* data_ptrs[] = {input.pointer().raw(platform),
                       output.pointer().raw(platform),
                       filter.pointer().raw(platform)};
  int64_t uids[] = {'x', 'y', 'w'};
  auto variant_pack =
      cudnn_frontend::VariantPackBuilder()
          .setWorkspacePointer(workspace.pointer().raw(platform))
          .setDataPointers(3, data_ptrs)
          .setUids(3, uids)
          .build();
  if (variant_pack.get_status())
    return wrapper::MakeError(variant_pack.get_status(),
                              variant_pack.describe().c_str());

  return wrapper::CudnnBackendExecute(*current, handle.get(),
                                      execution_plan.get_raw_desc(),
                                      variant_pack.get_raw_desc());
}

Expected<cudnn_frontend::ExecutionPlan> DnnBuildFusedConvolution(
    const GpuDnnHandle& handle, double alpha, double alpha2,
    // Needs to be sorted alphabetically by attribute name!
    Attribute<int32_t> activation_mode_attr,
    Attribute<int32_t> backend_type_attr, ArrayAttr bias_dims_attr,
    ArrayAttr bias_strides, Attribute<int32_t> bias_type_attr,
    ArrayAttr conv_dilations, Attribute<int32_t> conv_dim_attr,
    Attribute<int32_t> conv_mode_attr, ArrayAttr conv_padding,
    ArrayAttr conv_strides, Attribute<int32_t> engine_id,
    ArrayAttr filter_dims_attr, ArrayAttr filter_strides,
    ArrayAttr input_dims_attr, ArrayAttr input_strides,
    Attribute<int32_t> input_type_attr, ArrayAttr output_dims_attr,
    ArrayAttr output_strides_attr, Attribute<int32_t> output_type_attr,
    ArrayAttr tuning_knob_ids, ArrayAttr tuning_knob_values) {
  auto input_type = wrapper::DnnDataType::FromOpaqueValue(*input_type_attr);
  auto output_type = wrapper::DnnDataType::FromOpaqueValue(*output_type_attr);
  auto bias_type = wrapper::DnnDataType::FromOpaqueValue(*bias_type_attr);

  auto unvectorized_input_type =
      wrapper::GetUnvectorizedDnnDataType(input_type);
  auto unvectorized_output_type =
      wrapper::GetUnvectorizedDnnDataType(output_type);
  auto unvectorized_bias_type = wrapper::GetUnvectorizedDnnDataType(bias_type);

  auto accumulator_type = GetConvAccumulatorType(
      unvectorized_input_type,
      DnnEnvVar<ConvDoFP32ComputationFP16Input>::IsEnabled());
  auto activation_type = GetConvActivationType(
      unvectorized_input_type,
      DnnEnvVar<ConvDoFP32ComputationFP16Input>::IsEnabled());

  // CUDNN fused operation supports the pattern in the form of
  // Conv + Add + BiasAdd + Act. Therefore, we need to build a graph of the
  // four ops with their input/output tensor edges:
  // Conv   : input: tensor_x, tensor_w;    output: tensor_conv (virtual)
  // Add    : input: tensor_conv, tensor_z; output: tensor_add (virtual)
  // BiasAdd: input: tensor_add, tensor_b;  output: tensor_bias (virtual)
  // Act    : input: tensor_bias;           output: tensor_y
  auto input_dims = input_dims_attr.GetValue<int64_t>();
  auto tensor_x = BuildTensor(input_type, input_dims.size(), input_dims.data(),
                              input_strides.GetValue<int64_t>().data(), 'x',
                              unvectorized_input_type);
  if (!tensor_x) return tensor_x.takeError();

  auto output_dims = output_dims_attr.GetValue<int64_t>();
  auto output_strides = output_strides_attr.GetValue<int64_t>();
  auto tensor_y =
      BuildTensor(output_type, output_dims.size(), output_dims.data(),
                  output_strides.data(), 'y', unvectorized_output_type);
  if (!tensor_y) return tensor_y.takeError();

  auto tensor_z =
      BuildTensor(output_type, output_dims.size(), &output_dims[0],
                  &output_strides[0], 'z', unvectorized_output_type);
  if (!tensor_z) return tensor_z.takeError();

  auto filter_dims = filter_dims_attr.GetValue<int64_t>();
  auto tensor_w = BuildTensor(
      input_type, filter_dims.size(), filter_dims.data(),
      filter_strides.GetValue<int64_t>().data(), 'w', unvectorized_input_type);
  if (!tensor_w) return tensor_w.takeError();

  auto bias_dims = bias_dims_attr.GetValue<int64_t>();
  auto tensor_b = BuildTensor(input_type, bias_dims.size(), bias_dims.data(),
                              bias_strides.GetValue<int64_t>().data(), 'b',
                              unvectorized_bias_type);
  if (!tensor_b) return tensor_b.takeError();

  auto tensor_conv =
      BuildTensor(output_type, output_dims.size(), &output_dims[0],
                  &output_strides[0], 'C', accumulator_type,
                  /*set_virtual=*/true);
  if (!tensor_conv) return tensor_conv.takeError();

  auto tensor_add =
      BuildTensor(output_type, output_dims.size(), &output_dims[0],
                  &output_strides[0], 'A', activation_type,
                  /*set_virtual=*/true);
  if (!tensor_add) return tensor_add.takeError();

  auto tensor_bias =
      BuildTensor(output_type, output_dims.size(), &output_dims[0],
                  &output_strides[0], 'B', activation_type,
                  /*set_virtual=*/true);
  if (!tensor_bias) return tensor_bias.takeError();

  // conv_desc.
  cudnnConvolutionMode_t conv_mode =
      wrapper::DnnConvolutionMode::FromOpaqueValue(*conv_mode_attr);
  int conv_dim = *conv_dim_attr;
  auto conv_desc =
      cudnn_frontend::ConvDescBuilder()
          .setComputePrecision(accumulator_type)
          .setMathMode(conv_mode)
          .setNDims(conv_dim)
          .setStrides(conv_dim, conv_strides.GetValue<int64_t>().data())
          .setPrePadding(conv_dim, conv_padding.GetValue<int64_t>().data())
          .setPostPadding(conv_dim, conv_padding.GetValue<int64_t>().data())
          .setDilation(conv_dim, conv_dilations.GetValue<int64_t>().data())
          .build();
  if (conv_desc.get_status())
    return wrapper::MakeError(conv_desc.get_status(),
                              conv_desc.describe().c_str());

  // CUDNN Operation
  auto backend_type =
      static_cast<cudnnBackendDescriptorType_t>(*backend_type_attr);
  auto conv_op = cudnn_frontend::OperationBuilder(backend_type)
                     .setxDesc(*tensor_x)
                     .setyDesc(*tensor_conv)
                     .setwDesc(*tensor_w)
                     .setcDesc(conv_desc)
                     .setAlpha(1.0f)
                     .setBeta(0.0f)
                     .build();
  if (conv_op.get_status())
    return wrapper::MakeError(conv_op.get_status(), conv_op.describe().c_str());

  auto add_desc = cudnn_frontend::PointWiseDescBuilder()
                      .setMode(CUDNN_POINTWISE_ADD)
                      .setMathPrecision(activation_type)
                      .build();
  auto add_op = cudnn_frontend::OperationBuilder(
                    CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                    .setxDesc(conv_op.getOutputTensor())
                    .setbDesc(*tensor_z)
                    .setyDesc(*tensor_add)
                    .setpwDesc(add_desc)
                    .setAlpha(alpha)
                    .setAlpha2(alpha2)
                    .build();
  if (add_op.get_status())
    return wrapper::MakeError(add_op.get_status(), add_op.describe().c_str());

  auto bias_add_desc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_ADD)
                           .setMathPrecision(activation_type)
                           .build();

  // If the activation is the identity function, then the bias-add is the last
  // op, and it writes to the output, tensor_y.  Otherwise, it writes to the
  // "virtual tensor" (temp buffer) tensor_bias, to which we apply the
  // activation.
  auto activation_mode =
      static_cast<cudnnActivationMode_t>(*activation_mode_attr);
  auto& bias_out_desc =
      activation_mode == CUDNN_ACTIVATION_IDENTITY ? *tensor_y : *tensor_bias;
  auto bias_add_op = cudnn_frontend::OperationBuilder(
                         CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                         .setxDesc(add_op.getOutputTensor())
                         .setbDesc(*tensor_b)
                         .setyDesc(bias_out_desc)
                         .setpwDesc(bias_add_desc)
                         .build();
  if (bias_add_op.get_status())
    return wrapper::MakeError(bias_add_op.get_status(),
                              bias_add_op.describe().c_str());

  // CUDNN OperationGraph
  llvm::SmallVector<cudnn_frontend::Operation const*, 4> ops = {
      &conv_op, &add_op, &bias_add_op};

  llvm::Optional<cudnn_frontend::PointWiseDesc_v8> act_desc;
  llvm::Optional<cudnn_frontend::Operation_v8> act_op;
  switch (activation_mode) {
    case CUDNN_ACTIVATION_IDENTITY:
      break;
    case CUDNN_ACTIVATION_RELU:
      act_desc.emplace(cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_RELU_FWD)
                           .setMathPrecision(activation_type)
                           .build());
      if (act_desc->get_status())
        return wrapper::MakeError(act_desc->get_status(),
                                  act_desc->describe().c_str());
      act_op.emplace(cudnn_frontend::OperationBuilder(
                         CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                         .setxDesc(bias_add_op.getOutputTensor())
                         .setyDesc(*tensor_y)
                         .setpwDesc(*act_desc)
                         .build());
      if (act_op->get_status())
        return wrapper::MakeError(act_op->get_status(),
                                  act_op->describe().c_str());
      ops.push_back(&*act_op);
      break;
    default:
      return MakeStringError("Unimplemented activation mode");
  }

  auto op_graph = cudnn_frontend::OperationGraphBuilder()
                      .setHandle(handle.get())
                      .setOperationGraph(ops.size(), ops.data())
                      .build();
  if (op_graph.get_status())
    return wrapper::MakeError(op_graph.get_status(),
                              op_graph.describe().c_str());
  return BuildExecutionPlan(handle, std::move(op_graph), *engine_id,
                            tuning_knob_ids.GetValue<int64_t>(),
                            tuning_knob_values.GetValue<int64_t>());
}

Error DnnRunFusedConvolution(
    const GpuDnnHandle& handle, const GpuStream& stream,
    const cudnn_frontend::ExecutionPlan& execution_plan, const GpuBuffer& input,
    const GpuBuffer& output, const GpuBuffer& filter,
    const GpuBuffer& side_input, const GpuBuffer& bias,
    const GpuBuffer& workspace) {
  auto current = wrapper::CtxSetCurrent(handle.context()->get());
  if (!current) return current.takeError();

  if (auto error = wrapper::DnnSetStream(handle.get(), stream.get()))
    return error;

  auto platform = handle->platform();
  void* data_ptrs[] = {
      input.pointer().raw(platform), output.pointer().raw(platform),
      filter.pointer().raw(platform), side_input.pointer().raw(platform),
      bias.pointer().raw(platform)};
  int64_t uids[] = {'x', 'y', 'w', 'z', 'b'};
  auto variant_pack =
      cudnn_frontend::VariantPackBuilder()
          .setWorkspacePointer(workspace.pointer().raw(platform))
          .setDataPointers(5, data_ptrs)
          .setUids(5, uids)
          .build();
  if (variant_pack.get_status())
    return wrapper::MakeError(variant_pack.get_status(),
                              variant_pack.describe().c_str());

  return wrapper::CudnnBackendExecute(*current, handle.get(),
                                      execution_plan.get_raw_desc(),
                                      variant_pack.get_raw_desc());
}

void RegisterGpuDnnKernels(KernelRegistry* kernel_reg) {
  kernel_reg->AddKernel("tfrt_gpu.dnn.create", TFRT_KERNEL(DnnCreate));
  kernel_reg->AddKernel("tfrt_gpu.dnn.create_convolution_descriptor",
                        TFRT_KERNEL(DnnCreateConvolutionDescriptor));
  kernel_reg->AddKernel("tfrt_gpu.dnn.create_filter_descriptor",
                        TFRT_KERNEL(DnnCreateFilterDescriptor));
  kernel_reg->AddKernel("tfrt_gpu.dnn.create_pooling_descriptor",
                        TFRT_KERNEL(DnnCreatePoolingDescriptor));
  kernel_reg->AddKernel("tfrt_gpu.dnn.create_tensor_descriptor",
                        TFRT_KERNEL(DnnCreateTensorDescriptor));
  kernel_reg->AddKernel("tfrt_gpu.dnn.create_activation_descriptor",
                        TFRT_KERNEL(DnnCreateActivationDescriptor));
  kernel_reg->AddKernel("tfrt_gpu.dnn.pooling_forward",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(DnnPoolingForward));
  kernel_reg->AddKernel("tfrt_gpu.dnn.pooling_backward",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(DnnPoolingBackward));
  kernel_reg->AddKernel("tfrt_gpu.dnn.convolution_forward_algorithm",
                        TFRT_KERNEL(DnnConvolutionAlgorithm));
  kernel_reg->AddKernel("tfrt_gpu.dnn.convolution_backward_data_algorithm",
                        TFRT_KERNEL(DnnConvolutionAlgorithm));
  kernel_reg->AddKernel("tfrt_gpu.dnn.convolution_backward_filter_algorithm",
                        TFRT_KERNEL(DnnConvolutionAlgorithm));
  kernel_reg->AddKernel("tfrt_gpu.dnn.convolution_forward",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(DnnConvolutionForward));
  kernel_reg->AddKernel(
      "tfrt_gpu.dnn.convolution_backward_data",
      TFRT_KERNEL_WITH_CHAIN_RESULT(DnnConvolutionBackwardData));
  kernel_reg->AddKernel(
      "tfrt_gpu.dnn.convolution_backward_filter",
      TFRT_KERNEL_WITH_CHAIN_RESULT(DnnConvolutionBackwardFilter));
  kernel_reg->AddKernel(
      "tfrt_gpu.dnn.convolution_bias_activation_forward",
      TFRT_KERNEL_WITH_CHAIN_RESULT(CudnnConvolutionBiasActivationForward));
  kernel_reg->AddKernel("tfrt_gpu.dnn.build_convolution",
                        TFRT_KERNEL(DnnBuildConvolution));
  kernel_reg->AddKernel("tfrt_gpu.dnn.run_convolution",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(DnnRunConvolution));
  kernel_reg->AddKernel("tfrt_gpu.dnn.build_fused_convolution",
                        TFRT_KERNEL(DnnBuildFusedConvolution));
  kernel_reg->AddKernel("tfrt_gpu.dnn.run_fused_convolution",
                        TFRT_KERNEL_WITH_CHAIN_RESULT(DnnRunFusedConvolution));
}

}  // namespace gpu
}  // namespace tfrt
