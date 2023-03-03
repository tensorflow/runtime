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

// Collates list of all TF DNN operations.

#include <numeric>
#include <optional>
#include <unordered_map>

#include "dnn_ops_cu.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "matmul_op.h"
#include "pad_op_noncuda.h"
#include "tfrt/common/ops/tf/dnn_ops_util.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/dtype/dtype.h"
#include "tfrt/gpu/core_runtime/gpu_dispatch_context.h"
#include "tfrt/gpu/core_runtime/gpu_op_registry.h"
#include "tfrt/gpu/core_runtime/gpu_op_utils.h"
#include "tfrt/gpu/tensor/dense_gpu_tensor.h"
#include "tfrt/gpu/wrapper/cudnn_wrapper.h"
#include "tfrt/gpu/wrapper/dnn_wrapper.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/string_util.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/tensor_metadata.h"
#include "tfrt/tensor/tensor_shape.h"

namespace tfrt {
namespace gpu {

static llvm::Expected<cudnnDataType_t> ToCudnnDataType(DType kind) {
  switch (kind) {
    case DType::F16:
      return CUDNN_DATA_HALF;
    case DType::F32:
      return CUDNN_DATA_FLOAT;
    case DType::F64:
      return CUDNN_DATA_DOUBLE;
    default:
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Unsupported type");
  }
}

struct TensorDescriptorData {
  cudnnDataType_t dtype;
  llvm::SmallVector<int, 4> dimensions;
  llvm::SmallVector<int, 4> strides;
};
static bool operator==(const TensorDescriptorData& left,
                       const TensorDescriptorData& right) {
  return left.dtype == right.dtype && left.dimensions == right.dimensions &&
         left.strides == right.strides;
}

static llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                     const TensorDescriptorData& data) {
  os << "TensorDescriptor<dtype=" << data.dtype << ", dimensions=["
     << Join(data.dimensions, ", ") << "]"
     << ", strides=[" << Join(data.strides, ", ") << "]>";
  return os;
}

static llvm::Expected<TensorDescriptorData> GetTensorDescriptorData(
    DType dtype, llvm::ArrayRef<Index> dimensions, ChannelOrder channel_order) {
  TensorDescriptorData result;
  TFRT_ASSIGN_OR_RETURN(result.dtype, ToCudnnDataType(dtype));

  // Expand to at least rank 4.
  auto rank = std::max<int>(4, dimensions.size());
  result.dimensions.resize(rank - dimensions.size(), 1);

  for (auto dim : dimensions)
    result.dimensions.push_back(static_cast<int>(dim));

  llvm::SmallVector<Index, 4> transpose;
  transpose.reserve(rank);
  for (int i = rank - 1; i >= 0; --i) transpose.push_back(i);
  if (channel_order == ChannelOrder::ChannelLast)
    RotateRight(llvm::MutableArrayRef<Index>(transpose).drop_back());

  result.strides.resize(rank);
  int stride = 1;
  for (auto i : transpose) {
    result.strides[i] = stride;
    stride *= result.dimensions[i];
  }
  return result;
}

static llvm::Expected<wrapper::OwningDnnTensorDescriptor>
CreateTensorDescriptor(const TensorDescriptorData& data) {
  TFRT_ASSIGN_OR_RETURN(auto descriptor,
                        wrapper::CudnnCreateTensorDescriptor());
  if (auto error = wrapper::CudnnSetTensorDescriptor(
          descriptor.get(), data.dtype, data.dimensions, data.strides))
    return std::move(error);
  return std::move(descriptor);
}

static llvm::Expected<wrapper::OwningDnnFilterDescriptor>
CreateFilterDescriptor(cudnnDataType_t dtype, ChannelOrder channel_order,
                       llvm::ArrayRef<int> dimensions) {
  TFRT_ASSIGN_OR_RETURN(auto descriptor,
                        wrapper::CudnnCreateFilterDescriptor());
  auto layout = [&] {
    switch (channel_order) {
      case ChannelOrder::ChannelFirst:
        return CUDNN_TENSOR_NCHW;
      case ChannelOrder::ChannelLast:
        return CUDNN_TENSOR_NHWC;
    }
  }();
  if (auto error = wrapper::CudnnSetFilterDescriptor(descriptor.get(), dtype,
                                                     layout, dimensions))
    return std::move(error);
  return std::move(descriptor);
}

static llvm::Error CheckPadding(const WindowedOutputData& data) {
  for (int i = 0; i < data.paddings_before.size(); ++i) {
    auto padding_before = data.paddings_before[i];
    auto padding_after = data.paddings_after[i];
    if (padding_before != padding_after && padding_before + 1 != padding_after)
      return MakeStringError("Padding not supported by cuDNN");
  }
  return llvm::Error::success();
}

static auto ToIntVec(const llvm::ArrayRef<Index> array) {
  return llvm::SmallVector<int, 2>(array.begin(), array.end());
}

static auto AllocateBuffer(GpuDispatchContext* dctx, const DType& dtype,
                           const TensorShape& shape) {
  return GpuBuffer::Allocate(dctx->allocator(),
                             shape.GetNumElements() * GetHostSize(dtype),
                             dctx->stream());
}

namespace {
// The scaling factor parameters 'alpha' and 'beta' of the cudnnTransform* and
// cudnnConvolution* functions are type punned pointers. The storage type is
// double for double output tensors, and float otherwise.
union ScalingFactor {
  ScalingFactor(double value, cudnnDataType_t data_type) {
    if (data_type == CUDNN_DATA_DOUBLE) {
      double_value = value;
    } else {
      float_value = static_cast<float>(value);
    }
  }

  wrapper::Pointer<const void> pointer(wrapper::Platform platform) const {
    return wrapper::Pointer<const void>(this, platform);
  }

 private:
  float float_value;
  double double_value;
};

}  // namespace

static llvm::Expected<FusedBatchNormActivationMode>
ParseFusedBatchNormActivationMode(string_view mode) {
  if (mode == "Identity") return FusedBatchNormActivationMode::kIdentity;
  if (mode == "Relu") return FusedBatchNormActivationMode::kRelu;
  return MakeStringError("Unknown FusedBatchNormActivationMode: ", mode);
}

// A helper function to read the default algorithm for cudnnConvolutionForward.
// Returns None if default is not specified.
static std::optional<cudnnConvolutionFwdAlgo_t>
DefaultCudnnCovolutionForwardAlgorithm() {
  static auto default_algo = []() -> std::optional<cudnnConvolutionFwdAlgo_t> {
    string_view var = std::getenv("TFRT_DEBUG_DEFAULT_CONV_FWD_ALGO");
    if (var.empty()) return std::nullopt;

    cudnnConvolutionFwdAlgo_t default_algo =
        llvm::StringSwitch<cudnnConvolutionFwdAlgo_t>(var)
#define ALGO_CASE(algo) Case(#algo, algo)
            .ALGO_CASE(CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)
            .ALGO_CASE(CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)
            .ALGO_CASE(CUDNN_CONVOLUTION_FWD_ALGO_GEMM)
            .ALGO_CASE(CUDNN_CONVOLUTION_FWD_ALGO_DIRECT)
            .ALGO_CASE(CUDNN_CONVOLUTION_FWD_ALGO_FFT)
            .ALGO_CASE(CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING)
            .ALGO_CASE(CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD)
            .ALGO_CASE(CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED);
#undef ALGO_CASE

    return default_algo;
  }();
  return default_algo;
}

using ConvolutionAlgorithmKey =
    std::tuple<TensorDescriptorData, TensorDescriptorData,
               TensorDescriptorData>;

struct ConvolutionAlgorithmHash {
  size_t operator()(const ConvolutionAlgorithmKey& key) const {
    auto result = Hash(std::get<0>(key));
    Combine(result, Hash(std::get<1>(key)));
    Combine(result, Hash(std::get<2>(key)));
    return result;
  }

 private:
  static size_t Hash(const TensorDescriptorData& data) {
    std::hash<int> hash;
    size_t result = hash(data.dtype);
    for (const auto& dim : data.dimensions) Combine(result, hash(dim));
    for (const auto& stride : data.strides) Combine(result, hash(stride));
    return result;
  }

  static void Combine(size_t& seed, size_t value) {
    // boost::hash_combine.
    seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
};

// T should be cudnnConvolution{Fwd,BwdData,BwdFilter}Algo_t.
template <typename T>
using ConvolutionAlgorithmMap = std::unordered_map<
    ConvolutionAlgorithmKey,
    std::tuple<T /*algo*/, size_t /*workspace_size_bytes*/, cudnnMathType_t>,
    ConvolutionAlgorithmHash>;

auto& GetConvolutionForwardAlgorithmMap() {
  static auto* map = new ConvolutionAlgorithmMap<cudnnConvolutionFwdAlgo_t>();
  return *map;
}

static llvm::Expected<DenseGpuTensor> ComputeConvGpuOp(
    GpuDispatchContext* dctx, const DenseGpuTensor& input,
    const DenseGpuTensor& filter, const OpAttrsRef& attrs,
    const TensorMetadata& result_md) {
  TFRT_ASSIGN_OR_RETURN(auto temp_buffer,
                        AllocateBuffer(dctx, filter.dtype(), filter.shape()));

  TFRT_ASSIGN_OR_RETURN(auto output_buffer,
                        AllocateBuffer(dctx, result_md.dtype, result_md.shape));

  // TODO(b/153682815): Report error if attribute is absent.
  auto padding = attrs.GetStringAsserting("padding");
  auto explicit_paddings = attrs.GetArrayOptional<int>("explicit_paddings");
  auto data_format = attrs.GetStringOptional("data_format");
  auto strides = attrs.GetArrayOptional<Index>("strides");
  auto dilations = attrs.GetArrayOptional<Index>("dilations");

  auto rank = input.shape().GetRank();
  auto channel_order = GetTfChannelOrder(data_format);

  // Determine how to transpose from HWIO to the desired filter layout.
  llvm::SmallVector<Index, 4> transpose;
  transpose.reserve(rank);
  for (int i = 0; i < rank; ++i) transpose.push_back(i);
  switch (channel_order) {
    case ChannelOrder::ChannelFirst:  // HWIO -> OIHW, i.e. {3, 2, 0, 1}
      RotateRight(transpose, 2);
      std::swap(transpose[0], transpose[1]);
      break;
    case ChannelOrder::ChannelLast:  // HWIO -> OHWI, i.e. {3, 0, 1, 2}
      RotateRight(transpose, 1);
      break;
  }

  auto filter_dims_hwio = GetDimensions(filter.shape());
  llvm::SmallVector<Index, 4> filter_dims;  // OIHW or OHWI
  filter_dims.reserve(rank);
  for (auto i : transpose) filter_dims.push_back(filter_dims_hwio[i]);

  auto filter_dims_oihw = filter_dims;
  auto input_dims_nchw = GetDimensions(input.shape());
  auto output_dims_nchw = GetDimensions(result_md.shape);
  if (channel_order == ChannelOrder::ChannelLast) {
    // If layout is NHWC, convert to NCHW.
    RotateRight(llvm::MutableArrayRef<Index>(input_dims_nchw).drop_front());
    RotateRight(llvm::MutableArrayRef<Index>(output_dims_nchw).drop_front());
    RotateRight(llvm::MutableArrayRef<Index>(filter_dims_oihw).drop_front());
  }
  TFRT_ASSIGN_OR_RETURN(
      auto windowed_output_data,
      GetTfWindowedOutputData(input_dims_nchw, filter_dims_oihw, channel_order,
                              padding, explicit_paddings, strides, dilations));

  auto input_ptr = input.buffer().pointer();
  std::optional<DenseGpuTensor> padded_input;
  auto paddings = windowed_output_data.paddings_before;
  // Pad input manually if before and after padding is not the same.
  if (paddings != windowed_output_data.paddings_after) {
    llvm::SmallVector<int64_t, 8> pads_manual;
    pads_manual.reserve(2 * rank);
    // Zero paddings before spatial dimensions.
    pads_manual.resize(channel_order == ChannelOrder::ChannelLast ? 2 : 4, 0);
    for (int i = 0; i < rank - 2; ++i) {
      auto pad_before = windowed_output_data.paddings_before[i];
      auto pad_after = windowed_output_data.paddings_after[i];
      auto difference = pad_before - pad_after;
      pads_manual.push_back(std::max<Index>(0, +difference));
      pads_manual.push_back(std::max<Index>(0, -difference));
      paddings[i] = std::min(pad_before, pad_after);
      // Update input dimensions.
      input_dims_nchw[2 + i] += std::abs(difference);
    }
    // Zero paddings after spatial dimensions.
    pads_manual.resize(2 * rank, 0);

    DenseView pads_manual_view(GetDType<int64_t>(), {rank, 2},
                               pads_manual.data());
    TFRT_ASSIGN_OR_RETURN(
        auto output_metadata,
        TfPadOutputShape<int64_t>(input.metadata(),
                                  pads_manual_view.GetTensor<int64_t, 2>()));
    TFRT_ASSIGN_OR_RETURN(
        auto pad_output,
        CallGpuPadOp(dctx, input, pads_manual_view, output_metadata));
    input_ptr = pad_output.buffer().pointer();
    padded_input.emplace(std::move(pad_output));
  }

  // If image is channels last and filter is 1x1, we may not need to transpose
  // the filter and evaluate a gemm instead of a convolution.
  auto all_equal_to = [](llvm::ArrayRef<Index> array, Index value) {
    return array.front() == value && llvm::all_equal(array);
  };
  if (channel_order == ChannelOrder::ChannelLast &&
      input_dims_nchw[1] == filter_dims_oihw[1] &&  // No grouped convolutions.
      all_equal_to(llvm::ArrayRef(filter_dims_oihw).drop_front(2), 1) &&
      all_equal_to(windowed_output_data.strides, 1) &&
      all_equal_to(paddings, 0)) {
    auto batch_count = input_dims_nchw[0];
    auto channel_count = input_dims_nchw[1];
    auto pixel_count =
        std::accumulate(input_dims_nchw.begin() + 2, input_dims_nchw.end(), 1,
                        std::multiplies<Index>());
    auto reshaped_input =
        (padded_input ? *padded_input : input)
            .WithShape(TensorShape({batch_count * pixel_count, channel_count}));
    auto reshaped_filter = filter.WithShape(
        TensorShape(llvm::ArrayRef(filter_dims_hwio).take_back(2)));
    if (auto error = RunCublasGemm(dctx->current_context(), dctx->blas_handle(),
                                   /*transpose_a=*/false, /*transpose_b=*/false,
                                   reshaped_input.value(),
                                   reshaped_filter.value(), output_buffer)) {
      return std::move(error);
    }
    return DenseGpuTensor(
        result_md.shape, result_md.dtype,
        MakeAvailableAsyncValueRef<GpuBuffer>(std::move(output_buffer)));
  }

  TFRT_ASSIGN_OR_RETURN(
      auto input_data,
      GetTensorDescriptorData(input.dtype(), input_dims_nchw, channel_order));
  TFRT_ASSIGN_OR_RETURN(
      auto filter_data,
      GetTensorDescriptorData(filter.dtype(), filter_dims, channel_order));
  TFRT_ASSIGN_OR_RETURN(auto output_data,
                        GetTensorDescriptorData(
                            result_md.dtype, output_dims_nchw, channel_order));

  TFRT_ASSIGN_OR_RETURN(auto input_desc, CreateTensorDescriptor(input_data));
  TFRT_ASSIGN_OR_RETURN(auto filter_desc,
                        CreateFilterDescriptor(filter_data.dtype, channel_order,
                                               ToIntVec(filter_dims_oihw)));
  TFRT_ASSIGN_OR_RETURN(auto output_desc, CreateTensorDescriptor(output_data));

  auto alpha = ScalingFactor(1.0, input_data.dtype);
  auto beta = ScalingFactor(0.0, output_data.dtype);
  auto platform = dctx->dnn_handle().platform();

  // TODO(iga): Make this function take channel_order instead of FORMAT_IOHW.
  if (auto error =
          TransformFilterTensor(dctx->current_context(), dctx->stream(),
                                channel_order, filter, temp_buffer))
    return std::move(error);

  auto conv_dtype = input_data.dtype;
  // Always use mixed precision for fp16.
  if (conv_dtype == CUDNN_DATA_HALF) conv_dtype = CUDNN_DATA_FLOAT;

  TFRT_ASSIGN_OR_RETURN(auto conv_desc,
                        wrapper::CudnnCreateConvolutionDescriptor());
  if (auto error = wrapper::CudnnSetConvolutionDescriptor(
          conv_desc.get(), ToIntVec(paddings),
          ToIntVec(windowed_output_data.strides),
          ToIntVec(windowed_output_data.dilations), CUDNN_CROSS_CORRELATION,
          conv_dtype))
    return std::move(error);

  // Opt-in to use tensor cores. This might be overwritten below.
  if (auto error = wrapper::CudnnSetConvolutionMathType(conv_desc.get(),
                                                        CUDNN_TENSOR_OP_MATH))
    return std::move(error);

  cudnnConvolutionFwdAlgo_t algo;
  size_t workspace_size_bytes = 0;
  GpuBuffer workspace_buffer;

  // TODO(tfrt-devs): Instead of reading default algorithms from an
  // environment variable, we need to pass these options explicitly through op
  // specific interfaces.
  if (auto default_algo = DefaultCudnnCovolutionForwardAlgorithm()) {
    algo = *default_algo;
    TFRT_ASSIGN_OR_RETURN(
        workspace_size_bytes,
        wrapper::CudnnGetConvolutionForwardWorkspaceSize(
            dctx->dnn_handle(), input_desc.get(), filter_desc.get(),
            conv_desc.get(), output_desc.get(), algo));
  } else {
    auto& map = GetConvolutionForwardAlgorithmMap();
    auto key = std::make_tuple(input_data, filter_data, output_data);
    auto it = map.find(key);
    if (it == map.end()) {
      for (size_t mega_bytes : {1024, 128, 16, 0}) {
        workspace_size_bytes = mega_bytes * 1024 * 1024;
        if (workspace_size_bytes == 0) break;
        if (auto workspace_buffer_or_error = GpuBuffer::Allocate(
                dctx->allocator(), workspace_size_bytes, dctx->stream())) {
          workspace_buffer = std::move(*workspace_buffer_or_error);
          break;
        }
      }
      TFRT_ASSIGN_OR_RETURN(
          auto algo_perfs,
          wrapper::CudnnFindConvolutionForwardAlgorithm(
              dctx->current_context(), dctx->dnn_handle(), input_desc.get(),
              input_ptr, filter_desc.get(), temp_buffer.pointer(),
              conv_desc.get(), output_desc.get(), output_buffer.pointer(), 1,
              workspace_buffer.pointer(), workspace_size_bytes));
      const auto& algo_perf = algo_perfs.front();
      it = map.emplace_hint(it, key,
                            std::make_tuple(algo_perf.algo, algo_perf.memory,
                                            algo_perf.mathType));
    }
    algo = std::get<cudnnConvolutionFwdAlgo_t>(it->second);
    workspace_size_bytes = std::get<size_t>(it->second);
    if (auto error = wrapper::CudnnSetConvolutionMathType(
            conv_desc.get(), std::get<cudnnMathType_t>(it->second)))
      return std::move(error);
  }

  TFRT_ASSIGN_OR_RETURN(
      auto workspace_ptr, [&]() -> llvm::Expected<wrapper::Pointer<void>> {
        if (workspace_size_bytes == 0) {
          return wrapper::Pointer<void>(nullptr, platform);
        }
        if (!workspace_buffer ||
            workspace_buffer.size() < workspace_size_bytes) {
          TFRT_ASSIGN_OR_RETURN(
              workspace_buffer,
              GpuBuffer::Allocate(dctx->allocator(), workspace_size_bytes,
                                  dctx->stream()));
        }
        return workspace_buffer.pointer();
      }());

  if (auto error = wrapper::CudnnConvolutionForward(
          dctx->current_context(), dctx->dnn_handle(), &alpha, input_desc.get(),
          input_ptr, filter_desc.get(), temp_buffer.pointer(), conv_desc.get(),
          algo, workspace_ptr, workspace_size_bytes, &beta, output_desc.get(),
          output_buffer.pointer())) {
    return std::move(error);
  }

  return DenseGpuTensor(
      result_md.shape, result_md.dtype,
      MakeAvailableAsyncValueRef<GpuBuffer>(std::move(output_buffer)));
}

static llvm::Expected<DenseGpuTensor> ComputeMaxPoolGpuOp(
    GpuDispatchContext* dctx, const DenseGpuTensor& input,
    const OpAttrsRef& attrs, const TensorMetadata& result_md) {
  TFRT_ASSIGN_OR_RETURN(auto output_buffer,
                        AllocateBuffer(dctx, result_md.dtype, result_md.shape));

  auto padding = attrs.GetStringAsserting("padding");
  auto explicit_paddings = attrs.GetArrayOptional<int>("explicit_paddings");
  auto data_format = attrs.GetStringOptional("data_format");
  auto strides = attrs.GetArrayOptional<Index>("strides");
  auto dilations = attrs.GetArrayOptional<Index>("dilations");
  auto ksize = attrs.GetArrayOptional<Index>("ksize");

  auto rank = input.shape().GetRank();
  auto channel_order = GetTfChannelOrder(data_format);

  auto input_dims_nchw = GetDimensions(input.shape());
  auto output_dims_nchw = GetDimensions(result_md.shape);
  // If layout is NHWC, convert to NCHW.
  if (channel_order == ChannelOrder::ChannelLast) {
    RotateRight(llvm::MutableArrayRef<Index>(input_dims_nchw).drop_front());
    RotateRight(llvm::MutableArrayRef<Index>(output_dims_nchw).drop_front());
  }

  TFRT_ASSIGN_OR_RETURN(
      auto input_data,
      GetTensorDescriptorData(input.dtype(), input_dims_nchw, channel_order));
  TFRT_ASSIGN_OR_RETURN(auto output_data,
                        GetTensorDescriptorData(
                            result_md.dtype, output_dims_nchw, channel_order));

  TFRT_ASSIGN_OR_RETURN(auto input_desc, CreateTensorDescriptor(input_data));
  TFRT_ASSIGN_OR_RETURN(auto output_desc, CreateTensorDescriptor(output_data));

  auto filter_dims = MaybeExpandFilterSizes(ksize, rank, channel_order);

  TFRT_ASSIGN_OR_RETURN(
      auto windowed_output_data,
      GetTfWindowedOutputData(input_dims_nchw, filter_dims, channel_order,
                              padding, explicit_paddings, strides, dilations));
  if (auto error = CheckPadding(windowed_output_data)) return std::move(error);

  TFRT_ASSIGN_OR_RETURN(auto pooling_desc,
                        wrapper::CudnnCreatePoolingDescriptor());
  if (auto error = wrapper::CudnnSetPoolingDescriptor(
          pooling_desc.get(), CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,
          ToIntVec(llvm::ArrayRef(filter_dims).drop_front(2)),
          // Use 'paddings_before' (equal to or one smaller than
          // 'paddings_after') to match TF's behaviour.
          //
          // cuDNN pooling can deal with output sizes and paddings that don't
          // add up by implicitly adding one extra padding row at the end. In
          // that case the filters are placed one pixel further to the left than
          // if we used the larger 'paddings_after'.
          ToIntVec(windowed_output_data.paddings_before),
          ToIntVec(windowed_output_data.strides)))
    return std::move(error);

  auto alpha = ScalingFactor(1.0, input_data.dtype);
  auto beta = ScalingFactor(0.0, output_data.dtype);
  auto platform = dctx->dnn_handle().platform();

  if (auto error = wrapper::CudnnPoolingForward(
          dctx->current_context(), dctx->dnn_handle(), pooling_desc.get(),
          alpha.pointer(platform), input_desc.get(), input.buffer().pointer(),
          beta.pointer(platform), output_desc.get(), output_buffer.pointer()))
    return std::move(error);

  return DenseGpuTensor(
      result_md.shape, result_md.dtype,
      MakeAvailableAsyncValueRef<GpuBuffer>(std::move(output_buffer)));
}

static llvm::Expected<DenseGpuTensor> ComputeSoftMaxGpuOp(
    GpuDispatchContext* dctx, const DenseGpuTensor& input,
    const OpAttrsRef& attrs, const TensorMetadata& result_md) {
  TFRT_ASSIGN_OR_RETURN(auto output_buffer,
                        AllocateBuffer(dctx, result_md.dtype, result_md.shape));

  auto rank = input.shape().GetRank();
  auto channel_order = ChannelOrder::ChannelLast;

  int axis = attrs.GetOptional<int>("axis").value_or(-1);

  // TODO(csigg): pad shape so that axis==1 and use ChannelFirst.
  if (axis != -1 && axis != rank - 1)
    return MakeStringError("Unsupported 'axis' attribute value ", axis);

  auto input_dims_nchw = GetDimensions(input.shape());
  auto output_dims_nchw = GetDimensions(result_md.shape);
  // If layout is NHWC, convert to NCHW.
  if (channel_order == ChannelOrder::ChannelLast) {
    RotateRight(llvm::MutableArrayRef<Index>(input_dims_nchw).drop_front());
    RotateRight(llvm::MutableArrayRef<Index>(output_dims_nchw).drop_front());
  }

  TFRT_ASSIGN_OR_RETURN(
      auto input_data,
      GetTensorDescriptorData(input.dtype(), input_dims_nchw, channel_order));
  TFRT_ASSIGN_OR_RETURN(auto output_data,
                        GetTensorDescriptorData(
                            result_md.dtype, output_dims_nchw, channel_order));

  TFRT_ASSIGN_OR_RETURN(auto input_desc, CreateTensorDescriptor(input_data));
  TFRT_ASSIGN_OR_RETURN(auto output_desc, CreateTensorDescriptor(output_data));

  auto alpha = ScalingFactor(1.0, input_data.dtype);
  auto beta = ScalingFactor(0.0, output_data.dtype);
  auto platform = dctx->dnn_handle().platform();

  if (auto error = wrapper::CudnnSoftmaxForward(
          dctx->current_context(), dctx->dnn_handle(), CUDNN_SOFTMAX_FAST,
          CUDNN_SOFTMAX_MODE_INSTANCE, alpha.pointer(platform),
          input_desc.get(), input.buffer().pointer(), beta.pointer(platform),
          output_desc.get(), output_buffer.pointer()))
    return std::move(error);

  return DenseGpuTensor(
      result_md.shape, result_md.dtype,
      MakeAvailableAsyncValueRef<GpuBuffer>(std::move(output_buffer)));
}

static llvm::Expected<
    std::tuple<DenseGpuTensor, DenseGpuTensor, DenseGpuTensor, DenseGpuTensor,
               DenseGpuTensor, DenseGpuTensor>>
ComputeBatchNormGpuOp(GpuDispatchContext* dctx, const DenseGpuTensor& input,
                      const DenseGpuTensor& scale, const DenseGpuTensor& bias,
                      const DenseGpuTensor& mean,
                      const DenseGpuTensor& variance, const OpAttrsRef& attrs,
                      const TensorMetadata& result_md0,
                      const TensorMetadata& result_md1,
                      const TensorMetadata& result_md2,
                      const TensorMetadata& result_md3,
                      const TensorMetadata& result_md4,
                      const TensorMetadata& result_md5) {
  // TODO(tfrt-devs): use the correct result_md in the code below.
  const auto& result_md = result_md0;

  TFRT_ASSIGN_OR_RETURN(auto output_buffer,
                        AllocateBuffer(dctx, result_md.dtype, result_md.shape));

  auto data_format = attrs.GetStringOptional("data_format");
  auto epsilon = [&]() -> double {
    float eps_float;
    if (attrs.Get("epsilon", &eps_float)) return eps_float;
    double eps_double;
    if (attrs.Get("epsilon", &eps_double)) return eps_double;
    return 1e-5;
  }();

  auto rank = input.shape().GetRank();
  // Note: tf.nn.batch_normalization does not have data_format, but at least in
  // the test we specify it. ChannelList is the default, so this should be ok.
  auto channel_order = GetTfChannelOrder(data_format);

  auto input_dims_nchw = GetDimensions(input.shape());
  auto output_dims_nchw = GetDimensions(result_md.shape);
  // If layout is NHWC, convert to NCHW.
  if (channel_order == ChannelOrder::ChannelLast) {
    RotateRight(llvm::MutableArrayRef<Index>(input_dims_nchw).drop_front());
    RotateRight(llvm::MutableArrayRef<Index>(output_dims_nchw).drop_front());
  }

  auto mean_dims = GetDimensions(mean.shape());
  if (mean_dims.size() == 1) {
    // A one-dimensional mean refers to the channel dimension.
    mean_dims.resize(rank, 1);
    std::swap(mean_dims[0], mean_dims[1]);
  } else if (channel_order == ChannelOrder::ChannelLast) {
    // If layout is NHWC, convert to NCHW.
    RotateRight(llvm::MutableArrayRef<Index>(mean_dims).drop_front());
  }

  TFRT_ASSIGN_OR_RETURN(
      auto input_data,
      GetTensorDescriptorData(input.dtype(), input_dims_nchw, channel_order));
  TFRT_ASSIGN_OR_RETURN(
      auto scale_bias_mean_var_data,
      GetTensorDescriptorData(mean.dtype(), mean_dims, channel_order));
  TFRT_ASSIGN_OR_RETURN(auto output_data,
                        GetTensorDescriptorData(
                            result_md.dtype, output_dims_nchw, channel_order));

  TFRT_ASSIGN_OR_RETURN(auto input_desc, CreateTensorDescriptor(input_data));
  TFRT_ASSIGN_OR_RETURN(auto scale_bias_mean_var_desc,
                        CreateTensorDescriptor(scale_bias_mean_var_data));
  TFRT_ASSIGN_OR_RETURN(auto output_desc, CreateTensorDescriptor(output_data));

  auto alpha = ScalingFactor(1.0, input_data.dtype);
  auto beta = ScalingFactor(0.0, output_data.dtype);
  auto platform = dctx->dnn_handle().platform();

  if (auto error = wrapper::CudnnBatchNormalizationForwardInference(
          dctx->current_context(), dctx->dnn_handle(), CUDNN_BATCHNORM_SPATIAL,
          alpha.pointer(platform), beta.pointer(platform), input_desc.get(),
          input.buffer().pointer(), output_desc.get(), output_buffer.pointer(),
          scale_bias_mean_var_desc.get(), scale.buffer().pointer(),
          bias.buffer().pointer(), mean.buffer().pointer(),
          variance.buffer().pointer(), epsilon))
    return std::move(error);

  // TODO(tfrt-devs): Return correct results for the last 5 outputs.
  // Use a dummy buffer for last 5 outputs.
  TFRT_ASSIGN_OR_RETURN(auto dummy_buffer_allocated,
                        AllocateBuffer(dctx, result_md.dtype, result_md.shape));
  auto dummy_buffer =
      MakeAvailableAsyncValueRef<GpuBuffer>(std::move(dummy_buffer_allocated));

  return std::make_tuple(
      DenseGpuTensor(
          result_md.shape, result_md.dtype,
          MakeAvailableAsyncValueRef<GpuBuffer>(std::move(output_buffer))),
      DenseGpuTensor(result_md.shape, result_md.dtype, dummy_buffer.CopyRef()),
      DenseGpuTensor(result_md.shape, result_md.dtype, dummy_buffer.CopyRef()),
      DenseGpuTensor(result_md.shape, result_md.dtype, dummy_buffer.CopyRef()),
      DenseGpuTensor(result_md.shape, result_md.dtype, dummy_buffer.CopyRef()),
      DenseGpuTensor(result_md.shape, result_md.dtype, dummy_buffer.CopyRef()));
}

static llvm::Expected<
    std::tuple<DenseGpuTensor, DenseGpuTensor, DenseGpuTensor, DenseGpuTensor,
               DenseGpuTensor, DenseGpuTensor>>
FusedBatchNormExOp(GpuDispatchContext* dctx, const DenseGpuTensor& input,
                   const DenseGpuTensor& scale, const DenseGpuTensor& bias,
                   const DenseGpuTensor& mean, const DenseGpuTensor& variance,
                   OptionalOpArg<DenseGpuTensor> side_input,
                   const OpAttrsRef& attrs, const TensorMetadata& result_md0,
                   const TensorMetadata& result_md1,
                   const TensorMetadata& result_md2,
                   const TensorMetadata& result_md3,
                   const TensorMetadata& result_md4,
                   const TensorMetadata& result_md5) {
  const auto& result_md = result_md0;
  TFRT_ASSIGN_OR_RETURN(auto output_buffer,
                        AllocateBuffer(dctx, result_md.dtype, result_md.shape));

  // Process op attributes.
  float epsilon = attrs.GetAsserting<float>("epsilon");
  auto activation_mode_str = attrs.GetStringAsserting("activation_mode");
  auto data_format = attrs.GetStringOptional("data_format");

  ChannelOrder channel_order = GetTfChannelOrder(data_format);
  TFRT_ASSIGN_OR_RETURN(auto activation_mode,
                        ParseFusedBatchNormActivationMode(activation_mode_str));

  // Call Cuda kernel.
  if (auto error = FusedBatchNormEx(dctx->current_context(), dctx->stream(),
                                    channel_order, input, scale, bias, mean,
                                    variance, side_input.get(), epsilon,
                                    activation_mode, output_buffer))
    return std::move(error);

  // TODO(tfrt-devs): Return correct results for the last 5 outputs.
  // Use a dummy buffer for last 5 outputs.
  TFRT_ASSIGN_OR_RETURN(auto dummy_buffer_allocated,
                        AllocateBuffer(dctx, result_md.dtype, result_md.shape));
  auto dummy_buffer =
      MakeAvailableAsyncValueRef<GpuBuffer>(std::move(dummy_buffer_allocated));

  return std::make_tuple(
      DenseGpuTensor(
          result_md.shape, result_md.dtype,
          MakeAvailableAsyncValueRef<GpuBuffer>(std::move(output_buffer))),
      DenseGpuTensor(result_md.shape, result_md.dtype, dummy_buffer.CopyRef()),
      DenseGpuTensor(result_md.shape, result_md.dtype, dummy_buffer.CopyRef()),
      DenseGpuTensor(result_md.shape, result_md.dtype, dummy_buffer.CopyRef()),
      DenseGpuTensor(result_md.shape, result_md.dtype, dummy_buffer.CopyRef()),
      DenseGpuTensor(result_md.shape, result_md.dtype, dummy_buffer.CopyRef()));
}

void RegisterDnnGpuTfOps(GpuOpRegistry* registry) {
  // TODO(fishx): Add attribute names to the registry.
  registry->AddOp(
      "tf.Conv2D", TFRT_GPU_OP(gpu::ComputeConvGpuOp),
      {"padding", "explicit_paddings", "data_format", "strides", "dilations"});
  registry->AddOp("tf.MaxPool", TFRT_GPU_OP(gpu::ComputeMaxPoolGpuOp),
                  {"padding", "explicit_paddings", "data_format", "strides",
                   "dilations", "ksize"});
  registry->AddOp("tf.Softmax", TFRT_GPU_OP(gpu::ComputeSoftMaxGpuOp),
                  {"axis"});
  registry->AddOp("tf.FusedBatchNormV3",
                  TFRT_GPU_OP(gpu::ComputeBatchNormGpuOp),
                  {"data_format", "epsilon"});
  registry->AddOp("tf._FusedBatchNormEx", TFRT_GPU_OP(gpu::FusedBatchNormExOp),
                  {"epsilon", "activation_mode", "data_format"});
}

}  // namespace gpu
}  // namespace tfrt
