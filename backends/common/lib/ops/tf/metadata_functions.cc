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

//===- metadata_functions.cc - Metadata functions for TF ops ----*- C++ -*-===//
//
// This file contains Metadata functions for TF ops.
//
//===----------------------------------------------------------------------===//

#include "tfrt/common/ops/tf/metadata_functions.h"

#include <tuple>

#include "tfrt/common/ops/tf/bcast.h"
#include "tfrt/common/ops/tf/dnn_ops_util.h"
#include "tfrt/core_runtime/op_attr_type.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/core_runtime/op_utils.h"
#include "tfrt/support/error_util.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/tensor_metadata.h"
#include "tfrt/tensor/tensor_serialize_utils.h"

namespace tfrt {

static DType OpAttrTypeToDType(OpAttrType type) {
  switch (type) {
    default:
      return DType(DType::Invalid);
#define DTYPE_NUMERIC(ENUM) \
  case OpAttrType::ENUM:    \
    return DType(DType::ENUM);
#include "tfrt/dtype/dtype.def"  // NOLINT
  }
}

// Elementwise binary operation operation.
static Expected<TensorMetadata> TfBinaryOpMd(const TensorMetadata& lhs,
                                             const TensorMetadata& rhs) {
  if (lhs.dtype != rhs.dtype)
    return MakeStringError("incompatible dtypes for test.add");

  // Handle the broadcasting case.
  // A knob can be added to turn off broadcasting.
  TFRT_ASSIGN_OR_RETURN(auto broadcasted_shape,
                        GetBroadcastedShape(lhs.shape, rhs.shape));
  return TensorMetadata(lhs.dtype, broadcasted_shape);
}

static Expected<TensorMetadata> ConstOpMd(const OpAttrsRef& attrs) {
  tfrt::DenseAttr dense_attr;
  if (!attrs.Get("value", &dense_attr)) {
    return MakeStringError("tf.Const needs a `value` dense attribute");
  }
  TensorMetadata md = CreateTensorMetadata(dense_attr);

  OpAttrType dtype;
  if (attrs.Get("dtype", &dtype) && OpAttrTypeToDType(dtype) != md.dtype) {
    return MakeStringError(
        "dtype attribute mismatch with expected tensor dtype");
  }
  return md;
}

// result = unaryop(source).  Result and source have same metadata.
static TensorMetadata UnaryIdentityMd(const TensorMetadata& input) {
  return input;
}

static Expected<TensorMetadata> MatMulMd(const TensorMetadata& a,
                                         const TensorMetadata& b,
                                         VariadicOpArg<TensorMetadata> _,
                                         const OpAttrsRef& attrs) {
  if (a.dtype != b.dtype)
    return MakeStringError("incompatible dtypes for MatMul: In[0]: ", a.dtype,
                           ", In[1]: ", b.dtype);

  if (a.shape.GetRank() != 2)
    return MakeStringError(
        "argument 0 of matmul op is not a rank-2 tensor. Actual rank is ",
        a.shape.GetRank());

  if (b.shape.GetRank() != 2)
    return MakeStringError(
        "argument 1 of matmul op is not a rank-2 tensor. Actual rank is ",
        b.shape.GetRank());

  bool transpose_a;
  if (!attrs.Get("transpose_a", &transpose_a)) {
    return MakeStringError(
        "'transpose_a' attribute is not specified for MatMul op");
  }
  bool transpose_b;
  if (!attrs.Get("transpose_b", &transpose_b)) {
    return MakeStringError(
        "'transpose_b' attribute is not specified for MatMul op");
  }

  int a_matching_dim = transpose_a ? 0 : 1;
  int b_matching_dim = transpose_b ? 1 : 0;

  if (a.shape.GetDimensionSize(a_matching_dim) !=
      b.shape.GetDimensionSize(b_matching_dim))
    return MakeStringError(
        "matmul arguments have incompatible shapes: In[0]: ", a.shape,
        ", In[1]: ", b.shape, ". transpose_a: ", transpose_a,
        ", transpose_b: ", transpose_b);

  int a_remaining_dim = 1 - a_matching_dim;
  int b_remaining_dim = 1 - b_matching_dim;

  return TensorMetadata(a.dtype, {a.shape.GetDimensionSize(a_remaining_dim),
                                  b.shape.GetDimensionSize(b_remaining_dim)});
}

static Expected<TensorMetadata> TfConvOpMd(const TensorMetadata& input,
                                           const TensorMetadata& filter,
                                           const OpAttrsRef& attrs) {
  auto data_format = attrs.GetStringOptional("data_format");
  auto channel_order = GetTfChannelOrder(data_format);

  auto filter_dims = GetDimensions(filter.shape);
  // TF filter is HWIO, convert to OIHW.
  RotateRight(filter_dims, 2);
  std::swap(filter_dims[0], filter_dims[1]);

  auto input_dims_nchw = GetDimensions(input.shape);
  // If input is NHWC, convert to NCHW.
  if (channel_order == ChannelOrder::ChannelLast)
    RotateRight(llvm::MutableArrayRef<ssize_t>(input_dims_nchw).drop_front());

  auto padding = attrs.GetStringAsserting("padding");
  auto explicit_paddings = attrs.GetArrayOptional<int>("explicit_paddings");
  auto strides = attrs.GetArrayOptional<ssize_t>("strides");
  auto dilations = attrs.GetArrayOptional<ssize_t>("dilations");
  TFRT_ASSIGN_OR_RETURN(
      auto windowed_output_data,
      GetTfWindowedOutputData(input_dims_nchw, filter_dims, channel_order,
                              padding, explicit_paddings, strides, dilations));

  auto output_dims_nchw = windowed_output_data.output_dims;
  // If input is NHWC, convert output to NHWC as well.
  if (channel_order == ChannelOrder::ChannelLast) {
    RotateRight(llvm::MutableArrayRef<ssize_t>(output_dims_nchw).drop_front(),
                output_dims_nchw.size() - 2);
  }

  return TensorMetadata(input.dtype, output_dims_nchw);
}

static Expected<TensorMetadata> TfShapeOpMd(const TensorMetadata& input,
                                            const OpAttrsRef& attrs) {
  auto out_type = attrs.GetAsserting<OpAttrType>("out_type");
  auto dtype = OpAttrTypeToDType(out_type);

  if (dtype.kind() != DType::I32 && dtype.kind() != DType::I64)
    return MakeStringError("Unsupported `out_type` value: ", dtype.kind());

  return TensorMetadata(dtype, ArrayRef<ssize_t>{input.shape.GetRank()});
}

static Expected<TensorMetadata> TfZerosLikeOpMd(const TensorMetadata& input,
                                                const OpAttrsRef& attrs) {
  auto out_type = attrs.GetAsserting<OpAttrType>("T");
  auto dtype = OpAttrTypeToDType(out_type);

  return TensorMetadata(dtype, input.shape);
}

static Expected<TensorMetadata> TfMaxPoolOpMd(const TensorMetadata& input,
                                              const OpAttrsRef& attrs) {
  auto padding = attrs.GetStringAsserting("padding");
  auto explicit_paddings = attrs.GetArrayOptional<int>("explicit_paddings");
  auto data_format = attrs.GetStringOptional("data_format");
  auto strides = attrs.GetArrayOptional<ssize_t>("strides");
  auto dilations = attrs.GetArrayOptional<ssize_t>("dilations");
  auto ksize = attrs.GetArrayOptional<ssize_t>("ksize");
  auto channel_order = GetTfChannelOrder(data_format);

  auto input_dims_nchw = GetDimensions(input.shape);
  // If input is NHWC, convert to NCHW.
  if (channel_order == ChannelOrder::ChannelLast)
    RotateRight(llvm::MutableArrayRef<ssize_t>(input_dims_nchw).drop_front());

  auto filter_dims =
      MaybeExpandFilterSizes(ksize, input.shape.GetRank(), channel_order);

  if (filter_dims[0] != 1 || filter_dims[1] != 1)
    return MakeStringError("Expected ksize 'NC' elements to be 1");

  // Number of output channels as used by GetTfWindowedOutputData.
  filter_dims[0] = input_dims_nchw[1];

  TFRT_ASSIGN_OR_RETURN(
      auto windowed_output_data,
      GetTfWindowedOutputData(input_dims_nchw, filter_dims, channel_order,
                              padding, explicit_paddings, strides, dilations));

  auto output_dims_nchw = windowed_output_data.output_dims;
  // If input is NHWC, convert output to NHWC as well.
  if (channel_order == ChannelOrder::ChannelLast) {
    RotateRight(llvm::MutableArrayRef<ssize_t>(output_dims_nchw).drop_front(),
                output_dims_nchw.size() - 2);
  }

  return TensorMetadata(input.dtype, output_dims_nchw);
}

static Expected<TensorMetadata> TfBiasAddOpMd(const TensorMetadata& value,
                                              const TensorMetadata& bias,
                                              const OpAttrsRef& attrs) {
  if (value.dtype != bias.dtype)
    return MakeStringError("incompatible dtypes for tf.BiasAdd");

  string_view data_format;
  bool has_data_format_attr = attrs.GetString("data_format", &data_format);

  if (has_data_format_attr && data_format != "NHWC") {
    return MakeStringError("invalid data format. Currently only support NHWC");
  }

  if (bias.shape.GetRank() != 1) {
    return MakeStringError("bias must be 1-D");
  }

  if (bias.shape.GetDimensionSize(0) !=
      value.shape.GetDimensionSize(value.shape.GetRank() - 1)) {
    return MakeStringError(
        "bias must has the size of the last dimension of value");
  }

  return value;
}

static Expected<std::tuple<TensorMetadata, TensorMetadata, TensorMetadata,
                           TensorMetadata, TensorMetadata, TensorMetadata>>
TfBatchNormOpMd(const TensorMetadata& input, const TensorMetadata& mean,
                const TensorMetadata& variance, const TensorMetadata& bias,
                const TensorMetadata& scale, const OpAttrsRef& attrs) {
  bool is_training;
  if (attrs.Get("is_training", &is_training) && is_training)
    return MakeStringError("BatchNorm training not currently supported");
  if (mean.shape != variance.shape || variance.shape != bias.shape ||
      bias.shape != scale.shape)
    return MakeStringError(
        "Mean, variance, bias, and scale are expected to have the same shape");
  // TODO(tfrt-devs): Return correct metadata.
  return std::make_tuple(input, TensorMetadata(), TensorMetadata(),
                         TensorMetadata(), TensorMetadata(), TensorMetadata());
}

static Expected<std::tuple<TensorMetadata, TensorMetadata, TensorMetadata,
                           TensorMetadata, TensorMetadata, TensorMetadata>>
TfFusedBatchNormExOpMd(const TensorMetadata& input, const TensorMetadata& mean,
                       const TensorMetadata& variance,
                       const TensorMetadata& bias, const TensorMetadata& scale,
                       OptionalOpArg<TensorMetadata> side_input,
                       const OpAttrsRef& attrs) {
  bool is_training;
  if (attrs.Get("is_training", &is_training) && is_training)
    return MakeStringError("BatchNorm training not currently supported");
  if (side_input && input.shape != side_input->shape)
    return MakeStringError(
        "Input and side_input are expected to have the same shape");
  if (mean.shape != variance.shape || variance.shape != bias.shape ||
      bias.shape != scale.shape)
    return MakeStringError(
        "Mean, variance, bias, and scale are expected to have the same shape");
  // TODO(tfrt-devs): Return correct metadata.
  return std::make_tuple(input, TensorMetadata(), TensorMetadata(),
                         TensorMetadata(), TensorMetadata(), TensorMetadata());
}

Expected<TensorMetadata> CallTfPadOutputShape(const TensorMetadata& input,
                                              const DenseView& paddings) {
  // TODO(iga): Add a non-templated DHTIndexableView class to avoid these cases.
  switch (paddings.dtype().kind()) {
    case DType::I32:
      return TfPadOutputShape(input, paddings.GetTensor<int32_t, 2>());
    case DType::I64:
      return TfPadOutputShape(input, paddings.GetTensor<int64_t, 2>());
    default:
      return MakeStringError(
          "tf.Pad paddings type must be either int32 or int64. Actual type "
          "is: ",
          paddings.dtype());
  }
}

static Expected<TensorMetadata> TfPadOpMdImpl(const TensorMetadata& input,
                                              const DenseView& paddings,
                                              const OpAttrsRef& attrs) {
  constexpr int kMaxDims = 8;
  if (input.shape.GetRank() > kMaxDims)
    return MakeStringError(
        "tf.Pad input rank over ", kMaxDims,
        " is not supported. Given value: ", input.shape.GetRank());

  if (paddings.shape().GetRank() != 2 ||
      paddings.shape().GetDimensionSize(1) != 2) {
    return MakeStringError(
        "tf.Pad paddings must be a matrix with 2 columns. Actual shape: ",
        paddings.shape());
  }

  if (input.shape.GetRank() != paddings.shape().GetDimensionSize(0)) {
    return MakeStringError(
        "For tf.Pad operation, the first dimension of paddings must equal to "
        "the rank of inputs. Input shape: ",
        input.shape, ". Paddings shape: ", paddings.shape());
  }

  Expected<TensorMetadata> result_md = CallTfPadOutputShape(input, paddings);
  if (!result_md) {
    return result_md.takeError();
  }

  return result_md;
}

static Expected<TensorMetadata> TfPadOpMd(
    const TensorMetadata& input,
    const TensorMetadata& /* paddings input is ignored */,
    const OpAttrsRef& attrs) {
  // TODO(tfrt-devs): read paddings from dense host tensor.
  llvm::SmallVector<int32_t, 8> default_paddings(8, 0);
  auto channel_order = GuessChannelOrder(input.shape);
  if (!channel_order) return MakeStringError("Could not guess channel order.");
  auto spatial_offset = *channel_order == ChannelOrder::ChannelLast ? 2 : 4;
  std::fill_n(default_paddings.begin() + spatial_offset, 4, 3);

  DenseView default_paddings_view(GetDType<int32_t>(), {4, 2},
                                  default_paddings.data());

  return TfPadOpMdImpl(input, default_paddings_view, attrs);
}

static Expected<TensorMetadata> TfPadOpFoldedMd(const TensorMetadata& input,
                                                const OpAttrsRef& attrs) {
  DenseAttr dense_attr;
  if (!attrs.Get("paddings", &dense_attr)) {
    return MakeStringError("tf.Pad is missing a required attribute `paddings`");
  }
  DenseView paddings = CreateDenseView(dense_attr);
  return TfPadOpMdImpl(input, paddings, attrs);
}

static Expected<TensorMetadata> TfTransposeOpMdImpl(const TensorMetadata& input,
                                                    ArrayRef<ssize_t> perm,
                                                    const OpAttrsRef& attrs) {
  if (perm.size() != input.shape.GetRank()) {
    return MakeStringError(
        "tf.Transpose `perm` must size must match input rank");
  }

  llvm::SmallVector<ssize_t, 4> output_dims;
  for (int i = 0; i < input.shape.GetRank(); ++i) {
    output_dims.push_back(input.shape.GetDimensionSize(perm[i]));
  }

  return TensorMetadata(input.dtype, output_dims);
}

static Expected<TensorMetadata> TfTransposeOpMd(const TensorMetadata& input,
                                                const TensorMetadata& /*perm*/,
                                                const OpAttrsRef& attrs) {
  static constexpr ssize_t default_perm[] = {0, 3, 1, 2};
  return TfTransposeOpMdImpl(input, default_perm, attrs);
}

static Expected<TensorMetadata> TfTransposeOpFoldedMd(
    const TensorMetadata& input, const OpAttrsRef& attrs) {
  DenseAttr perm_attr;
  if (!attrs.Get("perm", &perm_attr)) {
    return MakeStringError("tf.Transpose needs a `perm` dense attribute");
  }

  DenseView perm_view = CreateDenseView(perm_attr);
  assert(perm_view.shape().GetRank() == 1);

  SmallVector<ssize_t, 4> perm;
  switch (perm_view.dtype().kind()) {
    case DType::I32: {
      auto value = perm_view.GetFlat<int32_t>();
      perm.assign(value.begin(), value.end());
      break;
    }
    case DType::I64: {
      auto value = perm_view.GetFlat<int64_t>();
      perm.assign(value.begin(), value.end());
      break;
    }
    default:
      llvm_unreachable("unsupported dtype for perm in tf.Transpose");
  }

  return TfTransposeOpMdImpl(input, perm, attrs);
}

static Expected<TensorMetadata> TfCastOpMd(const TensorMetadata& input,
                                           const OpAttrsRef& attrs) {
  // TODO(b/149063226): Change it back to OpAttrType once we implement fp16.
  OpAttrType dest_type;
  if (!attrs.Get("DstT", &dest_type)) {
    return MakeStringError("cannot get destination type");
  }
  TensorMetadata result_md = input;
  result_md.dtype = OpAttrTypeToDType(dest_type);
  return result_md;
}

template <typename ReductionIndexT>
static Expected<TensorMetadata> TfMeanOpMdImpl(
    const TensorMetadata& input, ArrayRef<ReductionIndexT> reduction_indices,
    const OpAttrsRef& attrs) {
  static_assert(std::is_same<ReductionIndexT, int32_t>::value ||
                    std::is_same<ReductionIndexT, int64_t>::value,
                "Reduction index should be I32 or I64.");
  llvm::SmallVector<bool, 4> reduced_dim(input.shape.GetRank(), false);
  for (auto reduction_index : reduction_indices) {
    if (reduction_index < 0 || reduction_index >= input.shape.GetRank()) {
      return MakeStringError(
          "tf.Mean reduction index must be in [0, input_rank) range");
    }
    if (reduced_dim[reduction_index]) {
      return MakeStringError("tf.Mean reduction indices must be unique");
    }

    reduced_dim[reduction_index] = true;
  }

  llvm::SmallVector<ssize_t, 4> output_dims;
  for (int i = 0; i < input.shape.GetRank(); ++i) {
    if (!reduced_dim[i]) output_dims.push_back(input.shape.GetDimensionSize(i));
  }

  return TensorMetadata(input.dtype, output_dims);
}

static Expected<TensorMetadata> TfMeanOpMd(
    const TensorMetadata& input, const TensorMetadata& /* reduction_indices */,
    const OpAttrsRef& attrs) {
  // TODO(tfrt-devs): Read reduction_indices from the tensor argument.
  auto channel_order = GuessChannelOrder(input.shape);
  if (!channel_order) return MakeStringError("Could not guess channel order.");
  auto spatial_offset = *channel_order == ChannelOrder::ChannelLast ? 1 : 2;
  llvm::SmallVector<int32_t, 2> default_reduction_indices = {
      spatial_offset, spatial_offset + 1};

  return TfMeanOpMdImpl<int32_t>(input, default_reduction_indices, attrs);
}

static Expected<TensorMetadata> TfMeanOpFoldedMd(const TensorMetadata& input,
                                                 const OpAttrsRef& attrs) {
  DenseAttr dense_attr;
  if (!attrs.Get("reduction_indices", &dense_attr)) {
    return MakeStringError(
        "tf.Mean needs a `reduction_indices` dense attribute");
  }

  DenseView reduction_indices = CreateDenseView(dense_attr);
  assert(reduction_indices.shape().GetRank() == 1);

  switch (reduction_indices.dtype().kind()) {
    case DType::I32:
      return TfMeanOpMdImpl(input, reduction_indices.GetFlat<int32_t>(), attrs);
    case DType::I64:
      return TfMeanOpMdImpl(input, reduction_indices.GetFlat<int64_t>(), attrs);
    default:
      llvm_unreachable("unsupported dtype for reduction_indices in tf.Mean");
  }
}

llvm::ArrayRef<std::pair<llvm::StringRef, OpMetadataFn>>
GetAllTFMetadataFunctions() {
  static auto* md_functions = [] {
    auto* result = new std::vector<std::pair<llvm::StringRef, OpMetadataFn>>;
    result->emplace_back("tf.Const", TFRT_METADATA(ConstOpMd));
    result->emplace_back("tf.AddV2", TFRT_METADATA(TfBinaryOpMd));
    result->emplace_back("tf.Tanh", TFRT_METADATA(UnaryIdentityMd));
    result->emplace_back("tf.MatMul", TFRT_METADATA(MatMulMd));
    result->emplace_back("tf._FusedMatMul", TFRT_METADATA(MatMulMd));
    result->emplace_back("tf.Log", TFRT_METADATA(UnaryIdentityMd));
    result->emplace_back("tf.Log1p", TFRT_METADATA(UnaryIdentityMd));
    result->emplace_back("tf.Relu", TFRT_METADATA(UnaryIdentityMd));
    result->emplace_back("tf.Conv2D", TFRT_METADATA(TfConvOpMd));
    result->emplace_back("tf.MaxPool", TFRT_METADATA(TfMaxPoolOpMd));
    result->emplace_back("tf.Mean", TFRT_METADATA(TfMeanOpMd));
    result->emplace_back("_tf.Mean", TFRT_METADATA(TfMeanOpFoldedMd));
    result->emplace_back("tf.Mul", TFRT_METADATA(TfBinaryOpMd));
    result->emplace_back("tf.RealDiv", TFRT_METADATA(TfBinaryOpMd));
    result->emplace_back("tf.Rsqrt", TFRT_METADATA(UnaryIdentityMd));
    result->emplace_back("tf.Shape", TFRT_METADATA(TfShapeOpMd));
    result->emplace_back("tf.Softmax", TFRT_METADATA(UnaryIdentityMd));
    result->emplace_back("tf.Sigmoid", TFRT_METADATA(UnaryIdentityMd));
    result->emplace_back("tf.LogSoftmax", TFRT_METADATA(UnaryIdentityMd));
    result->emplace_back("tf.Sub", TFRT_METADATA(TfBinaryOpMd));
    result->emplace_back("tf.BiasAdd", TFRT_METADATA(TfBiasAddOpMd));
    result->emplace_back("tf.FusedBatchNormV3", TFRT_METADATA(TfBatchNormOpMd));
    result->emplace_back("tf._FusedBatchNormEx",
                         TFRT_METADATA(TfFusedBatchNormExOpMd));
    result->emplace_back("tf.Pad", TFRT_METADATA(TfPadOpMd));
    result->emplace_back("_tf.Pad", TFRT_METADATA(TfPadOpFoldedMd));
    result->emplace_back("tf.Transpose", TFRT_METADATA(TfTransposeOpMd));
    result->emplace_back("_tf.Transpose", TFRT_METADATA(TfTransposeOpFoldedMd));
    result->emplace_back("tf.Cast", TFRT_METADATA(TfCastOpMd));
    result->emplace_back("tf.ZerosLike", TFRT_METADATA(TfZerosLikeOpMd));
    return result;
  }();

  return *md_functions;
}
}  // namespace tfrt
