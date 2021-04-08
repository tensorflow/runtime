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

// This file defines the metadata functions for the test ops dialect.

#include "tfrt/common/ops/test/metadata_functions.h"
#include "tfrt/core_runtime/op_utils.h"
#include "tfrt/support/error_util.h"
#include "tfrt/tensor/tensor_metadata.h"

namespace tfrt {

// result = test.create_from_scalar(value=V, shape=Shape)
static Expected<TensorMetadata> CreateFromScalarMD(const OpAttrsRef& attrs) {
  ArrayRef<ssize_t> shape;
  if (!attrs.GetArray("shape", &shape))
    return MakeStringError(
        "tfrt_test.create_from_scalar must have 'shape' attribute");

  auto* value = attrs.GetRaw("value");
  if (!value || value->IsArray())
    return MakeStringError(
        "tfrt_test.create_from_scalar must have 'value' attribute");

  switch (value->type) {
    default:
      return MakeStringError("tfrt_test.create_from_scalar unsupported dtype");
#define DTYPE_TRIVIAL(ENUM) \
  case OpAttrType::ENUM:    \
    return TensorMetadata(DType(DType::ENUM), shape);
#include "tfrt/dtype/dtype.def"
  }
}

static Expected<TensorMetadata> MatMulMD(const TensorMetadata& lhs,
                                         const TensorMetadata& rhs,
                                         const OpAttrsRef& attrs) {
  if (lhs.dtype != rhs.dtype)
    return MakeStringError("incompatible dtypes for MatMul");

  if (lhs.shape.GetRank() != 2)
    return MakeStringError("argument 0 of matmul op is not a rank-2 tensor");

  if (rhs.shape.GetRank() != 2)
    return MakeStringError("argument 1 of matmul op is not a rank-2 tensor");

  bool transpose_a = attrs.GetAsserting<bool>("transpose_a");
  bool transpose_b = attrs.GetAsserting<bool>("transpose_b");
  std::array<int, 2> dim_pair;
  dim_pair[0] = transpose_a ? 0 : 1;
  dim_pair[1] = transpose_b ? 1 : 0;

  if (lhs.shape.GetDimensionSize(dim_pair[0]) !=
      rhs.shape.GetDimensionSize(dim_pair[1]))
    return MakeStringError("matmul arguments have incompatible shapes");
  return TensorMetadata(lhs.dtype,
                        {lhs.shape.GetDimensionSize(1 - dim_pair[0]),
                         rhs.shape.GetDimensionSize(1 - dim_pair[1])});
}

static TensorMetadata ReluMD(const TensorMetadata& input) { return input; }

static Expected<TensorMetadata> ElementwiseOpMD(const TensorMetadata& lhs,
                                                const TensorMetadata& rhs) {
  if (lhs.dtype != rhs.dtype)
    return MakeStringError("incompatible dtypes for element-wise op");

  if (lhs.shape != rhs.shape)
    return MakeStringError(
        "arguments to element-wise op must have identical shapes");

  return lhs;
}

static Expected<TensorMetadata> BroadcastMD(const TensorMetadata& arg,
                                            const OpAttrsRef& attrs) {
  ArrayRef<ssize_t> shape;
  if (!attrs.GetArray("shape", &shape))
    return MakeStringError("missing 'shape' attribute");

  // TODO(fishx): Support other shapes.
  if (shape.size() != 2)
    return MakeStringError("result shape must be 2 dimensional (for now)");

  TensorShape tensor_shape(shape);
  if (tensor_shape.GetDimensionSize(1) != arg.shape.GetDimensionSize(0)) {
    return MakeStringError("input dimension and target shape mismatch");
  }

  return TensorMetadata(arg.dtype, tensor_shape);
}

static Expected<TensorMetadata> CastMD(const TensorMetadata& arg,
                                       const OpAttrsRef& attrs) {
  string_view type;
  if (!attrs.GetString("type", &type))
    return MakeStringError("missing 'type' attribute");

  // TODO(fishx): Support other types.
  if (type == "f32") {
    return TensorMetadata(GetDType<float>(), arg.shape);
  } else if (type == "f16") {
    return TensorMetadata(DType(DType::F16), arg.shape);
  } else {
    return MakeStringError("only casting to f32/f16 is supported");
  }
}

static Expected<TensorMetadata> ReduceOpShape(const TensorMetadata& arg,
                                              int32_t axis, DType dtype) {
  auto rank = arg.shape.GetRank();
  if (axis >= rank) return MakeStringError("axis must less than input rank");

  SmallVector<ssize_t, 4> result_dims;
  result_dims.resize(rank - 1);
  size_t out_axis = 0;
  for (size_t in_axis = 0; in_axis < rank; ++in_axis) {
    if (in_axis != axis) {
      result_dims[out_axis++] = arg.shape.GetDimensionSize(in_axis);
    }
  }
  return TensorMetadata(dtype, result_dims);
}

static Expected<TensorMetadata> ArgmaxMD(const TensorMetadata& arg,
                                         const OpAttrsRef& attrs) {
  int32_t axis;
  if (!attrs.Get("axis", &axis))
    return MakeStringError("missing 'axis' attribute");

  return ReduceOpShape(arg, axis, GetDType<int32_t>());
}

static Expected<TensorMetadata> ReduceMeanMD(const TensorMetadata& arg,
                                             const OpAttrsRef& attrs) {
  int32_t axis;
  if (!attrs.Get("axis", &axis))
    return MakeStringError("missing 'axis' attribute");

  return ReduceOpShape(arg, axis, arg.dtype);
}

static Expected<TensorMetadata> CreateDenseTensorMD(const OpAttrsRef& attrs) {
  ArrayRef<ssize_t> shape;
  if (!attrs.GetArray("shape", &shape))
    return MakeStringError("missing 'shape' attribute");

  const auto* values_raw = attrs.GetRaw("values");
  if (!values_raw) return MakeStringError("missing 'values' attribute");

  if (!values_raw->IsArray())
    return MakeStringError("'values' attribute should be an array of values");

  TensorShape tensor_shape(shape);
  if (values_raw->array_size != 1 &&
      values_raw->array_size != tensor_shape.GetNumElements()) {
    return MakeStringError(
        "size of 'values' must either be 1 or match num of elements.");
  }

  switch (values_raw->type) {
    default:
      return MakeStringError("tfrt_test.create_dense_tensor unsupported dtype");
#define DTYPE_TRIVIAL(ENUM) \
  case OpAttrType::ENUM:    \
    return TensorMetadata(DType(DType::ENUM), tensor_shape);
#include "tfrt/dtype/dtype.def"
  }
}

// Elementwise add operation.
// result = test.add(lhs, rhs)
Expected<TensorMetadata> TestAddMD(const TensorMetadata& lhs,
                                   const TensorMetadata& rhs) {
  if (lhs.dtype != rhs.dtype)
    return MakeStringError("incompatible dtypes for test.add");

  if (lhs.shape != rhs.shape)
    return MakeStringError("arguments of test.add must have same shape");
  return lhs;
}

// result = unaryop(source).  Result and source have same metadata.
static TensorMetadata UnaryIdentityMD(const TensorMetadata& input) {
  return input;
}

// A simple op for testing OptionalOpArg.
static TensorMetadata TestOptionalArgOpMD(
    const TensorMetadata& input, OptionalOpArg<TensorMetadata> input2) {
  if (input2) {
    return *input2;
  } else {
    return input;
  }
}

// A simple op for testing VariadicOpArg.
static TensorMetadata TestVariadicArgOpMD(
    const TensorMetadata& input, VariadicOpArg<TensorMetadata> input2) {
  if (input2.size() > 0) {
    return input2[0];
  } else {
    return input;
  }
}

static Expected<TensorMetadata> CreateCooTensorMD(
    const TensorMetadata& indices_md, const TensorMetadata& values_md,
    const OpAttrsRef& attrs) {
  if (indices_md.shape.GetRank() != 2)
    return MakeStringError(
        "tfrt_test.create_coo_tensor indices input must be rank 2");
  if (values_md.shape.GetRank() != 1)
    return MakeStringError(
        "tfrt_test.create_coo_tensor values input must be rank 1");
  ArrayRef<ssize_t> shape;
  if (!attrs.GetArray("shape", &shape))
    return MakeStringError(
        "tfrt_test.create_coo_tensor must have 'shape' attribute");
  if (indices_md.shape.GetDimensionSize(1) != shape.size())
    return MakeStringError(
        "each test.create_coo_tensor index must have the same size as shape "
        "attribute");
  return TensorMetadata(values_md.dtype, shape);
}

llvm::ArrayRef<std::pair<llvm::StringRef, OpMetadataFn>>
GetAllTestMetadataFunctions() {
  static auto* md_functions = [] {
    auto* result = new std::vector<std::pair<llvm::StringRef, OpMetadataFn>>;
    result->emplace_back("tfrt_test.create_from_scalar",
                         TFRT_METADATA(CreateFromScalarMD));
    result->emplace_back("tfrt_test.matmul", TFRT_METADATA(MatMulMD));
    result->emplace_back("tfrt_test.relu", TFRT_METADATA(ReluMD));
    result->emplace_back("tfrt_test.equal", TFRT_METADATA(ElementwiseOpMD));
    result->emplace_back("tfrt_test.cast", TFRT_METADATA(CastMD));
    result->emplace_back("tfrt_test.broadcast", TFRT_METADATA(BroadcastMD));
    result->emplace_back("tfrt_test.argmax", TFRT_METADATA(ArgmaxMD));
    result->emplace_back("tfrt_test.reduce_mean", TFRT_METADATA(ReduceMeanMD));
    result->emplace_back("tfrt_test.create_dense_tensor",
                         TFRT_METADATA(CreateDenseTensorMD));
    result->emplace_back("tfrt_test.add", TFRT_METADATA(TestAddMD));
    result->emplace_back("tfrt_test.add.denseonly", TFRT_METADATA(TestAddMD));
    result->emplace_back("tfrt_test.add.denseonly2", TFRT_METADATA(TestAddMD));
    result->emplace_back("tfrt_test.add.denseonly3", TFRT_METADATA(TestAddMD));
    result->emplace_back("tfrt_test.async.noop",
                         TFRT_METADATA(UnaryIdentityMD));
    result->emplace_back("tfrt_test.error.tensor",
                         TFRT_METADATA(UnaryIdentityMD));
    result->emplace_back("tfrt_test.test_optional_arg",
                         TFRT_METADATA(TestOptionalArgOpMD));
    result->emplace_back("tfrt_test.test_variadic_arg",
                         TFRT_METADATA(TestVariadicArgOpMD));
    result->emplace_back("tfrt_test.create_coo_tensor",
                         TFRT_METADATA(CreateCooTensorMD));
    return result;
  }();

  return *md_functions;
}
}  // namespace tfrt
