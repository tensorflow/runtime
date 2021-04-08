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

// This file defines the tensor kernels for resnet.

#include <cmath>

#include "../../kernels/cpu_kernels.h"
#include "tfrt/common/compat/eigen/eigen_kernel.h"
#include "tfrt/cpu/ops/test/cpu_ops_and_kernels.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/string_util.h"
#include "tfrt/tensor/dense_host_tensor_view.h"

using ::Eigen::Index;
using ::tfrt::compat::UnaryEigenKernelAsync;

namespace tfrt {

template <typename T>
static void MaxPool2D(ArgumentView<MutableDHTIndexableView<T, 4>> input,
                      ArgumentView<MutableDHTIndexableView<T, 4>> output,
                      Argument<Chain> chain_in, Result<Chain> chain_out,
                      StringAttribute padding,
                      ArrayAttribute<uint32_t> pool_size,
                      ArrayAttribute<uint32_t> strides,
                      KernelErrorHandler handler) {
  // shape_input has format (batch_size, height, width, channel_num)
  const auto& shape_input = input->FixedShape();
  // shape_output has format (batch_size, height, width, channel_num)
  const auto& shape_output = output->FixedShape();

  if (pool_size.size() != 2) {
    handler.ReportError("MaxPool2D expects pool_size to have 2 elements");
    return;
  }

  if (strides.size() != 2) {
    handler.ReportError("MaxPool2D expects strides to have 2 elements");
    return;
  }

  // padding for upper, bottom, left and right
  int padding_numbers[4] = {0, 0, 0, 0};

  if (padding.str() == "same") {
    int total_padding_height = pool_size[0] - strides[0];
    if (shape_input[1] % strides[0] != 0) {
      total_padding_height = pool_size[0] - (shape_input[1] % strides[0]);
    }
    int total_padding_width = pool_size[1] - strides[1];
    if (shape_input[2] % strides[1] != 0) {
      total_padding_width = pool_size[1] - (shape_input[2] % strides[1]);
    }

    padding_numbers[0] = static_cast<int>(total_padding_height / 2.0);
    padding_numbers[1] = static_cast<int>(total_padding_height / 2.0 + 0.5);
    padding_numbers[2] = static_cast<int>(total_padding_width / 2.0);
    padding_numbers[3] = static_cast<int>(total_padding_width / 2.0 + 0.5);
  } else if (padding.str() != "valid") {
    handler.ReportError("MaxPool2D padding '", padding.str(),
                        "' is not recognized");
    return;
  }

  typename MutableDHTIndexableView<T, 4>::FixedShapeType expected_output_shape(
      {shape_input[0],
       (shape_input[1] + padding_numbers[0] + padding_numbers[1] -
        pool_size[0]) /
               strides[0] +
           1,
       (shape_input[2] + padding_numbers[2] + padding_numbers[3] -
        pool_size[1]) /
               strides[1] +
           1,
       shape_input[3]});

  if (shape_output != expected_output_shape) {
    handler.ReportError("MaxPool2D output shape ", shape_output,
                        " does not match the expected output shape ",
                        expected_output_shape);
    return;
  }

  for (int i = 0, e = shape_output[0]; i < e; i++) {
    for (int j = 0, e = shape_output[3]; j < e; j++) {
      for (int k = 0, e = shape_output[1]; k < e; k++) {
        for (int l = 0, e = shape_output[2]; l < e; l++) {
          T value = std::numeric_limits<T>::min();
          for (int x = 0, e = pool_size[0]; x < e; x++) {
            for (int y = 0; y < pool_size[1]; y++) {
              int pos_1 = k * strides[0] + x - padding_numbers[0];
              int pos_2 = l * strides[1] + y - padding_numbers[2];
              if (pos_1 < 0 || pos_1 >= shape_input[1] || pos_2 < 0 ||
                  pos_2 >= shape_input[2])
                continue;
              value = std::max(value, input->ElementAt(i, pos_1, pos_2, j));
            }
          }
          output->ElementAt(i, k, l, j) = value;
        }
      }
    }
  }

  chain_out.Set(chain_in);
}

template <typename T>
static AsyncValueRef<Chain> GlobalAveragePool(
    const DenseHostTensor& input, DenseHostTensor* output, Chain chain_in,
    const ExecutionContext& exec_ctx) {
  std::array<int32_t, 2> reduction_indices({1, 2});

  return cpu::Mean<T>(input, reduction_indices, output, exec_ctx);
}

template <typename T>
static void Flatten(ArgumentView<MutableDHTArrayView<T>> input,
                    ArgumentView<MutableDHTArrayView<T>> output,
                    Argument<Chain> chain_in, Result<Chain> chain_out,
                    KernelErrorHandler handler,
                    const ExecutionContext& exec_ctx) {
  // shape_input has format (batch_size, height, width, in_channel_num)
  const auto& shape_input = input->Shape();
  // shape_output has format (batch_size, length)
  const auto& shape_output = output->Shape();

  if (shape_input.GetNumElements() != shape_output.GetNumElements() ||
      shape_input.GetDimensionSize(0) != shape_output.GetDimensionSize(0)) {
    handler.ReportError("Flatten output shape ", shape_output,
                        " does not match the input shape ", shape_input);
    return;
  }

  std::copy(input->data(), input->data() + input->NumElements(),
            output->data());
  chain_out.Set(chain_in);
}

template <typename T>
static void ZeroPadding(ArgumentView<MutableDHTIndexableView<T, 4>> input,
                        ArgumentView<MutableDHTIndexableView<T, 4>> output,
                        Argument<Chain> chain_in, Result<Chain> chain_out,
                        ArrayAttribute<uint32_t> padding,
                        KernelErrorHandler handler,
                        const ExecutionContext& exec_ctx) {
  // shape_input has format (batch_size, height, width, in_channel_num)
  const auto& shape_input = input->FixedShape();
  const auto& shape_output = output->FixedShape();

  if (padding.size() != 2) {
    handler.ReportError("ZeroPadding expects padding of length 2");
    return;
  }

  // shape_output has format (batch_size, height, width, out_channel_num)
  typename MutableDHTIndexableView<T, 4>::FixedShapeType expected_output_shape(
      {shape_input[0], shape_input[1] + 2 * padding[0],
       shape_input[2] + 2 * padding[1], shape_input[3]});

  if (shape_output != expected_output_shape) {
    handler.ReportError("ZeroPadding output shape ", shape_output,
                        " does not match the expected output shape ",
                        expected_output_shape);
    return;
  }

  for (int i = 0, e = shape_output[0]; i < e; i++) {
    for (int j = 0, e = shape_output[3]; j < e; j++) {
      for (int k = 0, e = shape_output[1]; k < e; k++) {
        for (int l = 0, e = shape_output[2]; l < e; l++) {
          T value = 0;
          int pos_1 = k - padding[0];
          int pos_2 = l - padding[1];
          if (pos_1 >= 0 && pos_1 < shape_input[1] && pos_2 >= 0 &&
              pos_2 < shape_input[2]) {
            value = input->ElementAt(i, pos_1, pos_2, j);
          }
          output->ElementAt(i, k, l, j) = value;
        }
      }
    }
  }

  chain_out.Set(chain_in);
}

static void SoftMaxInPlace(ArgumentView<MutableDHTArrayView<float>> A,
                           Argument<Chain> in_chain, Result<Chain> out_chain,
                           KernelErrorHandler handler,
                           const ExecutionContext& exec_ctx) {
  float max_value = std::numeric_limits<float>::min();
  for (const auto value : A->Elements()) {
    max_value = std::max(max_value, value);
  }

  float sum = 0;
  for (auto& value : A->Elements()) {
    value = exp(value);
    sum += value;
  }

  for (auto& value : A->Elements()) {
    value /= sum;
  }

  out_chain.Set(in_chain);
}

// Computes output = output - gradient * lr.
static AsyncValueRef<Chain> GradientDescent(
    Argument<DenseHostTensor> gradient,
    ArgumentView<MutableDHTArrayView<float>> lr,
    Argument<DenseHostTensor> output, const ExecutionContext& exec_ctx) {
  if (lr->NumElements() != 1) {
    return EmitErrorAsync(exec_ctx,
                          "GradientDescent lr should have only one element");
  }

  float learning_rate = lr->Elements()[0];

  auto fn = [learning_rate](auto& a, auto& b) { return b - a * learning_rate; };
  return UnaryEigenKernelAsync<float, float>(gradient.get(), &output.get(),
                                             std::move(fn), exec_ctx);
}

// Computes output -= input.
template <typename T>
static AsyncValueRef<Chain> ElementwiseSubtractInPlace(
    Argument<DenseHostTensor> input, Argument<DenseHostTensor> output,
    const ExecutionContext& exec_ctx) {
  auto fn = [](auto& a, auto& b) { return b - a; };
  return UnaryEigenKernelAsync<T, T>(input.get(), &output.get(), std::move(fn),
                                     exec_ctx);
}

template <typename T>
static void TensorTranspose(ArgumentView<MutableDHTIndexableView<T, 2>> input,
                            ArgumentView<MutableDHTIndexableView<T, 2>> output,
                            Argument<Chain> in_chain, Result<Chain> out_chain,
                            KernelErrorHandler handler,
                            const ExecutionContext& exec_ctx) {
  const auto& shape_output = output->FixedShape();
  const auto& shape_input = input->FixedShape();

  if (shape_output[1] != shape_input[0] || shape_output[0] != shape_input[1]) {
    handler.ReportError("TensorTranspose output shape ", shape_output,
                        " does not match the input shape ", shape_input);
    return;
  }

  for (int i = 0, end = shape_output[0]; i < end; i++) {
    for (int j = 0, end = shape_output[1]; j < end; j++) {
      output->ElementAt(i, j) = input->ElementAt(j, i);
    }
  }

  out_chain.Set(in_chain);
}

static void MeanAxisZero(ArgumentView<MutableDHTIndexableView<float, 2>> input,
                         ArgumentView<MutableDHTIndexableView<float, 1>> output,
                         Argument<Chain> chain_in, Result<Chain> chain_out,
                         KernelErrorHandler handler,
                         const ExecutionContext& exec_ctx) {
  const auto& shape_input = input->FixedShape();
  const auto& shape_output = output->FixedShape();

  if (shape_input[1] != shape_output[0]) {
    handler.ReportError("MeanAxisZero output shape ", shape_output,
                        " does not match the input shape ", shape_input);
    return;
  }

  for (int i = 0, e = shape_input[1]; i < e; i++) {
    float sum = 0.0;
    for (int j = 0, e = shape_input[0]; j < e; j++) {
      sum += input->ElementAt(j, i);
    }
    output->ElementAt(i) = sum / shape_input[0];
  }

  chain_out.Set(chain_in);
}

static void Broadcast2D(ArgumentView<MutableDHTIndexableView<float, 2>> input,
                        ArgumentView<MutableDHTIndexableView<float, 4>> output,
                        Argument<Chain> chain_in, Result<Chain> chain_out,
                        KernelErrorHandler handler,
                        const ExecutionContext& exec_ctx) {
  const auto& shape_input = input->FixedShape();
  const auto& shape_output = output->FixedShape();

  if (shape_input[0] != shape_output[0] && shape_input[1] != shape_output[3]) {
    handler.ReportError("Broadcast2D output shape ", shape_output,
                        " does not match the input shape ", shape_input);
    return;
  }

  for (int i = 0, e = shape_output[0]; i < e; i++) {
    for (int j = 0, e = shape_output[1]; j < e; j++) {
      for (int k = 0, e = shape_output[2]; k < e; k++) {
        for (int l = 0, e = shape_output[3]; l < e; l++) {
          output->ElementAt(i, j, k, l) = input->ElementAt(i, l);
        }
      }
    }
  }

  chain_out.Set(chain_in);
}

void RegisterResNetTensorKernels(KernelRegistry* registry) {
  registry->AddKernel("tfrt_test.max_pooling_2d.f32",
                      TFRT_KERNEL(MaxPool2D<float>));
  registry->AddKernel("tfrt_test.global_average_pooling.f32",
                      TFRT_KERNEL(GlobalAveragePool<float>));
  registry->AddKernel("tfrt_test.flatten.f32", TFRT_KERNEL(Flatten<float>));
  registry->AddKernel("tfrt_test.zero_padding.f32",
                      TFRT_KERNEL(ZeroPadding<float>));
  registry->AddKernel("tfrt_test.softmax_inplace.f32",
                      TFRT_KERNEL(SoftMaxInPlace));
  registry->AddKernel("tfrt_test.gradient_descent.f32",
                      TFRT_KERNEL(GradientDescent));
  registry->AddKernel("tfrt_test.subtract_inplace.f32",
                      TFRT_KERNEL(ElementwiseSubtractInPlace<float>));
  registry->AddKernel("tfrt_test.tensor_transpose.f32",
                      TFRT_KERNEL(TensorTranspose<float>));
  registry->AddKernel("tfrt_test.mean_axis_zero.f32",
                      TFRT_KERNEL(MeanAxisZero));
  registry->AddKernel("tfrt_test.broadcast_2d.f32", TFRT_KERNEL(Broadcast2D));
}

}  // namespace tfrt
