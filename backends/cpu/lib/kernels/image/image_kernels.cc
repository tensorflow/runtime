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

// This file implements kernels that process images.

#include "jpeg/jpeg_mem.h"
#include "resize_bilinear_op.h"
#include "tfrt/host_context/async_dispatch.h"
#include "tfrt/host_context/function.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/error_util.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/dense_host_tensor_view.h"
#include "tfrt/tensor/dense_tensor_utils.h"
#include "tfrt/tracing/tracing.h"

namespace tfrt {
namespace image {

// Returns tf.image.decode_jpeg(data, channels=3)
static AsyncValueRef<DenseHostTensor> DecodeJpeg(
    Argument<std::string> data, const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();
  auto output = MakeUnconstructedAsyncValueRef<DenseHostTensor>(host);

  EnqueueWork(exec_ctx, [data = data.ValueRef(), output = output.CopyRef(),
                         exec_ctx] {
    TFRT_TRACE_SCOPE(Default, "DecodeJpeg");
    if (!llvm::StringRef(data.get()).startswith("\xff\xd8\xff")) {
      auto diag = EmitError(exec_ctx, "image does not have jpeg format");
      output.SetError(diag);
      return;
    }
    jpeg::UncompressFlags flags;
    flags.components = 3;
    flags.dct_method = JDCT_IFAST;

    AsyncValueRef<DenseHostTensor> buffer;
    jpeg::Uncompress(
        data.get().data(), data.get().size(), flags, nullptr /* nwarn */,
        [exec_ctx, &buffer](int width, int height, int channels) -> uint8_t* {
          auto tensor = DenseHostTensor::CreateUninitialized<uint8_t>(
              TensorShape({height, width, channels}), exec_ctx.host());
          if (!tensor) {
            buffer = EmitErrorAsync(exec_ctx, "cannot allocate tensor");
            return nullptr;
          }
          buffer = MakeAvailableAsyncValueRef<DenseHostTensor>(
              exec_ctx.host(), std::move(*tensor));
          return static_cast<uint8_t*>(buffer.get().data());
        });
    if (buffer.IsError()) {
      output.SetError(buffer.GetError());
      return;
    }
    output.emplace(std::move(buffer.get()));
  });

  return output;
}

// IDEA(donglin): allocate tensor buffer outside this kernel
// Returns tf.compat.v1.image.resize(input, [height, width])
static AsyncValueRef<DenseHostTensor> ResizeBilinear(
    Argument<DenseHostTensor> input, Index height, Index width,
    const ExecutionContext& exec_ctx) {
  using ReturnTy = Expected<DenseHostTensor>;
  return EnqueueWork(
      exec_ctx,
      [input = input.ValueRef(), height, width, exec_ctx]() -> ReturnTy {
        TFRT_TRACE_SCOPE(Default, "ResizeBilinear");
        const TensorShape& shape = input->shape();
        if (shape.GetRank() != 3) {
          auto diag = EmitError(exec_ctx, "input tensor shape must be 3");
          return MakeStringError(diag.message);
        }

        Index batch_size = 1;
        Index input_height = shape.GetDimensionSize(0);
        Index input_width = shape.GetDimensionSize(1);
        Index channels = shape.GetDimensionSize(2);
        float height_scale = input_height / static_cast<float>(height);
        float width_scale = input_width / static_cast<float>(width);

        // Create the temporary output tensor with batch_size=1. This follows
        // the same logic in tf.image.resize which adds a batch dimension if the
        // input image does not have the batch dimension. It is easier to port
        // the code from TF to TFRT by following the same logic. And in the
        // future we may also want this kernel to process input image with or
        // without the batch dimension.
        auto dht = DenseHostTensor::CreateUninitialized<float>(
            TensorShape({batch_size, height, width, channels}),
            exec_ctx.host());
        if (!dht) {
          auto diag = EmitError(exec_ctx, "cannot allocate tensor");
          return MakeStringError(diag.message);
        }
        resize_image(input.get(), height_scale, width_scale, *dht);

        // Remove the batch_size dimension before returning the result.
        TensorMetadata output_metadata(GetDType<float>(),
                                       {height, width, channels});
        DenseHostTensor output(output_metadata, dht->ReleaseBuffer());
        return std::move(output);
      });
}

// This is the entrypoint to the library.
void RegisterImageKernels(KernelRegistry* registry) {
  registry->AddKernel("tfrt_test.decode_jpeg", TFRT_KERNEL(DecodeJpeg));
  registry->AddKernel("tfrt_test.resize_bilinear", TFRT_KERNEL(ResizeBilinear));
}

}  // namespace image
}  // namespace tfrt
