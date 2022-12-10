/*
 * Copyright 2021 The TensorFlow Runtime Authors
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

#include "tfrt/jitrt/arguments.h"

#include "llvm/ADT/ArrayRef.h"
#include "tfrt/dtype/dtype.h"
#include "tfrt/support/error_util.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "third_party/tensorflow/compiler/xla/runtime/arguments.h"

namespace tfrt {
namespace jitrt {

using xla::PrimitiveType;
using xla::runtime::MemrefDesc;

static PrimitiveType ToPrimitiveType(DType dtype) {
  switch (dtype) {
    // Unsigned integer types.
    case DType::UI8:
      return PrimitiveType::U8;
    case DType::UI16:
      return PrimitiveType::U16;
    case DType::UI32:
      return PrimitiveType::U32;
    case DType::UI64:
      return PrimitiveType::U64;

    // Signed integer types.
    case DType::I1:
      return PrimitiveType::PRED;
    case DType::I8:
      return PrimitiveType::S8;
    case DType::I16:
      return PrimitiveType::S16;
    case DType::I32:
      return PrimitiveType::S32;
    case DType::I64:
      return PrimitiveType::S64;

    // Floating point types.
    case DType::F16:
      return PrimitiveType::F16;
    case DType::F32:
      return PrimitiveType::F32;
    case DType::F64:
      return PrimitiveType::F64;
    case DType::BF16:
      return PrimitiveType::BF16;

    // Complex types.
    case DType::Complex64:
      return PrimitiveType::C64;
    case DType::Complex128:
      return PrimitiveType::C128;

    default:
      LOG(FATAL) << "Unsupported data type: " << dtype;
  }
}

Expected<MemrefDesc> ConvertTensorToMemrefDesc(const Tensor& tensor) {
  if (auto* dht = dyn_cast<DenseHostTensor>(&tensor)) {
    return MemrefDesc(
        dht->shape().GetRank(), ToPrimitiveType(dht->dtype()),
        const_cast<void*>(dht->data()), 0, [&](auto sizes, auto strides) {
          MutableArrayRef<int64_t> sizes_ref(sizes.data(), sizes.size());
          MutableArrayRef<int64_t> strides_ref(strides.data(), strides.size());
          dht->shape().GetDimensions(sizes_ref);
          dht->shape().GetStrides(strides_ref);
        });
  }

  return MakeStringError("unsupported tensor type: ", tensor.tensor_type());
}

}  // namespace jitrt
}  // namespace tfrt
