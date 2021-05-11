/*
 * Copyright 2020 The TensorFlow Runtime Authors
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

// This file declares serialization and deserialization utils for DenseElement
// attributes.

#ifndef TFRT_SUPPORT_BEF_SERIALIZE_H_
#define TFRT_SUPPORT_BEF_SERIALIZE_H_

#include <cstddef>
#include <cstdint>

#include "llvm/ADT/ArrayRef.h"
#include "tfrt/bef/bef_encoding.h"
#include "tfrt/bef_converter/bef_attr_encoder.h"
#include "tfrt/host_context/host_buffer.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/tensor/dense_view.h"
#include "tfrt/tensor/tensor_metadata.h"

namespace tfrt {

class DenseHostTensor;
class HostContext;
class DenseAttr;

// DenseHostTensor to DenseAttr.
size_t SerializeDenseHostTensorToDenseAttr(const DenseHostTensor& dht,
                                           BefAttrEncoder* encoder);

// DenseAttr to DenseHostTensor.
llvm::Expected<DenseHostTensor> DeserializeDenseHostTensorFromDenseAttr(
    DenseAttr attr, HostContext* host);

TensorMetadata CreateTensorMetadata(const DenseAttr& attr);

DenseView CreateDenseView(const DenseAttr& attr);

std::string SerializeTensorMetadata(const TensorMetadata& md);

llvm::Expected<TensorMetadata> DeserializeTensorMetadata(
    string_view serialized);

llvm::Expected<llvm::SmallVector<RCReference<HostBuffer>, 4>>
SerializeDenseHostTensor(const DenseHostTensor& dht, HostContext* host);

llvm::Expected<DenseHostTensor> DeserializeDenseHostTensor(
    const llvm::SmallVector<RCReference<HostBuffer>, 4>& serialized,
    HostContext* host);

}  // namespace tfrt

#endif  // TFRT_SUPPORT_BEF_SERIALIZE_H_
