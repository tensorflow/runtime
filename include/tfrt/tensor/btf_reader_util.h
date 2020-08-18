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

//===- btf_reader_util.h ----------------------------------------*- C++ -*-===//
//
// This file contains utilities for reading BTF (Binary Tensor Format).
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_TENSOR_BTF_READER_UTIL_H_
#define TFRT_TENSOR_BTF_READER_UTIL_H_

#include <cstdint>
#include <fstream>

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/Support/Error.h"
#include "tfrt/dtype/dtype.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/tensor/btf.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/dense_host_tensor_view.h"
#include "tfrt/tensor/tensor_shape.h"

namespace tfrt {

// Utility function to read n elements of data of type T from the input stream.
template <typename T>
bool ReadStream(std::istream* stream, T* value, size_t n = 1) {
  return stream->read(reinterpret_cast<char*>(value), n * sizeof(T)).good();
}

// The DenseHostTensor parser is kept in this util library because it is used by
// another format.
template <typename DType, size_t Rank>
Expected<DenseHostTensor> ParseDenseHostTensorFromStream(
    std::ifstream* stream, size_t offset, btf::TensorLayout layout,
    HostContext* host) {
  if (layout != btf::TensorLayout::kRMD) {
    return MakeStringError("unexpected tensor layout ", layout);
  }

  std::array<ssize_t, Rank> dims;
  if (!ReadStream(stream, dims.data(), Rank)) {
    return MakeStringError("failed to read tensor dims at offset ", offset);
  }

  auto dht =
      DenseHostTensor::CreateUninitialized<DType>(TensorShape(dims), host);
  if (!dht.hasValue()) {
    return MakeStringError("cannot allocate result tensor");
  }

  using DHTView = MutableDHTIndexableView<DType, Rank>;
  DHTView tensor{dht.getPointer()};

  // This can read a large amount of data from the stream. Depending on the
  // underlying file system implementation, we may need to have a more optimal
  // strategy for reading the file.
  if (!ReadStream(stream, tensor.data(), tensor.NumElements())) {
    return MakeStringError("failed to read tensor data from stream at offset ",
                           offset);
  }
  return std::move(*dht);
}

template <typename ParseTensorTraits>
Expected<typename ParseTensorTraits::TensorTy> ReadTensorFromBTFHelper(
    std::string path, int32_t index, HostContext* host) {
  std::ifstream stream(path, std::ios_base::binary);
  if (!stream) {
    return MakeStringError("failed to open file ", path, " for reading");
  }

  // Read the number of tensors from the file.
  uint64_t num_tensors;
  if (!ReadStream(&stream, &num_tensors)) {
    return MakeStringError("failed to read tensor num_tensors from path ",
                           path);
  }

  if (index >= num_tensors) {
    return MakeStringError("invalid tensor index ", index,
                           " to read tensor from path ", path,
                           " which contains ", num_tensors, " tensors");
  }

  // Read the offset from the target index from the file.
  uint64_t offset;
  // Seek to the position for the offset.
  stream.seekg(sizeof(uint64_t) * (index + 1));
  if (!ReadStream(&stream, &offset)) {
    return MakeStringError("failed to read tensor offset from ", path,
                           " for tensor index ", index);
  }

  // Seek to the beginning of the tensor.
  stream.seekg(offset);

  // Read the tensor header and verify dtype and rank.
  btf::TensorHeader tensor_header;
  if (!ReadStream(&stream, &tensor_header)) {
    return MakeStringError(
        "failed to read tensor header from stream at offset ", offset);
  }

  if (tensor_header.dtype !=
      btf::GetTensorDType(typename ParseTensorTraits::DType())) {
    return MakeStringError(
        "unexpected tensor dtype ", tensor_header.dtype, ". Expected dtype is ",
        btf::GetTensorDType(typename ParseTensorTraits::DType()));
  }

  if (tensor_header.rank != ParseTensorTraits::kRank) {
    // statically casting to uint64_t to work around GCC complaint of not able
    // to bind packed field 'btf::TensorHeader::rank' to 'long unsigned int&'
    return MakeStringError("unexpected tensor rank ",
                           static_cast<uint64_t>(tensor_header.rank),
                           ". Expected rank is ", ParseTensorTraits::kRank);
  }

  return ParseTensorTraits::kParseTensorFn(&stream, offset,
                                           tensor_header.layout, host);
}

// Kernel to read a tensor from an input stream.
// The arguments of the kernel are:
//   argument 0 (std::string): The path of a binary tensor file.
//   argument 1 (int32_t): The index of the tensor in the input file to read.
//
// The return values are:
//   value 0: The tensor object read from the file.
//
// The format of the tensor in the file is as follows:
//
// <num_tensors:uint64_t><offsets:uint64_t[]><TensorRecord_1><TensorRecord_2>...
//
// The format of each TensorRecord is as follows:
//
// <rank:uint64_t><dtype:uint64_t><dims:uint64_t[rank]><tensor_data:dtype[]>
//
template <class ParseTensorTraits>
AsyncValueRef<typename ParseTensorTraits::TensorTy> ReadTensorFromBTF(
    std::string path, int32_t index, const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();

  using ReturnTy = Expected<typename ParseTensorTraits::TensorTy>;
  return host->EnqueueBlockingWork([host, path, index, exec_ctx]() -> ReturnTy {
    auto result = ReadTensorFromBTFHelper<ParseTensorTraits>(path, index, host);
    if (!result) {
      auto diag = EmitError(exec_ctx, result.takeError());
      return MakeStringError(diag.message);
    }
    return result;
  });
}

template <typename DType_, size_t Rank_>
struct ParseDenseHostTensorTraits {
  using DType = DType_;
  static constexpr auto kRank = Rank_;
  using TensorTy = DenseHostTensor;
  static constexpr auto kParseTensorFn =
      ParseDenseHostTensorFromStream<DType_, Rank_>;
};

}  // namespace tfrt

#endif  // TFRT_TENSOR_BTF_READER_UTIL_H_
