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

//===- BTF Inspector Utility ----------------------------------------------===//
//
// This file prints the contents of a BTF file to stdout.

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <iostream>
#include <optional>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "tfrt/host_context/host_buffer.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/tensor/btf.h"
#include "tfrt/tensor/coo_host_tensor.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/tensor_metadata.h"

using tfrt::CooHostTensor;
using tfrt::DenseHostTensor;
using tfrt::DType;
using tfrt::HostBuffer;
using tfrt::RCReference;
using tfrt::TensorMetadata;
using tfrt::TensorShape;
using tfrt::btf::TensorDType;
using tfrt::btf::TensorHeader;
using tfrt::btf::TensorLayout;

namespace {

llvm::cl::opt<std::string> cl_input_filename(  // NOLINT
    llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::Required);

class BtfFile {
 public:
  // Open the given file.
  llvm::Error Open(const char* file) {
    assert(fd_ == -1 && "Open called twice");
    fd_ = open(file, O_RDONLY);
    if (fd_ == -1) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Cannot open file");
    }

    struct stat file_info;
    if (fstat(fd_, &file_info) == -1) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Cannot stat file");
    }
    size_ = file_info.st_size;
    if (size_ < sizeof(int64_t)) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Invalid BTF file: missing tensor count");
    }

    buffer_ = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (buffer_ == MAP_FAILED)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Cannot mmap file");

    return llvm::Error::success();
  }

  ~BtfFile() {
    if (buffer_ != MAP_FAILED) munmap(const_cast<void*>(buffer_), size_);

    if (fd_ != -1) close(fd_);
  }

  // Return the number of tensors in the BTF file.
  int NumTensors() const {
    // Bounds checked should have already been performed by Open().
    assert(size_ >= sizeof(int64_t) &&
           "Cannot read NumTensors (has Open been called?)");
    return *static_cast<const int64_t*>(buffer_);
  }

  // Read the TensorHeader for the tensor i, and store the payload offset into
  // `payload_pos`.  Returns nullptr if the header is malformed.
  const TensorHeader* ReadTensorHeader(int i, size_t* payload_pos) const {
    assert(i >= 0 && i < NumTensors() && "Tensor index out of bounds");

    // Add one to account for NumTensors
    size_t pos = sizeof(int64_t) * (i + 1);

    // Find tensor record offset
    const int64_t* record_offset = Read<int64_t>(&pos);
    if (!record_offset) return nullptr;

    // Check record alignment
    if (*record_offset & 7) return nullptr;

    *payload_pos = *record_offset;
    return Read<TensorHeader>(payload_pos);
  }

  // Read an array of objects of type T located at the byte offset given by
  // `pos`, checking to ensure the buffer is large enough to contain this array.
  // The offset `pos` is updated to point past the end of the array.
  template <typename T>
  const T* Read(size_t* pos, size_t count = 1) const {
    if (*pos + sizeof(T) * count > size_) return nullptr;
    const T* result =
        reinterpret_cast<const T*>(static_cast<const char*>(buffer_) + *pos);
    *pos += sizeof(T) * count;
    return result;
  }

  // Read a dense tensor from the BTF file.
  //
  // The returned tensor points directly to the mmaped BTF data,
  // and therefore:
  //
  //   - The returned tensor should not be modified.
  //   - The returned tensor should not outlive this BtfFile.
  //
  // Returns std::nullopt if the payload is malformed.
  std::optional<DenseHostTensor> ReadDenseHostTensorPayload(
      size_t* pos, DType type, uint64_t rank) const {
    // Technically, the BTF spec defines dims as unsigned, but the difference
    // should not matter because each tensor dimension should be significantly
    // smaller than std::numeric_limits<int64_t>::max().
    const int64_t* dims = Read<int64_t>(pos, rank);
    if (!dims) return std::nullopt;

    TensorMetadata metadata(type, llvm::ArrayRef<tfrt::Index>(dims, rank));

    size_t data_size = GetHostSize(type) * metadata.shape.GetNumElements();
    const void* data = Read<char>(pos, data_size);
    if (!data) return std::nullopt;

    RCReference<HostBuffer> buf = HostBuffer::CreateFromExternal(
        const_cast<void*>(data), data_size, [](void*, size_t) {});
    return DenseHostTensor(metadata, std::move(buf));
  }

  // Read a COO tensor from the BTF file.
  //
  // The caveats described in the ReadDenseHostTensorPayload documentation also
  // apply to the returned CooHostTensor object.
  //
  // Returns std::nullopt if the payload is malformed.
  std::optional<CooHostTensor> ReadCooHostTensorPayload(size_t* pos, DType type,
                                                        uint64_t rank) const {
    const int64_t* dims = Read<int64_t>(pos, rank);
    if (!dims) return std::nullopt;

    TensorShape shape(llvm::ArrayRef<tfrt::Index>(dims, rank));

    std::optional<DenseHostTensor> indices =
        ReadDenseHostTensorPayload(pos, DType(DType::I64), 2);
    if (!indices) return std::nullopt;

    std::optional<DenseHostTensor> values =
        ReadDenseHostTensorPayload(pos, type, 1);
    if (!values) return std::nullopt;

    // Check that dimensions are valid
    if (indices->shape().GetDimensionSize(0) !=
            values->shape().GetDimensionSize(0) ||
        indices->shape().GetDimensionSize(1) != rank)
      return std::nullopt;

    return CooHostTensor(shape, type, std::move(*indices), std::move(*values));
  }

 private:
  int fd_ = -1;
  const void* buffer_ = MAP_FAILED;
  size_t size_ = 0;
};

}  // namespace

int main(int argc, char* argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "BTF inspector\n");

  BtfFile file;
  llvm::Error err = file.Open(cl_input_filename.c_str());
  if (err) {
    llvm::errs() << err << "\n";
    return 1;
  }

  std::vector<tfrt::Index> dims_vec;
  for (int i = 0; i < file.NumTensors(); i++) {
    size_t payload_pos;
    const TensorHeader* header = file.ReadTensorHeader(i, &payload_pos);
    if (!header) {
      llvm::errs() << "Could not parse header for tensor " << i << "\n";
      continue;
    }

    DType type(ToDTypeKind(header->dtype));

    switch (header->layout) {
      case TensorLayout::kRMD: {
        std::optional<DenseHostTensor> tensor =
            file.ReadDenseHostTensorPayload(&payload_pos, type, header->rank);
        if (!tensor) {
          llvm::errs() << "Could not parse dense payload for tensor " << i
                       << "\n";
          continue;
        }

        llvm::outs() << "[" << i << "] ";
        tensor->Print(llvm::outs());
        llvm::outs() << "\n";
      } break;

      case TensorLayout::kCOO_EXPERIMENTAL: {
        std::optional<CooHostTensor> tensor =
            file.ReadCooHostTensorPayload(&payload_pos, type, header->rank);
        if (!tensor) {
          llvm::errs() << "Could not parse COO payload for tensor " << i
                       << "\n";
          continue;
        }

        llvm::outs() << "[" << i << "] ";
        tensor->Print(llvm::outs());
        // Newline already printed by CooHostTensor::Print
      } break;
    }
  }
}
