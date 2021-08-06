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

#ifndef TFRT_GPU_WRAPPER_CCL_WRAPPER_H_
#define TFRT_GPU_WRAPPER_CCL_WRAPPER_H_

#include <cstddef>
#include <memory>

#include "src/nccl.h"  // from @nccl_headers
#include "tfrt/gpu/wrapper/wrapper.h"

namespace llvm {
template <>
struct PointerLikeTypeTraits<ncclComm_t> {
  static void* getAsVoidPointer(ncclComm_t comm) { return comm; }
  static ncclComm_t getFromVoidPointer(void* ptr) {
    return static_cast<ncclComm_t>(ptr);
  }
  // NOLINTNEXTLINE(readability-identifier-naming)
  static constexpr int NumLowBitsAvailable = 2;
};
}  // namespace llvm

namespace tfrt {
namespace gpu {
namespace wrapper {

using CclDataType = ncclDataType_t;
using CclReductionOp = ncclRedOp_t;

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, ncclResult_t result);

template <>
Expected<ncclDataType_t> Parse<ncclDataType_t>(llvm::StringRef name);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, ncclDataType_t value);

template <>
Expected<ncclRedOp_t> Parse<ncclRedOp_t>(llvm::StringRef name);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, ncclRedOp_t value);

// Non-owning NCCL communicator for a specific platform.
class CclComm {
 public:
  CclComm() = default;
  explicit CclComm(std::nullptr_t) : pair_(nullptr, Platform::NONE) {}
  CclComm(ncclComm_t comm, Platform platform) : pair_(comm, platform) {}
  // Required for std::unique_ptr<Resource>.
  CclComm& operator=(std::nullptr_t) {
    pair_.setPointer(nullptr);
    return *this;
  }
  // Required for std::unique_ptr<Resource>.
  operator bool() const {  // NOLINT(google-explicit-constructor)
    return *this != nullptr;
  }
  operator ncclComm_t() const {  // NOLINT(google-explicit-constructor)
    return static_cast<ncclComm_t>(pair_.getPointer());
  }
  Platform platform() const { return pair_.getInt(); }
  bool operator==(std::nullptr_t) const {
    return pair_.getPointer() == nullptr;
  }
  bool operator!=(std::nullptr_t) const {
    return pair_.getPointer() != nullptr;
  }
  bool operator==(CclComm other) const { return pair_ == other.pair_; }
  bool operator!=(CclComm other) const { return pair_ != other.pair_; }

  // For member access from std::unique_ptr.
  const CclComm* operator->() const { return this; }

 private:
  llvm::PointerIntPair<ncclComm_t, 2, Platform> pair_;

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                       const CclComm& comm) {
    return os << comm.pair_.getPointer() << " (" << comm.platform() << ")";
  }
};

namespace internal {
// Helper to wrap resources and memory into RAII types.
struct CclCommDeleter {
  using pointer = CclComm;
  void operator()(CclComm comm) const;
};
}  // namespace internal

// RAII wrappers for resources. Instances own the underlying resource.
//
// They are implemented as std::unique_ptrs with custom deleters.
//
// Use get() and release() to access the non-owning handle, please use with
// appropriate care.
using OwningCclComm = internal::OwningResource<internal::CclCommDeleter>;

llvm::Expected<int> CclGetVersion(Platform platform);
llvm::Expected<ncclUniqueId> CclGetUniqueId(Platform platform);
llvm::Expected<OwningCclComm> CclCommInitRank(CurrentContext current,
                                              int nranks, ncclUniqueId commId,
                                              int rank);
llvm::Error CclCommDestroy(CclComm comm);
llvm::Error CclCommAbort(CclComm comm);

llvm::Error CclCommGetAsyncError(CclComm comm);
llvm::Expected<int> CclCommCount(CclComm comm);
llvm::Expected<int> CclCommUserRank(CclComm comm);

llvm::Error CclReduce(CurrentContext current, Pointer<const void> sendbuff,
                      Pointer<void> recvbuff, size_t count,
                      ncclDataType_t datatype, ncclRedOp_t op, int root,
                      CclComm comm, Stream stream);
llvm::Error CclBcast(CurrentContext current, Pointer<void> buffer, size_t count,
                     ncclDataType_t datatype, int root, CclComm comm,
                     Stream stream);
llvm::Error CclBroadcast(CurrentContext current, Pointer<const void> sendbuff,
                         Pointer<void> recvbuff, size_t count,
                         ncclDataType_t datatype, int root, CclComm comm,
                         Stream stream);
llvm::Error CclAllReduce(CurrentContext current, Pointer<const void> sendbuff,
                         Pointer<void> recvbuff, size_t count,
                         ncclDataType_t datatype, ncclRedOp_t op, CclComm comm,
                         Stream stream);
llvm::Error CclReduceScatter(CurrentContext current,
                             Pointer<const void> sendbuff,
                             Pointer<void> recvbuff, size_t recvcount,
                             ncclDataType_t datatype, ncclRedOp_t op,
                             CclComm comm, Stream stream);
llvm::Error CclAllGather(CurrentContext current, Pointer<const void> sendbuff,
                         Pointer<void> recvbuff, size_t sendcount,
                         ncclDataType_t datatype, CclComm comm, Stream stream);
llvm::Error CclSend(CurrentContext current, Pointer<const void> sendbuff,
                    size_t count, ncclDataType_t datatype, int peer,
                    CclComm comm, Stream stream);
llvm::Error CclRecv(CurrentContext current, Pointer<void> recvbuff,
                    size_t count, ncclDataType_t datatype, int peer,
                    CclComm comm, Stream stream);

llvm::Error CclGroupStart(Platform platform);
llvm::Error CclGroupEnd(Platform platform);

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_WRAPPER_CCL_WRAPPER_H_
