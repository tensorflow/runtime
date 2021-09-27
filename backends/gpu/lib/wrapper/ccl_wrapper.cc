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

// Thin wrapper around the NCCL API adding llvm::Error.
#include "tfrt/gpu/wrapper/ccl_wrapper.h"

#include <utility>

#include "tfrt/gpu/wrapper/nccl_wrapper.h"
#include "tfrt/gpu/wrapper/rccl_wrapper.h"
#include "wrapper_detail.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

llvm::Expected<int> GetCclDataTypeSizeBytes(ncclDataType_t data_type) {
  switch (data_type) {
    case ncclInt8:
    case ncclUint8:
      return 1;
    case ncclFloat16:
#if defined(__CUDA_BF16_TYPES_EXIST__)
    case ncclBfloat16:
#endif
      return 2;
    case ncclInt32:
    case ncclUint32:
    case ncclFloat32:
      return 4;
    case ncclInt64:
    case ncclUint64:
    case ncclFloat64:
      return 8;
    default:
      return MakeStringError("Unknown ncclDataType_t: ", data_type);
  }
}

void internal::CclCommDeleter::operator()(CclComm comm) const {
  LogIfError(CclCommDestroy(comm));
}

llvm::Expected<int> CclGetVersion(Platform platform) {
  switch (platform) {
    case Platform::CUDA:
      return NcclGetVersion();
    case Platform::ROCm:
      return RcclGetVersion();
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<ncclUniqueId> CclGetUniqueId(Platform platform) {
  switch (platform) {
    case Platform::CUDA:
      return NcclGetUniqueId();
    case Platform::ROCm:
      return RcclGetUniqueId();
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<OwningCclComm> CclCommInitRank(CurrentContext current,
                                              int nranks, ncclUniqueId commId,
                                              int rank) {
  Platform platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return NcclCommInitRank(current, nranks, commId, rank);
    case Platform::ROCm:
      return RcclCommInitRank(current, nranks, commId, rank);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error CclCommDestroy(CclComm comm) {
  Platform platform = comm.platform();
  switch (platform) {
    case Platform::CUDA:
      return NcclCommDestroy(comm);
    case Platform::ROCm:
      return RcclCommDestroy(comm);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error CclCommAbort(CclComm comm) {
  Platform platform = comm.platform();
  switch (platform) {
    case Platform::CUDA:
      return NcclCommAbort(comm);
    case Platform::ROCm:
      return RcclCommAbort(comm);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error CclCommGetAsyncError(CclComm comm) {
  Platform platform = comm.platform();
  switch (platform) {
    case Platform::CUDA:
      return NcclCommGetAsyncError(comm);
    case Platform::ROCm:
      return RcclCommGetAsyncError(comm);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<int> CclCommCount(CclComm comm) {
  Platform platform = comm.platform();
  switch (platform) {
    case Platform::CUDA:
      return NcclCommCount(comm);
    case Platform::ROCm:
      return RcclCommCount(comm);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Expected<int> CclCommUserRank(CclComm comm) {
  Platform platform = comm.platform();
  switch (platform) {
    case Platform::CUDA:
      return NcclCommUserRank(comm);
    case Platform::ROCm:
      return RcclCommUserRank(comm);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error CclReduce(CurrentContext current, Pointer<const void> sendbuff,
                      Pointer<void> recvbuff, size_t count,
                      ncclDataType_t datatype, ncclRedOp_t op, int root,
                      CclComm comm, Stream stream) {
  Platform platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return NcclReduce(current, sendbuff, recvbuff, count, datatype, op, root,
                        comm, stream);
    case Platform::ROCm:
      return RcclReduce(current, sendbuff, recvbuff, count, datatype, op, root,
                        comm, stream);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error CclBcast(CurrentContext current, Pointer<void> buffer, size_t count,
                     ncclDataType_t datatype, int root, CclComm comm,
                     Stream stream) {
  Platform platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return NcclBcast(current, buffer, count, datatype, root, comm, stream);
    case Platform::ROCm:
      return RcclBcast(current, buffer, count, datatype, root, comm, stream);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error CclBroadcast(CurrentContext current, Pointer<const void> sendbuff,
                         Pointer<void> recvbuff, size_t count,
                         ncclDataType_t datatype, int root, CclComm comm,
                         Stream stream) {
  Platform platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return NcclBroadcast(current, sendbuff, recvbuff, count, datatype, root,
                           comm, stream);
    case Platform::ROCm:
      return RcclBroadcast(current, sendbuff, recvbuff, count, datatype, root,
                           comm, stream);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error CclAllReduce(CurrentContext current, Pointer<const void> sendbuff,
                         Pointer<void> recvbuff, size_t count,
                         ncclDataType_t datatype, ncclRedOp_t op, CclComm comm,
                         Stream stream) {
  Platform platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return NcclAllReduce(current, sendbuff, recvbuff, count, datatype, op,
                           comm, stream);
    case Platform::ROCm:
      return RcclAllReduce(current, sendbuff, recvbuff, count, datatype, op,
                           comm, stream);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error CclReduceScatter(CurrentContext current,
                             Pointer<const void> sendbuff,
                             Pointer<void> recvbuff, size_t recvcount,
                             ncclDataType_t datatype, ncclRedOp_t op,
                             CclComm comm, Stream stream) {
  Platform platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return NcclReduceScatter(current, sendbuff, recvbuff, recvcount, datatype,
                               op, comm, stream);
    case Platform::ROCm:
      return RcclReduceScatter(current, sendbuff, recvbuff, recvcount, datatype,
                               op, comm, stream);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error CclAllGather(CurrentContext current, Pointer<const void> sendbuff,
                         Pointer<void> recvbuff, size_t sendcount,
                         ncclDataType_t datatype, CclComm comm, Stream stream) {
  Platform platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return NcclAllGather(current, sendbuff, recvbuff, sendcount, datatype,
                           comm, stream);
    case Platform::ROCm:
      return RcclAllGather(current, sendbuff, recvbuff, sendcount, datatype,
                           comm, stream);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error CclSend(CurrentContext current, Pointer<const void> sendbuff,
                    size_t count, ncclDataType_t datatype, int peer,
                    CclComm comm, Stream stream) {
  Platform platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return NcclSend(current, sendbuff, count, datatype, peer, comm, stream);
    case Platform::ROCm:
      return RcclSend(current, sendbuff, count, datatype, peer, comm, stream);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error CclRecv(CurrentContext current, Pointer<void> recvbuff,
                    size_t count, ncclDataType_t datatype, int peer,
                    CclComm comm, Stream stream) {
  Platform platform = current.platform();
  switch (platform) {
    case Platform::CUDA:
      return NcclRecv(current, recvbuff, count, datatype, peer, comm, stream);
    case Platform::ROCm:
      return RcclRecv(current, recvbuff, count, datatype, peer, comm, stream);
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error CclGroupStart(Platform platform) {
  switch (platform) {
    case Platform::CUDA:
      return NcclGroupStart();
    case Platform::ROCm:
      return RcclGroupStart();
    default:
      return InvalidPlatform(platform);
  }
}

llvm::Error CclGroupEnd(Platform platform) {
  switch (platform) {
    case Platform::CUDA:
      return NcclGroupEnd();
    case Platform::ROCm:
      return RcclGroupEnd();
    default:
      return InvalidPlatform(platform);
  }
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
