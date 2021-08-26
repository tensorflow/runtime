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

// Thin wrapper around the RCCL API adding llvm::Error.
#include "tfrt/gpu/wrapper/rccl_wrapper.h"

#include <utility>

#include "tfrt/gpu/wrapper/rccl_stub.h"
#include "wrapper_detail.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

llvm::Expected<int> RcclGetVersion() {
  int version = 0;
  RETURN_IF_ERROR(rcclGetVersion(&version));
  return version;
}

llvm::Expected<ncclUniqueId> RcclGetUniqueId() {
  ncclUniqueId id;
  RETURN_IF_ERROR(rcclGetUniqueId(&id));
  return id;
}

llvm::Expected<OwningCclComm> RcclCommInitRank(CurrentContext current,
                                               int nranks, ncclUniqueId commId,
                                               int rank) {
  CheckHipContext(current);
  ncclComm_t comm;
  RETURN_IF_ERROR(rcclCommInitRank(&comm, nranks, commId, rank));
  CheckHipContext(current);  // See NcclCommInitRank() comment.
  return OwningCclComm({comm, Platform::ROCm});
}

llvm::Error RcclCommDestroy(ncclComm_t comm) {
  return TO_ERROR(rcclCommDestroy(comm));
}

llvm::Error RcclCommAbort(ncclComm_t comm) {
  return TO_ERROR(rcclCommAbort(comm));
}

llvm::Error RcclCommGetAsyncError(ncclComm_t comm) {
  ncclResult_t result;
  RETURN_IF_ERROR(rcclCommGetAsyncError(comm, &result));
  return TO_ERROR(result);
}

llvm::Expected<int> RcclCommCount(ncclComm_t comm) {
  int count = 0;
  RETURN_IF_ERROR(rcclCommCount(comm, &count));
  return count;
}

llvm::Expected<int> RcclCommUserRank(ncclComm_t comm) {
  int rank = 0;
  RETURN_IF_ERROR(rcclCommUserRank(comm, &rank));
  return rank;
}

llvm::Error RcclReduce(CurrentContext current, Pointer<const void> sendbuff,
                       Pointer<void> recvbuff, size_t count,
                       ncclDataType_t datatype, ncclRedOp_t op, int root,
                       ncclComm_t comm, hipStream_t stream) {
  CheckHipContext(current);
  RETURN_IF_ERROR(rcclReduce(sendbuff.raw(Platform::ROCm),
                             recvbuff.raw(Platform::ROCm), count, datatype, op,
                             root, comm, stream));
  CheckHipContext(current);
  return llvm::Error::success();
}

llvm::Error RcclBcast(CurrentContext current, Pointer<void> buffer,
                      size_t count, ncclDataType_t datatype, int root,
                      ncclComm_t comm, hipStream_t stream) {
  CheckHipContext(current);
  RETURN_IF_ERROR(rcclBcast(buffer.raw(Platform::ROCm), count, datatype, root,
                            comm, stream));
  CheckHipContext(current);
  return llvm::Error::success();
}

llvm::Error RcclBroadcast(CurrentContext current, Pointer<const void> sendbuff,
                          Pointer<void> recvbuff, size_t count,
                          ncclDataType_t datatype, int root, ncclComm_t comm,
                          hipStream_t stream) {
  CheckHipContext(current);
  RETURN_IF_ERROR(rcclBroadcast(sendbuff.raw(Platform::ROCm),
                                recvbuff.raw(Platform::ROCm), count, datatype,
                                root, comm, stream));
  CheckHipContext(current);
  return llvm::Error::success();
}

llvm::Error RcclAllReduce(CurrentContext current, Pointer<const void> sendbuff,
                          Pointer<void> recvbuff, size_t count,
                          ncclDataType_t datatype, ncclRedOp_t op,
                          ncclComm_t comm, hipStream_t stream) {
  CheckHipContext(current);
  RETURN_IF_ERROR(rcclAllReduce(sendbuff.raw(Platform::ROCm),
                                recvbuff.raw(Platform::ROCm), count, datatype,
                                op, comm, stream));
  CheckHipContext(current);
  return llvm::Error::success();
}

llvm::Error RcclReduceScatter(CurrentContext current,
                              Pointer<const void> sendbuff,
                              Pointer<void> recvbuff, size_t recvcount,
                              ncclDataType_t datatype, ncclRedOp_t op,
                              ncclComm_t comm, hipStream_t stream) {
  CheckHipContext(current);
  RETURN_IF_ERROR(rcclReduceScatter(sendbuff.raw(Platform::ROCm),
                                    recvbuff.raw(Platform::ROCm), recvcount,
                                    datatype, op, comm, stream));
  CheckHipContext(current);
  return llvm::Error::success();
}

llvm::Error RcclAllGather(CurrentContext current, Pointer<const void> sendbuff,
                          Pointer<void> recvbuff, size_t sendcount,
                          ncclDataType_t datatype, ncclComm_t comm,
                          hipStream_t stream) {
  CheckHipContext(current);
  RETURN_IF_ERROR(rcclAllGather(sendbuff.raw(Platform::ROCm),
                                recvbuff.raw(Platform::ROCm), sendcount,
                                datatype, comm, stream));
  CheckHipContext(current);
  return llvm::Error::success();
}

llvm::Error RcclSend(CurrentContext current, Pointer<const void> sendbuff,
                     size_t count, ncclDataType_t datatype, int peer,
                     ncclComm_t comm, hipStream_t stream) {
  CheckHipContext(current);
  RETURN_IF_ERROR(rcclSend(sendbuff.raw(Platform::ROCm), count, datatype, peer,
                           comm, stream));
  CheckHipContext(current);
  return llvm::Error::success();
}

llvm::Error RcclRecv(CurrentContext current, Pointer<void> recvbuff,
                     size_t count, ncclDataType_t datatype, int peer,
                     ncclComm_t comm, hipStream_t stream) {
  CheckHipContext(current);
  RETURN_IF_ERROR(rcclRecv(recvbuff.raw(Platform::ROCm), count, datatype, peer,
                           comm, stream));
  CheckHipContext(current);
  return llvm::Error::success();
}

llvm::Error RcclGroupStart() { return TO_ERROR(rcclGroupStart()); }

llvm::Error RcclGroupEnd() { return TO_ERROR(rcclGroupEnd()); }

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
