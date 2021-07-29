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

#ifndef TFRT_GPU_WRAPPER_NCCL_WRAPPER_H_
#define TFRT_GPU_WRAPPER_NCCL_WRAPPER_H_

#include <cstddef>
#include <memory>

#include "tfrt/gpu/wrapper/ccl_wrapper.h"

namespace tfrt {
namespace gpu {
namespace wrapper {

llvm::Expected<int> NcclGetVersion();
llvm::Expected<ncclUniqueId> NcclGetUniqueId();
llvm::Expected<OwningCclComm> NcclCommInitRank(CurrentContext current,
                                               int nranks, ncclUniqueId commId,
                                               int rank);
llvm::Error NcclCommDestroy(ncclComm_t comm);
llvm::Error NcclCommAbort(ncclComm_t comm);

llvm::Error NcclCommGetAsyncError(ncclComm_t comm);
llvm::Expected<int> NcclCommCount(ncclComm_t comm);
llvm::Expected<int> NcclCommUserRank(ncclComm_t comm);

llvm::Error NcclReduce(CurrentContext current, Pointer<const void> sendbuff,
                       Pointer<void> recvbuff, size_t count,
                       ncclDataType_t datatype, ncclRedOp_t op, int root,
                       ncclComm_t comm, cudaStream_t stream);
llvm::Error NcclBcast(CurrentContext current, Pointer<void> buffer,
                      size_t count, ncclDataType_t datatype, int root,
                      ncclComm_t comm, cudaStream_t stream);
llvm::Error NcclBroadcast(CurrentContext current, Pointer<const void> sendbuff,
                          Pointer<void> recvbuff, size_t count,
                          ncclDataType_t datatype, int root, ncclComm_t comm,
                          cudaStream_t stream);
llvm::Error NcclAllReduce(CurrentContext current, Pointer<const void> sendbuff,
                          Pointer<void> recvbuff, size_t count,
                          ncclDataType_t datatype, ncclRedOp_t op,
                          ncclComm_t comm, cudaStream_t stream);
llvm::Error NcclReduceScatter(CurrentContext current,
                              Pointer<const void> sendbuff,
                              Pointer<void> recvbuff, size_t recvcount,
                              ncclDataType_t datatype, ncclRedOp_t op,
                              ncclComm_t comm, cudaStream_t stream);
llvm::Error NcclAllGather(CurrentContext current, Pointer<const void> sendbuff,
                          Pointer<void> recvbuff, size_t sendcount,
                          ncclDataType_t datatype, ncclComm_t comm,
                          cudaStream_t stream);
llvm::Error NcclSend(CurrentContext current, Pointer<const void> sendbuff,
                     size_t count, ncclDataType_t datatype, int peer,
                     ncclComm_t comm, cudaStream_t stream);
llvm::Error NcclRecv(CurrentContext current, Pointer<void> recvbuff,
                     size_t count, ncclDataType_t datatype, int peer,
                     ncclComm_t comm, cudaStream_t stream);

llvm::Error NcclGroupStart();
llvm::Error NcclGroupEnd();

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_WRAPPER_NCCL_WRAPPER_H_
