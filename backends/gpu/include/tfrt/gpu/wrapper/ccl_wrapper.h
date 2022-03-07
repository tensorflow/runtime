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

// Thin abstraction layer for NCCL and RCCL.
#ifndef TFRT_GPU_WRAPPER_CCL_WRAPPER_H_
#define TFRT_GPU_WRAPPER_CCL_WRAPPER_H_

#include <cstddef>
#include <memory>

#include "tfrt/gpu/wrapper/ccl_types.h"
#include "tfrt/gpu/wrapper/wrapper.h"
// Note: this file must not be included before nccl.h from RCCL.
#include "src/nccl.h"  // from @nccl_headers

namespace tfrt {
namespace gpu {
namespace wrapper {

raw_ostream& Print(raw_ostream& os, ncclResult_t result);
raw_ostream& Print(raw_ostream& os, ncclDataType_t value);
raw_ostream& Print(raw_ostream& os, ncclRedOp_t value);

Expected<ncclDataType_t> Parse(llvm::StringRef name, ncclDataType_t);
Expected<ncclRedOp_t> Parse(llvm::StringRef name, ncclRedOp_t);

static_assert(sizeof(CclUniqueId) == sizeof(ncclUniqueId), "size mismatch");
static_assert(sizeof(CclDataType) == sizeof(ncclDataType_t), "size mismatch");
static_assert(sizeof(CclReductionOp) == sizeof(ncclRedOp_t), "size mismatch");

namespace internal {
template <>
class IsCclType<CclUniqueId, ncclUniqueId> : public std::true_type {};
template <>
class IsCclType<CclDataType, ncclDataType_t> : public std::true_type {};
template <>
class IsCclType<CclReductionOp, ncclRedOp_t> : public std::true_type {};

template <>
struct EnumStream<CclDataType, Platform::NONE>
    : EnumStreamPtrs<ncclDataType_t, Parse, Print> {};
template <>
struct EnumStream<CclReductionOp, Platform::NONE>
    : EnumStreamPtrs<ncclRedOp_t, Parse, Print> {};
}  // namespace internal

llvm::Expected<int> GetCclDataTypeSizeBytes(ncclDataType_t data_type);

llvm::Expected<int> CclGetVersion(Platform platform);
llvm::Expected<ncclUniqueId> CclGetUniqueId(Platform platform);
// Note: 'current' needs to be a primary context, or the call needs to be
// surrounded by CclGroupStart/End(). Same for the functions below.
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
