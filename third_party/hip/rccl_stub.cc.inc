ncclResult_t rcclGetVersion(int* version) {
  return DynamicCall<decltype(rcclGetVersion), &rcclGetVersion>(
      "ncclGetVersion", version);
}

ncclResult_t rcclGetUniqueId(ncclUniqueId* uniqueId) {
  return DynamicCall<decltype(rcclGetUniqueId), &rcclGetUniqueId>(
      "ncclGetUniqueId", uniqueId);
}

ncclResult_t rcclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId,
                              int rank) {
  return DynamicCall<decltype(rcclCommInitRank), &rcclCommInitRank>(
      "ncclCommInitRank", comm, nranks, commId, rank);
}

ncclResult_t rcclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist) {
  return DynamicCall<decltype(rcclCommInitAll), &rcclCommInitAll>(
      "ncclCommInitAll", comm, ndev, devlist);
}

ncclResult_t rcclCommDestroy(ncclComm_t comm) {
  return DynamicCall<decltype(rcclCommDestroy), &rcclCommDestroy>(
      "ncclCommDestroy", comm);
}

ncclResult_t rcclCommAbort(ncclComm_t comm) {
  return DynamicCall<decltype(rcclCommAbort), &rcclCommAbort>("ncclCommAbort",
                                                              comm);
}

ncclResult_t rcclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError) {
  return DynamicCall<decltype(rcclCommGetAsyncError), &rcclCommGetAsyncError>(
      "ncclCommGetAsyncError", comm, asyncError);
}

ncclResult_t rcclCommCount(const ncclComm_t comm, int* count) {
  return DynamicCall<decltype(rcclCommCount), &rcclCommCount>("ncclCommCount",
                                                              comm, count);
}

ncclResult_t rcclCommCuDevice(const ncclComm_t comm, int* device) {
  return DynamicCall<decltype(rcclCommCuDevice), &rcclCommCuDevice>(
      "ncclCommCuDevice", comm, device);
}

ncclResult_t rcclCommUserRank(const ncclComm_t comm, int* rank) {
  return DynamicCall<decltype(rcclCommUserRank), &rcclCommUserRank>(
      "ncclCommUserRank", comm, rank);
}

ncclResult_t rcclReduce(const void* sendbuff, void* recvbuff, size_t count,
                        ncclDataType_t datatype, ncclRedOp_t op, int root,
                        ncclComm_t comm, hipStream_t stream) {
  return DynamicCall<decltype(rcclReduce), &rcclReduce>(
      "ncclReduce", sendbuff, recvbuff, count, datatype, op, root, comm,
      stream);
}

ncclResult_t rcclBcast(void* buff, size_t count, ncclDataType_t datatype,
                       int root, ncclComm_t comm, hipStream_t stream) {
  return DynamicCall<decltype(rcclBcast), &rcclBcast>(
      "ncclBcast", buff, count, datatype, root, comm, stream);
}

ncclResult_t rcclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, int root, ncclComm_t comm,
                           hipStream_t stream) {
  return DynamicCall<decltype(rcclBroadcast), &rcclBroadcast>(
      "ncclBroadcast", sendbuff, recvbuff, count, datatype, root, comm, stream);
}

ncclResult_t rcclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, hipStream_t stream) {
  return DynamicCall<decltype(rcclAllReduce), &rcclAllReduce>(
      "ncclAllReduce", sendbuff, recvbuff, count, datatype, op, comm, stream);
}

ncclResult_t rcclReduceScatter(const void* sendbuff, void* recvbuff,
                               size_t recvcount, ncclDataType_t datatype,
                               ncclRedOp_t op, ncclComm_t comm,
                               hipStream_t stream) {
  return DynamicCall<decltype(rcclReduceScatter), &rcclReduceScatter>(
      "ncclReduceScatter", sendbuff, recvbuff, recvcount, datatype, op, comm,
      stream);
}

ncclResult_t rcclAllGather(const void* sendbuff, void* recvbuff,
                           size_t sendcount, ncclDataType_t datatype,
                           ncclComm_t comm, hipStream_t stream) {
  return DynamicCall<decltype(rcclAllGather), &rcclAllGather>(
      "ncclAllGather", sendbuff, recvbuff, sendcount, datatype, comm, stream);
}

ncclResult_t rcclSend(const void* sendbuff, size_t count,
                      ncclDataType_t datatype, int peer, ncclComm_t comm,
                      hipStream_t stream) {
  return DynamicCall<decltype(rcclSend), &rcclSend>(
      "ncclSend", sendbuff, count, datatype, peer, comm, stream);
}

ncclResult_t rcclRecv(void* recvbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, hipStream_t stream) {
  return DynamicCall<decltype(rcclRecv), &rcclRecv>(
      "ncclRecv", recvbuff, count, datatype, peer, comm, stream);
}

ncclResult_t rcclGroupStart() {
  return DynamicCall<decltype(rcclGroupStart), &rcclGroupStart>(
      "ncclGroupStart");
}

ncclResult_t rcclGroupEnd() {
  return DynamicCall<decltype(rcclGroupEnd), &rcclGroupEnd>("ncclGroupEnd");
}
