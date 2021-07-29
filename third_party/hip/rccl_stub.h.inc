ncclResult_t rcclGetVersion(int* version);

ncclResult_t rcclGetUniqueId(ncclUniqueId* uniqueId);

ncclResult_t rcclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId,
                              int rank);

ncclResult_t rcclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist);

ncclResult_t rcclCommDestroy(ncclComm_t comm);

ncclResult_t rcclCommAbort(ncclComm_t comm);

ncclResult_t rcclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError);

ncclResult_t rcclCommCount(const ncclComm_t comm, int* count);

ncclResult_t rcclCommCuDevice(const ncclComm_t comm, int* device);

ncclResult_t rcclCommUserRank(const ncclComm_t comm, int* rank);

ncclResult_t rcclReduce(const void* sendbuff, void* recvbuff, size_t count,
                        ncclDataType_t datatype, ncclRedOp_t op, int root,
                        ncclComm_t comm, hipStream_t stream);

ncclResult_t rcclBcast(void* buff, size_t count, ncclDataType_t datatype,
                       int root, ncclComm_t comm, hipStream_t stream);

ncclResult_t rcclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, int root, ncclComm_t comm,
                           hipStream_t stream);

ncclResult_t rcclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, hipStream_t stream);
ncclResult_t rcclReduceScatter(const void* sendbuff, void* recvbuff,
                               size_t recvcount, ncclDataType_t datatype,
                               ncclRedOp_t op, ncclComm_t comm,
                               hipStream_t stream);

ncclResult_t rcclAllGather(const void* sendbuff, void* recvbuff,
                           size_t sendcount, ncclDataType_t datatype,
                           ncclComm_t comm, hipStream_t stream);

ncclResult_t rcclSend(const void* sendbuff, size_t count,
                      ncclDataType_t datatype, int peer, ncclComm_t comm,
                      hipStream_t stream);

ncclResult_t rcclRecv(void* recvbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, hipStream_t stream);

ncclResult_t rcclGroupStart();

ncclResult_t rcclGroupEnd();
