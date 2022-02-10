hipError_t hipInit(unsigned int flags) {
  return DynamicCall<decltype(hipInit), &hipInit>("hipInit", flags);
}

hipError_t hipDriverGetVersion(int* driverVersion) {
  return DynamicCall<decltype(hipDriverGetVersion), &hipDriverGetVersion>(
      "hipDriverGetVersion", driverVersion);
}

hipError_t hipRuntimeGetVersion(int* runtimeVersion) {
  return DynamicCall<decltype(hipRuntimeGetVersion), &hipRuntimeGetVersion>(
      "hipRuntimeGetVersion", runtimeVersion);
}

hipError_t hipGetLastError(void) {
  return DynamicCall<decltype(hipGetLastError), &hipGetLastError>(
      "hipGetLastError");
}

hipError_t hipPeekAtLastError(void) {
  return DynamicCall<decltype(hipPeekAtLastError), &hipPeekAtLastError>(
      "hipPeekAtLastError");
}

hipError_t hipDeviceGet(hipDevice_t* device, int ordinal) {
  return DynamicCall<decltype(hipDeviceGet), &hipDeviceGet>("hipDeviceGet",
                                                            device, ordinal);
}

hipError_t hipDeviceGetName(char* name, int len, hipDevice_t device) {
  return DynamicCall<decltype(hipDeviceGetName), &hipDeviceGetName>(
      "hipDeviceGetName", name, len, device);
}

hipError_t hipDeviceGetPCIBusId(char* pciBusId, int len, int device) {
  return DynamicCall<decltype(hipDeviceGetPCIBusId), &hipDeviceGetPCIBusId>(
      "hipDeviceGetPCIBusId", pciBusId, len, device);
}

hipError_t hipDeviceTotalMem(size_t* bytes, hipDevice_t device) {
  return DynamicCall<decltype(hipDeviceTotalMem), &hipDeviceTotalMem>(
      "hipDeviceTotalMem", bytes, device);
}

hipError_t hipDeviceSynchronize(void) {
  return DynamicCall<decltype(hipDeviceSynchronize), &hipDeviceSynchronize>(
      "hipDeviceSynchronize");
}

hipError_t hipGetDevice(int* deviceId) {
  return DynamicCall<decltype(hipGetDevice), &hipGetDevice>("hipGetDevice",
                                                            deviceId);
}

hipError_t hipGetDeviceCount(int* count) {
  return DynamicCall<decltype(hipGetDeviceCount), &hipGetDeviceCount>(
      "hipGetDeviceCount", count);
}

hipError_t hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr,
                                 int deviceId) {
  return DynamicCall<decltype(hipDeviceGetAttribute), &hipDeviceGetAttribute>(
      "hipDeviceGetAttribute", pi, attr, deviceId);
}

hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int deviceId) {
  return DynamicCall<decltype(hipGetDeviceProperties), &hipGetDeviceProperties>(
      "hipGetDeviceProperties", prop, deviceId);
}

hipError_t hipDeviceGetLimit(size_t* pValue, enum hipLimit_t limit) {
  return DynamicCall<decltype(hipDeviceGetLimit), &hipDeviceGetLimit>(
      "hipDeviceGetLimit", pValue, limit);
}

hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags) {
  return DynamicCall<decltype(hipStreamCreateWithFlags),
                     &hipStreamCreateWithFlags>("hipStreamCreateWithFlags",
                                                stream, flags);
}

hipError_t hipStreamCreateWithPriority(hipStream_t* stream, unsigned int flags,
                                       int priority) {
  return DynamicCall<decltype(hipStreamCreateWithPriority),
                     &hipStreamCreateWithPriority>(
      "hipStreamCreateWithPriority", stream, flags, priority);
}

hipError_t hipDeviceGetStreamPriorityRange(int* leastPriority,
                                           int* greatestPriority) {
  return DynamicCall<decltype(hipDeviceGetStreamPriorityRange),
                     &hipDeviceGetStreamPriorityRange>(
      "hipDeviceGetStreamPriorityRange", leastPriority, greatestPriority);
}

hipError_t hipStreamDestroy(hipStream_t stream) {
  return DynamicCall<decltype(hipStreamDestroy), &hipStreamDestroy>(
      "hipStreamDestroy", stream);
}

hipError_t hipStreamQuery(hipStream_t stream) {
  return DynamicCall<decltype(hipStreamQuery), &hipStreamQuery>(
      "hipStreamQuery", stream);
}

hipError_t hipStreamSynchronize(hipStream_t stream) {
  return DynamicCall<decltype(hipStreamSynchronize), &hipStreamSynchronize>(
      "hipStreamSynchronize", stream);
}

hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event,
                              unsigned int flags) {
  return DynamicCall<decltype(hipStreamWaitEvent), &hipStreamWaitEvent>(
      "hipStreamWaitEvent", stream, event, flags);
}

hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int* flags) {
  return DynamicCall<decltype(hipStreamGetFlags), &hipStreamGetFlags>(
      "hipStreamGetFlags", stream, flags);
}

hipError_t hipStreamGetPriority(hipStream_t stream, int* priority) {
  return DynamicCall<decltype(hipStreamGetPriority), &hipStreamGetPriority>(
      "hipStreamGetPriority", stream, priority);
}

hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned flags) {
  return DynamicCall<decltype(hipEventCreateWithFlags),
                     &hipEventCreateWithFlags>("hipEventCreateWithFlags", event,
                                               flags);
}

hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
  return DynamicCall<decltype(hipEventRecord), &hipEventRecord>(
      "hipEventRecord", event, stream);
}

hipError_t hipEventDestroy(hipEvent_t event) {
  return DynamicCall<decltype(hipEventDestroy), &hipEventDestroy>(
      "hipEventDestroy", event);
}

hipError_t hipEventSynchronize(hipEvent_t event) {
  return DynamicCall<decltype(hipEventSynchronize), &hipEventSynchronize>(
      "hipEventSynchronize", event);
}

hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop) {
  return DynamicCall<decltype(hipEventElapsedTime), &hipEventElapsedTime>(
      "hipEventElapsedTime", ms, start, stop);
}

hipError_t hipEventQuery(hipEvent_t event) {
  return DynamicCall<decltype(hipEventQuery), &hipEventQuery>("hipEventQuery",
                                                              event);
}

hipError_t hipPointerGetAttributes(hipPointerAttribute_t* attributes,
                                   const void* ptr) {
  return DynamicCall<decltype(hipPointerGetAttributes),
                     &hipPointerGetAttributes>("hipPointerGetAttributes",
                                               attributes, ptr);
}

hipError_t hipMalloc(void** ptr, size_t size) {
  return DynamicCall<decltype(hipMalloc), &hipMalloc>("hipMalloc", ptr, size);
}

hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags) {
  return DynamicCall<decltype(hipHostMalloc), &hipHostMalloc>("hipHostMalloc",
                                                              ptr, size, flags);
}

hipError_t hipMallocManaged(void** dev_ptr, size_t size,
                            unsigned int flags __dparm(hipMemAttachGlobal)) {
  return DynamicCall<decltype(hipMallocManaged), &hipMallocManaged>(
      "hipMallocManaged", dev_ptr, size, flags);
}

hipError_t hipHostGetDevicePointer(void** devPtr, void* hstPtr,
                                   unsigned int flags) {
  return DynamicCall<decltype(hipHostGetDevicePointer),
                     &hipHostGetDevicePointer>("hipHostGetDevicePointer",
                                               devPtr, hstPtr, flags);
}

hipError_t hipHostRegister(void* hostPtr, size_t sizeBytes,
                           unsigned int flags) {
  return DynamicCall<decltype(hipHostRegister), &hipHostRegister>(
      "hipHostRegister", hostPtr, sizeBytes, flags);
}

hipError_t hipHostUnregister(void* hostPtr) {
  return DynamicCall<decltype(hipHostUnregister), &hipHostUnregister>(
      "hipHostUnregister", hostPtr);
}

hipError_t hipFree(void* ptr) {
  return DynamicCall<decltype(hipFree), &hipFree>("hipFree", ptr);
}

hipError_t hipHostFree(void* ptr) {
  return DynamicCall<decltype(hipHostFree), &hipHostFree>("hipHostFree", ptr);
}

hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes,
                     hipMemcpyKind kind) {
  return DynamicCall<decltype(hipMemcpy), &hipMemcpy>("hipMemcpy", dst, src,
                                                      sizeBytes, kind);
}

hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes,
                          hipMemcpyKind kind, hipStream_t stream __dparm(0)) {
  return DynamicCall<decltype(hipMemcpyAsync), &hipMemcpyAsync>(
      "hipMemcpyAsync", dst, src, sizeBytes, kind, stream);
}

hipError_t hipMemset(void* dst, int value, size_t sizeBytes) {
  return DynamicCall<decltype(hipMemset), &hipMemset>("hipMemset", dst, value,
                                                      sizeBytes);
}

hipError_t hipMemsetD32(hipDeviceptr_t dest, int value, size_t count) {
  return DynamicCall<decltype(hipMemsetD32), &hipMemsetD32>("hipMemsetD32",
                                                            dest, value, count);
}

hipError_t hipMemsetAsync(void* dst, int value, size_t sizeBytes,
                          hipStream_t stream __dparm(0)) {
  return DynamicCall<decltype(hipMemsetAsync), &hipMemsetAsync>(
      "hipMemsetAsync", dst, value, sizeBytes, stream);
}

hipError_t hipMemsetD32Async(hipDeviceptr_t dst, int value, size_t count,
                             hipStream_t stream __dparm(0)) {
  return DynamicCall<decltype(hipMemsetD32Async), &hipMemsetD32Async>(
      "hipMemsetD32Async", dst, value, count, stream);
}

hipError_t hipMemGetInfo(size_t* free, size_t* total) {
  return DynamicCall<decltype(hipMemGetInfo), &hipMemGetInfo>("hipMemGetInfo",
                                                              free, total);
}

hipError_t hipDeviceCanAccessPeer(int* canAccessPeer, int deviceId,
                                  int peerDeviceId) {
  return DynamicCall<decltype(hipDeviceCanAccessPeer), &hipDeviceCanAccessPeer>(
      "hipDeviceCanAccessPeer", canAccessPeer, deviceId, peerDeviceId);
}

hipError_t hipDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags) {
  return DynamicCall<decltype(hipDeviceEnablePeerAccess),
                     &hipDeviceEnablePeerAccess>("hipDeviceEnablePeerAccess",
                                                 peerDeviceId, flags);
}

hipError_t hipDeviceDisablePeerAccess(int peerDeviceId) {
  return DynamicCall<decltype(hipDeviceDisablePeerAccess),
                     &hipDeviceDisablePeerAccess>("hipDeviceDisablePeerAccess",
                                                  peerDeviceId);
}

hipError_t hipMemGetAddressRange(hipDeviceptr_t* pbase, size_t* psize,
                                 hipDeviceptr_t dptr) {
  return DynamicCall<decltype(hipMemGetAddressRange), &hipMemGetAddressRange>(
      "hipMemGetAddressRange", pbase, psize, dptr);
}

hipError_t hipMemcpyPeer(void* dst, int dstDeviceId, const void* src,
                         int srcDeviceId, size_t sizeBytes) {
  return DynamicCall<decltype(hipMemcpyPeer), &hipMemcpyPeer>(
      "hipMemcpyPeer", dst, dstDeviceId, src, srcDeviceId, sizeBytes);
}

hipError_t hipMemcpyPeerAsync(void* dst, int dstDeviceId, const void* src,
                              int srcDevice, size_t sizeBytes,
                              hipStream_t stream __dparm(0)) {
  return DynamicCall<decltype(hipMemcpyPeerAsync), &hipMemcpyPeerAsync>(
      "hipMemcpyPeerAsync", dst, dstDeviceId, src, srcDevice, sizeBytes,
      stream);
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxCreate(hipCtx_t* ctx, unsigned int flags, hipDevice_t device) {
  return DynamicCall<decltype(hipCtxCreate), &hipCtxCreate>("hipCtxCreate", ctx,
                                                            flags, device);
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxDestroy(hipCtx_t ctx) {
  return DynamicCall<decltype(hipCtxDestroy), &hipCtxDestroy>("hipCtxDestroy",
                                                              ctx);
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxPopCurrent(hipCtx_t* ctx) {
  return DynamicCall<decltype(hipCtxPopCurrent), &hipCtxPopCurrent>(
      "hipCtxPopCurrent", ctx);
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxPushCurrent(hipCtx_t ctx) {
  return DynamicCall<decltype(hipCtxPushCurrent), &hipCtxPushCurrent>(
      "hipCtxPushCurrent", ctx);
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxSetCurrent(hipCtx_t ctx) {
  return DynamicCall<decltype(hipCtxSetCurrent), &hipCtxSetCurrent>(
      "hipCtxSetCurrent", ctx);
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxGetCurrent(hipCtx_t* ctx) {
  return DynamicCall<decltype(hipCtxGetCurrent), &hipCtxGetCurrent>(
      "hipCtxGetCurrent", ctx);
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxGetDevice(hipDevice_t* device) {
  return DynamicCall<decltype(hipCtxGetDevice), &hipCtxGetDevice>(
      "hipCtxGetDevice", device);
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxGetApiVersion(hipCtx_t ctx, int* apiVersion) {
  return DynamicCall<decltype(hipCtxGetApiVersion), &hipCtxGetApiVersion>(
      "hipCtxGetApiVersion", ctx, apiVersion);
}

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxGetFlags(unsigned int* flags) {
  return DynamicCall<decltype(hipCtxGetFlags), &hipCtxGetFlags>(
      "hipCtxGetFlags", flags);
}

hipError_t hipDevicePrimaryCtxGetState(hipDevice_t dev, unsigned int* flags,
                                       int* active) {
  return DynamicCall<decltype(hipDevicePrimaryCtxGetState),
                     &hipDevicePrimaryCtxGetState>(
      "hipDevicePrimaryCtxGetState", dev, flags, active);
}

hipError_t hipDevicePrimaryCtxRelease(hipDevice_t dev) {
  return DynamicCall<decltype(hipDevicePrimaryCtxRelease),
                     &hipDevicePrimaryCtxRelease>("hipDevicePrimaryCtxRelease",
                                                  dev);
}

hipError_t hipDevicePrimaryCtxRetain(hipCtx_t* pctx, hipDevice_t dev) {
  return DynamicCall<decltype(hipDevicePrimaryCtxRetain),
                     &hipDevicePrimaryCtxRetain>("hipDevicePrimaryCtxRetain",
                                                 pctx, dev);
}

hipError_t hipDevicePrimaryCtxReset(hipDevice_t dev) {
  return DynamicCall<decltype(hipDevicePrimaryCtxReset),
                     &hipDevicePrimaryCtxReset>("hipDevicePrimaryCtxReset",
                                                dev);
}

hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t dev, unsigned int flags) {
  return DynamicCall<decltype(hipDevicePrimaryCtxSetFlags),
                     &hipDevicePrimaryCtxSetFlags>(
      "hipDevicePrimaryCtxSetFlags", dev, flags);
}

hipError_t hipModuleUnload(hipModule_t module) {
  return DynamicCall<decltype(hipModuleUnload), &hipModuleUnload>(
      "hipModuleUnload", module);
}

hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module,
                                const char* kname) {
  return DynamicCall<decltype(hipModuleGetFunction), &hipModuleGetFunction>(
      "hipModuleGetFunction", function, module, kname);
}

hipError_t hipModuleGetGlobal(void** ptr, size_t* bytes, hipModule_t module,
                              const char* kname) {
  return DynamicCall<decltype(hipModuleGetGlobal), &hipModuleGetGlobal>(
      "hipModuleGetGlobal", ptr, bytes, module, kname);
}


hipError_t hipFuncGetAttributes(struct hipFuncAttributes* attr,
                                const void* func) {
  return DynamicCall<decltype(hipFuncGetAttributes), &hipFuncGetAttributes>(
      "hipFuncGetAttributes", attr, func);
}

hipError_t hipModuleLoadData(hipModule_t* module, const void* image) {
  return DynamicCall<decltype(hipModuleLoadData), &hipModuleLoadData>(
      "hipModuleLoadData", module, image);
}

hipError_t hipModuleLoadDataEx(hipModule_t* module, const void* image,
                               unsigned int numOptions, hipJitOption* options,
                               void** optionValues) {
  return DynamicCall<decltype(hipModuleLoadDataEx), &hipModuleLoadDataEx>(
      "hipModuleLoadDataEx", module, image, numOptions, options, optionValues);
}

hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX,
                                 unsigned int gridDimY, unsigned int gridDimZ,
                                 unsigned int blockDimX, unsigned int blockDimY,
                                 unsigned int blockDimZ,
                                 unsigned int sharedMemBytes,
                                 hipStream_t stream, void** kernelParams,
                                 void** extra) {
  return DynamicCall<decltype(hipModuleLaunchKernel), &hipModuleLaunchKernel>(
      "hipModuleLaunchKernel", f, gridDimX, gridDimY, gridDimZ, blockDimX,
      blockDimY, blockDimZ, sharedMemBytes, stream, kernelParams, extra);
}

hipError_t hipLaunchCooperativeKernel(const void* f, dim3 gridDim,
                                      dim3 blockDimX, void** kernelParams,
                                      unsigned int sharedMemBytes,
                                      hipStream_t stream) {
  return DynamicCall<decltype(hipLaunchCooperativeKernel),
                     &hipLaunchCooperativeKernel>(
      "hipLaunchCooperativeKernel", f, gridDim, blockDimX, kernelParams,
      sharedMemBytes, stream);
}

hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(
    int* numBlocks, const void* f, int blockSize, size_t dynSharedMemPerBlk) {
  return DynamicCall<decltype(hipOccupancyMaxActiveBlocksPerMultiprocessor),
                     &hipOccupancyMaxActiveBlocksPerMultiprocessor>(
      "hipOccupancyMaxActiveBlocksPerMultiprocessor", numBlocks, f, blockSize,
      dynSharedMemPerBlk);
}

hipError_t hipOccupancyMaxPotentialBlockSize(int* gridSize, int* blockSize,
                                             const void* f,
                                             size_t dynSharedMemPerBlk,
                                             int blockSizeLimit) {
  return DynamicCall<decltype(hipOccupancyMaxPotentialBlockSize),
                     &hipOccupancyMaxPotentialBlockSize>(
      "hipOccupancyMaxPotentialBlockSize", gridSize, blockSize, f,
      dynSharedMemPerBlk, blockSizeLimit);
}
