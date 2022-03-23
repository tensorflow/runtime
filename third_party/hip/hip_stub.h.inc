enum hipMemoryType {
  hipMemoryTypeHost,
  hipMemoryTypeDevice,
  hipMemoryTypeArray,
  hipMemoryTypeUnified,
};

enum hipError_t {
  hipSuccess = 0,
  hipErrorInvalidValue = 1,
  hipErrorOutOfMemory = 2,
  hipErrorMemoryAllocation = 2,
  hipErrorNotInitialized = 3,
  hipErrorInitializationError = 3,
  hipErrorDeinitialized = 4,
  hipErrorProfilerDisabled = 5,
  hipErrorProfilerNotInitialized = 6,
  hipErrorProfilerAlreadyStarted = 7,
  hipErrorProfilerAlreadyStopped = 8,
  hipErrorInvalidConfiguration = 9,
  hipErrorInvalidPitchValue = 12,
  hipErrorInvalidSymbol = 13,
  hipErrorInvalidDevicePointer = 17,
  hipErrorInvalidMemcpyDirection = 21,
  hipErrorInsufficientDriver = 35,
  hipErrorMissingConfiguration = 52,
  hipErrorPriorLaunchFailure = 53,
  hipErrorInvalidDeviceFunction = 98,
  hipErrorNoDevice = 100,
  hipErrorInvalidDevice = 101,
  hipErrorInvalidImage = 200,
  hipErrorInvalidContext = 201,
  hipErrorContextAlreadyCurrent = 202,
  hipErrorMapFailed = 205,
  hipErrorMapBufferObjectFailed = 205,
  hipErrorUnmapFailed = 206,
  hipErrorArrayIsMapped = 207,
  hipErrorAlreadyMapped = 208,
  hipErrorNoBinaryForGpu = 209,
  hipErrorAlreadyAcquired = 210,
  hipErrorNotMapped = 211,
  hipErrorNotMappedAsArray = 212,
  hipErrorNotMappedAsPointer = 213,
  hipErrorECCNotCorrectable = 214,
  hipErrorUnsupportedLimit = 215,
  hipErrorContextAlreadyInUse = 216,
  hipErrorPeerAccessUnsupported = 217,
  hipErrorInvalidKernelFile = 218,
  hipErrorInvalidGraphicsContext = 219,
  hipErrorInvalidSource = 300,
  hipErrorFileNotFound = 301,
  hipErrorSharedObjectSymbolNotFound = 302,
  hipErrorSharedObjectInitFailed = 303,
  hipErrorOperatingSystem = 304,
  hipErrorInvalidHandle = 400,
  hipErrorInvalidResourceHandle = 400,
  hipErrorIllegalState = 401,
  hipErrorNotFound = 500,
  hipErrorNotReady = 600,
  hipErrorIllegalAddress = 700,
  hipErrorLaunchOutOfResources = 701,
  hipErrorLaunchTimeOut = 702,
  hipErrorPeerAccessAlreadyEnabled = 704,
  hipErrorPeerAccessNotEnabled = 705,
  hipErrorSetOnActiveProcess = 708,
  hipErrorContextIsDestroyed = 709,
  hipErrorAssert = 710,
  hipErrorHostMemoryAlreadyRegistered = 712,
  hipErrorHostMemoryNotRegistered = 713,
  hipErrorLaunchFailure = 719,
  hipErrorCooperativeLaunchTooLarge = 720,
  hipErrorNotSupported = 801,
  hipErrorStreamCaptureUnsupported = 900,
  hipErrorStreamCaptureInvalidated = 901,
  hipErrorStreamCaptureMerge = 902,
  hipErrorStreamCaptureUnmatched = 903,
  hipErrorStreamCaptureUnjoined = 904,
  hipErrorStreamCaptureIsolation = 905,
  hipErrorStreamCaptureImplicit = 906,
  hipErrorCapturedEvent = 907,
  hipErrorStreamCaptureWrongThread = 908,
  hipErrorGraphExecUpdateFailure = 910,
  hipErrorUnknown = 999,
  hipErrorRuntimeMemory = 1052,
  hipErrorRuntimeOther = 1053,
  hipErrorTbd,
};

enum hipDeviceAttribute_t {
  hipDeviceAttributeCudaCompatibleBegin = 0,
  hipDeviceAttributeEccEnabled = hipDeviceAttributeCudaCompatibleBegin,
  hipDeviceAttributeAccessPolicyMaxWindowSize,
  hipDeviceAttributeAsyncEngineCount,
  hipDeviceAttributeCanMapHostMemory,
  hipDeviceAttributeCanUseHostPointerForRegisteredMem,
  hipDeviceAttributeClockRate,
  hipDeviceAttributeComputeMode,
  hipDeviceAttributeComputePreemptionSupported,
  hipDeviceAttributeConcurrentKernels,
  hipDeviceAttributeConcurrentManagedAccess,
  hipDeviceAttributeCooperativeLaunch,
  hipDeviceAttributeCooperativeMultiDeviceLaunch,
  hipDeviceAttributeDeviceOverlap,
  hipDeviceAttributeDirectManagedMemAccessFromHost,
  hipDeviceAttributeGlobalL1CacheSupported,
  hipDeviceAttributeHostNativeAtomicSupported,
  hipDeviceAttributeIntegrated,
  hipDeviceAttributeIsMultiGpuBoard,
  hipDeviceAttributeKernelExecTimeout,
  hipDeviceAttributeL2CacheSize,
  hipDeviceAttributeLocalL1CacheSupported,
  hipDeviceAttributeLuid,
  hipDeviceAttributeLuidDeviceNodeMask,
  hipDeviceAttributeComputeCapabilityMajor,
  hipDeviceAttributeManagedMemory,
  hipDeviceAttributeMaxBlocksPerMultiProcessor,
  hipDeviceAttributeMaxBlockDimX,
  hipDeviceAttributeMaxBlockDimY,
  hipDeviceAttributeMaxBlockDimZ,
  hipDeviceAttributeMaxGridDimX,
  hipDeviceAttributeMaxGridDimY,
  hipDeviceAttributeMaxGridDimZ,
  hipDeviceAttributeMaxSurface1D,
  hipDeviceAttributeMaxSurface1DLayered,
  hipDeviceAttributeMaxSurface2D,
  hipDeviceAttributeMaxSurface2DLayered,
  hipDeviceAttributeMaxSurface3D,
  hipDeviceAttributeMaxSurfaceCubemap,
  hipDeviceAttributeMaxSurfaceCubemapLayered,
  hipDeviceAttributeMaxTexture1DWidth,
  hipDeviceAttributeMaxTexture1DLayered,
  hipDeviceAttributeMaxTexture1DLinear,
  hipDeviceAttributeMaxTexture1DMipmap,
  hipDeviceAttributeMaxTexture2DWidth,
  hipDeviceAttributeMaxTexture2DHeight,
  hipDeviceAttributeMaxTexture2DGather,
  hipDeviceAttributeMaxTexture2DLayered,
  hipDeviceAttributeMaxTexture2DLinear,
  hipDeviceAttributeMaxTexture2DMipmap,
  hipDeviceAttributeMaxTexture3DWidth,
  hipDeviceAttributeMaxTexture3DHeight,
  hipDeviceAttributeMaxTexture3DDepth,
  hipDeviceAttributeMaxTexture3DAlt,
  hipDeviceAttributeMaxTextureCubemap,
  hipDeviceAttributeMaxTextureCubemapLayered,
  hipDeviceAttributeMaxThreadsDim,
  hipDeviceAttributeMaxThreadsPerBlock,
  hipDeviceAttributeMaxThreadsPerMultiProcessor,
  hipDeviceAttributeMaxPitch,
  hipDeviceAttributeMemoryBusWidth,
  hipDeviceAttributeMemoryClockRate,
  hipDeviceAttributeComputeCapabilityMinor,
  hipDeviceAttributeMultiGpuBoardGroupID,
  hipDeviceAttributeMultiprocessorCount,
  hipDeviceAttributeName,
  hipDeviceAttributePageableMemoryAccess,
  hipDeviceAttributePageableMemoryAccessUsesHostPageTables,
  hipDeviceAttributePciBusId,
  hipDeviceAttributePciDeviceId,
  hipDeviceAttributePciDomainID,
  hipDeviceAttributePersistingL2CacheMaxSize,
  hipDeviceAttributeMaxRegistersPerBlock,
  hipDeviceAttributeMaxRegistersPerMultiprocessor,
  hipDeviceAttributeReservedSharedMemPerBlock,
  hipDeviceAttributeMaxSharedMemoryPerBlock,
  hipDeviceAttributeSharedMemPerBlockOptin,
  hipDeviceAttributeSharedMemPerMultiprocessor,
  hipDeviceAttributeSingleToDoublePrecisionPerfRatio,
  hipDeviceAttributeStreamPrioritiesSupported,
  hipDeviceAttributeSurfaceAlignment,
  hipDeviceAttributeTccDriver,
  hipDeviceAttributeTextureAlignment,
  hipDeviceAttributeTexturePitchAlignment,
  hipDeviceAttributeTotalConstantMemory,
  hipDeviceAttributeTotalGlobalMem,
  hipDeviceAttributeUnifiedAddressing,
  hipDeviceAttributeUuid,
  hipDeviceAttributeWarpSize,
  hipDeviceAttributeCudaCompatibleEnd = 9999,
  hipDeviceAttributeAmdSpecificBegin = 10000,
  hipDeviceAttributeClockInstructionRate = hipDeviceAttributeAmdSpecificBegin,
  hipDeviceAttributeArch,
  hipDeviceAttributeMaxSharedMemoryPerMultiprocessor,
  hipDeviceAttributeGcnArch,
  hipDeviceAttributeGcnArchName,
  hipDeviceAttributeHdpMemFlushCntl,
  hipDeviceAttributeHdpRegFlushCntl,
  hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc,
  hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim,
  hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim,
  hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem,
  hipDeviceAttributeIsLargeBar,
  hipDeviceAttributeAsicRevision,
  hipDeviceAttributeCanUseStreamWaitValue,
  hipDeviceAttributeImageSupport,
  hipDeviceAttributeAmdSpecificEnd = 19999,
  hipDeviceAttributeVendorSpecificBegin = 20000,
};

enum hipMemcpyKind {
  hipMemcpyHostToHost = 0,
  hipMemcpyHostToDevice = 1,
  hipMemcpyDeviceToHost = 2,
  hipMemcpyDeviceToDevice = 3,
  hipMemcpyDefault = 4,
};

enum hipFunction_attribute {
  HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
  HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
  HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
  HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
  HIP_FUNC_ATTRIBUTE_NUM_REGS,
  HIP_FUNC_ATTRIBUTE_PTX_VERSION,
  HIP_FUNC_ATTRIBUTE_BINARY_VERSION,
  HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA,
  HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
  HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
  HIP_FUNC_ATTRIBUTE_MAX,
};

enum hipLimit_t {
  hipLimitPrintfFifoSize = 0x01,
  hipLimitMallocHeapSize = 0x02,
};

enum hipJitOption {
  hipJitOptionMaxRegisters = 0,
  hipJitOptionThreadsPerBlock,
  hipJitOptionWallTime,
  hipJitOptionInfoLogBuffer,
  hipJitOptionInfoLogBufferSizeBytes,
  hipJitOptionErrorLogBuffer,
  hipJitOptionErrorLogBufferSizeBytes,
  hipJitOptionOptimizationLevel,
  hipJitOptionTargetFromContext,
  hipJitOptionTarget,
  hipJitOptionFallbackStrategy,
  hipJitOptionGenerateDebugInfo,
  hipJitOptionLogVerbose,
  hipJitOptionGenerateLineInfo,
  hipJitOptionCacheMode,
  hipJitOptionSm3xOpt,
  hipJitOptionFastCompile,
  hipJitOptionNumOptions,
};

enum hipFuncCache_t {
  hipFuncCachePreferNone,
  hipFuncCachePreferShared,
  hipFuncCachePreferL1,
  hipFuncCachePreferEqual,
};

enum hipSharedMemConfig {
  hipSharedMemBankSizeDefault,
  hipSharedMemBankSizeFourByte,
  hipSharedMemBankSizeEightByte,
};

hipError_t hipInit(unsigned int flags);

hipError_t hipDriverGetVersion(int* driverVersion);

hipError_t hipRuntimeGetVersion(int* runtimeVersion);

hipError_t hipDeviceGet(hipDevice_t* device, int ordinal);

hipError_t hipDeviceGetName(char* name, int len, hipDevice_t device);

hipError_t hipDeviceGetPCIBusId(char* pciBusId, int len, int device);

hipError_t hipDeviceTotalMem(size_t* bytes, hipDevice_t device);

hipError_t hipDeviceSynchronize(void);

hipError_t hipGetDevice(int* deviceId);

hipError_t hipGetDeviceCount(int* count);

hipError_t hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr,
                                 int deviceId);

hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int deviceId);

hipError_t hipDeviceGetLimit(size_t* pValue, enum hipLimit_t limit);

hipError_t hipGetLastError(void);

hipError_t hipPeekAtLastError(void);

hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags);

hipError_t hipStreamCreateWithPriority(hipStream_t* stream, unsigned int flags,
                                       int priority);

hipError_t hipDeviceGetStreamPriorityRange(int* leastPriority,
                                           int* greatestPriority);

hipError_t hipStreamDestroy(hipStream_t stream);

hipError_t hipStreamQuery(hipStream_t stream);

hipError_t hipStreamSynchronize(hipStream_t stream);

hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event,
                              unsigned int flags);

hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int* flags);

hipError_t hipStreamGetPriority(hipStream_t stream, int* priority);

hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned flags);

hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream);

hipError_t hipEventDestroy(hipEvent_t event);

hipError_t hipEventSynchronize(hipEvent_t event);

hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop);

hipError_t hipEventQuery(hipEvent_t event);

hipError_t hipPointerGetAttributes(hipPointerAttribute_t* attributes,
                                   const void* ptr);

hipError_t hipMalloc(void** ptr, size_t size);

hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags);

hipError_t hipMallocManaged(void** dev_ptr, size_t size,
                            unsigned int flags __dparm(hipMemAttachGlobal));

hipError_t hipHostGetDevicePointer(void** devPtr, void* hstPtr,
                                   unsigned int flags);

hipError_t hipHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags);

hipError_t hipHostUnregister(void* hostPtr);

hipError_t hipFree(void* ptr);

hipError_t hipHostFree(void* ptr);

hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes,
                     hipMemcpyKind kind);

hipError_t hipModuleGetGlobal(hipDeviceptr_t* dptr, size_t* bytes,
                              hipModule_t hmod, const char* name);

hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes,
                          hipMemcpyKind kind, hipStream_t stream __dparm(0));

hipError_t hipMemset(void* dst, int value, size_t sizeBytes);

hipError_t hipMemsetD32(hipDeviceptr_t dest, int value, size_t count);

hipError_t hipMemsetAsync(void* dst, int value, size_t sizeBytes,
                          hipStream_t stream __dparm(0));

hipError_t hipMemsetD32Async(hipDeviceptr_t dst, int value, size_t count,
                             hipStream_t stream __dparm(0));

hipError_t hipMemGetInfo(size_t* free, size_t* total);

hipError_t hipDeviceCanAccessPeer(int* canAccessPeer, int deviceId,
                                  int peerDeviceId);

hipError_t hipDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags);

hipError_t hipDeviceDisablePeerAccess(int peerDeviceId);

hipError_t hipMemGetAddressRange(hipDeviceptr_t* pbase, size_t* psize,
                                 hipDeviceptr_t dptr);

hipError_t hipMemcpyPeer(void* dst, int dstDeviceId, const void* src,
                         int srcDeviceId, size_t sizeBytes);

hipError_t hipMemcpyPeerAsync(void* dst, int dstDeviceId, const void* src,
                              int srcDevice, size_t sizeBytes,
                              hipStream_t stream __dparm(0));

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxCreate(hipCtx_t* ctx, unsigned int flags, hipDevice_t device);

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxDestroy(hipCtx_t ctx);

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxPopCurrent(hipCtx_t* ctx);

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxPushCurrent(hipCtx_t ctx);

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxSetCurrent(hipCtx_t ctx);

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxGetCurrent(hipCtx_t* ctx);

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxGetDevice(hipDevice_t* device);

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxGetApiVersion(hipCtx_t ctx, int* apiVersion);

DEPRECATED(DEPRECATED_MSG)
hipError_t hipCtxGetFlags(unsigned int* flags);

hipError_t hipDevicePrimaryCtxGetState(hipDevice_t dev, unsigned int* flags,
                                       int* active);

hipError_t hipDevicePrimaryCtxRelease(hipDevice_t dev);

hipError_t hipDevicePrimaryCtxRetain(hipCtx_t* pctx, hipDevice_t dev);

hipError_t hipDevicePrimaryCtxReset(hipDevice_t dev);

hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t dev, unsigned int flags);

hipError_t hipModuleUnload(hipModule_t module);

hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module,
                                const char* kname);

hipError_t hipFuncGetAttributes(struct hipFuncAttributes* attr,
                                const void* func);

hipError_t hipModuleLoadData(hipModule_t* module, const void* image);

hipError_t hipModuleLoadDataEx(hipModule_t* module, const void* image,
                               unsigned int numOptions, hipJitOption* options,
                               void** optionValues);

hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX,
                                 unsigned int gridDimY, unsigned int gridDimZ,
                                 unsigned int blockDimX, unsigned int blockDimY,
                                 unsigned int blockDimZ,
                                 unsigned int sharedMemBytes,
                                 hipStream_t stream, void** kernelParams,
                                 void** extra);

hipError_t hipLaunchCooperativeKernel(const void* f, dim3 gridDim,
                                      dim3 blockDimX, void** kernelParams,
                                      unsigned int sharedMemBytes,
                                      hipStream_t stream);

hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(
    int* numBlocks, const void* f, int blockSize, size_t dynSharedMemPerBlk);

hipError_t hipOccupancyMaxPotentialBlockSize(int* gridSize, int* blockSize,
                                             const void* f,
                                             size_t dynSharedMemPerBlk,
                                             int blockSizeLimit);
enum hiprtcResult {
  HIPRTC_SUCCESS = 0,
  HIPRTC_ERROR_OUT_OF_MEMORY = 1,
  HIPRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
  HIPRTC_ERROR_INVALID_INPUT = 3,
  HIPRTC_ERROR_INVALID_PROGRAM = 4,
  HIPRTC_ERROR_INVALID_OPTION = 5,
  HIPRTC_ERROR_COMPILATION = 6,
  HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
  HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
  HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
  HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
  HIPRTC_ERROR_INTERNAL_ERROR = 11,
};
