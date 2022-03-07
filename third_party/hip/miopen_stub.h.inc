typedef enum {
  miopenStatusSuccess = 0,
  miopenStatusNotInitialized = 1,
  miopenStatusInvalidValue = 2,
  miopenStatusBadParm = 3,
  miopenStatusAllocFailed = 4,
  miopenStatusInternalError = 5,
  miopenStatusNotImplemented = 6,
  miopenStatusUnknownError = 7,
  miopenStatusUnsupportedOp = 8,
  miopenStatusGpuOperationsSkipped = 9,
} miopenStatus_t;

MIOPEN_EXPORT miopenStatus_t miopenGetVersion(size_t* major, size_t* minor,
                                              size_t* patch);

MIOPEN_EXPORT miopenStatus_t miopenCreate(miopenHandle_t* handle);

MIOPEN_EXPORT miopenStatus_t
miopenCreateWithStream(miopenHandle_t* handle, miopenAcceleratorQueue_t stream);

MIOPEN_EXPORT miopenStatus_t miopenDestroy(miopenHandle_t handle);

MIOPEN_EXPORT miopenStatus_t miopenSetStream(miopenHandle_t handle,
                                             miopenAcceleratorQueue_t streamID);

MIOPEN_EXPORT miopenStatus_t
miopenGetStream(miopenHandle_t handle, miopenAcceleratorQueue_t* streamID);

MIOPEN_EXPORT miopenStatus_t miopenSetAllocator(
    miopenHandle_t handle, miopenAllocatorFunction allocator,
    miopenDeallocatorFunction deallocator, void* allocatorContext);

MIOPEN_EXPORT miopenStatus_t miopenGetKernelTime(miopenHandle_t handle,
                                                 float* time);

MIOPEN_EXPORT miopenStatus_t miopenEnableProfiling(miopenHandle_t handle,
                                                   bool enable);

typedef enum {
  miopenHalf = 0,
  miopenFloat = 1,
  miopenInt32 = 2,
  miopenInt8 = 3,
  miopenInt8x4 = 4,
  miopenBFloat16 = 5,
  miopenDouble = 6,
} miopenDataType_t;

typedef enum {
  miopenIndexUint8 = 0,
  miopenIndexUint16 = 1,
  miopenIndexUint32 = 2,
  miopenIndexUint64 = 3,
} miopenIndexType_t;

typedef enum {
  miopenTensorOpAdd = 0,
  miopenTensorOpMul = 1,
  miopenTensorOpMin = 2,
  miopenTensorOpMax = 3,
} miopenTensorOp_t;

typedef enum {
  miopenConvolution = 0,
  miopenTranspose = 1,
  miopenGroupConv = 2,
  miopenDepthwise = 3,
} miopenConvolutionMode_t;

typedef enum {
  miopenPaddingDefault = 0,
  miopenPaddingSame = 1,
  miopenPaddingValid = 2,
} miopenPaddingMode_t;

typedef enum {
  miopenPoolingMax = 0,
  miopenPoolingAverage = 1,
  miopenPoolingAverageInclusive = 2,
} miopenPoolingMode_t;

typedef enum {
  miopenPoolingWorkspaceIndexMask = 0,
  miopenPoolingWorkspaceIndexImage = 1,
} miopenPoolingWorkspaceIndexMode_t;

typedef enum {
  miopenLRNWithinChannel = 0,
  miopenLRNCrossChannel = 1,
} miopenLRNMode_t;

typedef enum {
  miopenBNPerActivation = 0,
  miopenBNSpatial = 1,
} miopenBatchNormMode_t;

typedef enum {
  miopenActivationPASTHRU = 0,
  miopenActivationLOGISTIC = 1,
  miopenActivationTANH = 2,
  miopenActivationRELU = 3,
  miopenActivationSOFTRELU = 4,
  miopenActivationABS = 5,
  miopenActivationPOWER = 6,
  miopenActivationCLIPPEDRELU = 7,
  miopenActivationLEAKYRELU = 8,
  miopenActivationELU = 9,
} miopenActivationMode_t;

typedef enum {
  MIOPEN_SOFTMAX_FAST = 0,
  MIOPEN_SOFTMAX_ACCURATE = 1,
  MIOPEN_SOFTMAX_LOG = 2,
} miopenSoftmaxAlgorithm_t;

typedef enum {
  MIOPEN_SOFTMAX_MODE_INSTANCE = 0,
  MIOPEN_SOFTMAX_MODE_CHANNEL = 1,
} miopenSoftmaxMode_t;

MIOPEN_EXPORT miopenStatus_t
miopenCreateTensorDescriptor(miopenTensorDescriptor_t* tensorDesc);

MIOPEN_EXPORT miopenStatus_t miopenSet4dTensorDescriptor(
    miopenTensorDescriptor_t tensorDesc, miopenDataType_t dataType, int n,
    int c, int h, int w);

MIOPEN_EXPORT miopenStatus_t miopenGet4dTensorDescriptor(
    miopenTensorDescriptor_t tensorDesc, miopenDataType_t* dataType, int* n,
    int* c, int* h, int* w, int* nStride, int* cStride, int* hStride,
    int* wStride);

MIOPEN_EXPORT miopenStatus_t miopenSetTensorDescriptor(
    miopenTensorDescriptor_t tensorDesc, miopenDataType_t dataType, int nbDims,
    int* dimsA, int* stridesA);

MIOPEN_EXPORT miopenStatus_t
miopenGetTensorDescriptorSize(miopenTensorDescriptor_t tensorDesc, int* size);

MIOPEN_EXPORT miopenStatus_t miopenGetTensorDescriptor(
    miopenTensorDescriptor_t tensorDesc, miopenDataType_t* dataType, int* dimsA,
    int* stridesA);

MIOPEN_EXPORT miopenStatus_t
miopenDestroyTensorDescriptor(miopenTensorDescriptor_t tensorDesc);

MIOPEN_EXPORT miopenStatus_t miopenOpTensor(
    miopenHandle_t handle, miopenTensorOp_t tensorOp, const void* alpha1,
    const miopenTensorDescriptor_t aDesc, const void* A, const void* alpha2,
    const miopenTensorDescriptor_t bDesc, const void* B, const void* beta,
    const miopenTensorDescriptor_t cDesc, void* C);

MIOPEN_EXPORT miopenStatus_t
miopenSetTensor(miopenHandle_t handle, const miopenTensorDescriptor_t yDesc,
                void* y, const void* alpha);

MIOPEN_EXPORT miopenStatus_t
miopenScaleTensor(miopenHandle_t handle, const miopenTensorDescriptor_t yDesc,
                  void* y, const void* alpha);

MIOPEN_EXPORT miopenStatus_t
miopenGetTensorNumBytes(miopenTensorDescriptor_t tensorDesc, size_t* numBytes);

MIOPEN_EXPORT miopenStatus_t miopenTransformTensor(
    miopenHandle_t handle, const void* alpha,
    const miopenTensorDescriptor_t xDesc, const void* x, const void* beta,
    const miopenTensorDescriptor_t yDesc, void* y);

MIOPEN_EXPORT miopenStatus_t
miopenCreateConvolutionDescriptor(miopenConvolutionDescriptor_t* convDesc);

MIOPEN_EXPORT miopenStatus_t miopenInitConvolutionDescriptor(
    miopenConvolutionDescriptor_t convDesc, miopenConvolutionMode_t c_mode,
    int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h,
    int dilation_w);

MIOPEN_EXPORT miopenStatus_t miopenInitConvolutionNdDescriptor(
    miopenConvolutionDescriptor_t convDesc, int spatialDim, int* padA,
    int* strideA, int* dilationA, miopenConvolutionMode_t c_mode);

MIOPEN_EXPORT miopenStatus_t miopenGetConvolutionDescriptor(
    miopenConvolutionDescriptor_t convDesc, miopenConvolutionMode_t* c_mode,
    int* pad_h, int* pad_w, int* stride_h, int* stride_w, int* dilation_h,
    int* dilation_w);

MIOPEN_EXPORT miopenStatus_t miopenGetConvolutionNdDescriptor(
    miopenConvolutionDescriptor_t convDesc, int requestedSpatialDim,
    int* spatialDim, int* padA, int* strideA, int* dilationA,
    miopenConvolutionMode_t* c_mode);

MIOPEN_EXPORT miopenStatus_t miopenSetConvolutionGroupCount(
    miopenConvolutionDescriptor_t convDesc, int groupCount);

MIOPEN_EXPORT miopenStatus_t miopenSetTransposeConvOutputPadding(
    miopenConvolutionDescriptor_t convDesc, int adj_h, int adj_w);

MIOPEN_EXPORT miopenStatus_t miopenSetTransposeConvNdOutputPadding(
    miopenConvolutionDescriptor_t convDesc, int spatialDim, int* adjA);

MIOPEN_EXPORT miopenStatus_t miopenGetConvolutionForwardOutputDim(
    miopenConvolutionDescriptor_t convDesc,
    const miopenTensorDescriptor_t inputTensorDesc,
    const miopenTensorDescriptor_t filterDesc, int* n, int* c, int* h, int* w);

MIOPEN_EXPORT miopenStatus_t miopenGetConvolutionNdForwardOutputDim(
    miopenConvolutionDescriptor_t convDesc,
    const miopenTensorDescriptor_t inputTensorDesc,
    const miopenTensorDescriptor_t filterDesc, int* nDim,
    int* outputTensorDimA);

MIOPEN_EXPORT miopenStatus_t
miopenDestroyConvolutionDescriptor(miopenConvolutionDescriptor_t convDesc);

typedef enum {
  miopenConvolutionFwdAlgoGEMM = 0,
  miopenConvolutionFwdAlgoDirect = 1,
  miopenConvolutionFwdAlgoFFT = 2,
  miopenConvolutionFwdAlgoWinograd = 3,
  miopenConvolutionFwdAlgoImplicitGEMM = 5,
} miopenConvFwdAlgorithm_t;

typedef enum {
  miopenConvolutionBwdWeightsAlgoGEMM = 0,
  miopenConvolutionBwdWeightsAlgoDirect = 1,
  miopenConvolutionBwdWeightsAlgoWinograd = 3,
  miopenConvolutionBwdWeightsAlgoImplicitGEMM = 5,
} miopenConvBwdWeightsAlgorithm_t;

typedef enum {
  miopenConvolutionBwdDataAlgoGEMM = 0,
  miopenConvolutionBwdDataAlgoDirect = 1,
  miopenConvolutionBwdDataAlgoFFT = 2,
  miopenConvolutionBwdDataAlgoWinograd = 3,
  miopenTransposeBwdDataAlgoGEMM = 4,
  miopenConvolutionBwdDataAlgoImplicitGEMM = 5,
} miopenConvBwdDataAlgorithm_t;

typedef enum {
  miopenConvolutionAlgoGEMM = 0,
  miopenConvolutionAlgoDirect = 1,
  miopenConvolutionAlgoFFT = 2,
  miopenConvolutionAlgoWinograd = 3,
  miopenConvolutionAlgoImplicitGEMM = 5,
} miopenConvAlgorithm_t;

MIOPEN_EXPORT miopenStatus_t miopenConvolutionForwardGetSolutionCount(
    miopenHandle_t handle, const miopenTensorDescriptor_t wDesc,
    const miopenTensorDescriptor_t xDesc,
    const miopenConvolutionDescriptor_t convDesc,
    const miopenTensorDescriptor_t yDesc, size_t* solutionCount);

MIOPEN_EXPORT miopenStatus_t miopenConvolutionForwardGetSolution(
    miopenHandle_t handle, const miopenTensorDescriptor_t wDesc,
    const miopenTensorDescriptor_t xDesc,
    const miopenConvolutionDescriptor_t convDesc,
    const miopenTensorDescriptor_t yDesc, const size_t maxSolutionCount,
    size_t* solutionCount, miopenConvSolution_t* solutions);

MIOPEN_EXPORT miopenStatus_t miopenConvolutionForwardGetSolutionWorkspaceSize(
    miopenHandle_t handle, const miopenTensorDescriptor_t wDesc,
    const miopenTensorDescriptor_t xDesc,
    const miopenConvolutionDescriptor_t convDesc,
    const miopenTensorDescriptor_t yDesc, const uint64_t solution_id,
    size_t* workSpaceSize);

MIOPEN_EXPORT miopenStatus_t miopenConvolutionForwardCompileSolution(
    miopenHandle_t handle, const miopenTensorDescriptor_t wDesc,
    const miopenTensorDescriptor_t xDesc,
    const miopenConvolutionDescriptor_t convDesc,
    const miopenTensorDescriptor_t yDesc, const uint64_t solution_id);

MIOPEN_EXPORT miopenStatus_t miopenConvolutionForwardImmediate(
    miopenHandle_t handle, const miopenTensorDescriptor_t wDesc, const void* w,
    const miopenTensorDescriptor_t xDesc, const void* x,
    const miopenConvolutionDescriptor_t convDesc,
    const miopenTensorDescriptor_t yDesc, void* y, void* workSpace,
    size_t workSpaceSize, const uint64_t solution_id);

MIOPEN_EXPORT miopenStatus_t miopenConvolutionBackwardDataGetSolutionCount(
    miopenHandle_t handle, const miopenTensorDescriptor_t dyDesc,
    const miopenTensorDescriptor_t wDesc,
    const miopenConvolutionDescriptor_t convDesc,
    const miopenTensorDescriptor_t dxDesc, size_t* solutionCount);

MIOPEN_EXPORT miopenStatus_t miopenConvolutionBackwardDataGetSolution(
    miopenHandle_t handle, const miopenTensorDescriptor_t dyDesc,
    const miopenTensorDescriptor_t wDesc,
    const miopenConvolutionDescriptor_t convDesc,
    const miopenTensorDescriptor_t dxDesc, const size_t maxSolutionCount,
    size_t* solutionCount, miopenConvSolution_t* solutions);

MIOPEN_EXPORT miopenStatus_t
miopenConvolutionBackwardDataGetSolutionWorkspaceSize(
    miopenHandle_t handle, const miopenTensorDescriptor_t dyDesc,
    const miopenTensorDescriptor_t wDesc,
    const miopenConvolutionDescriptor_t convDesc,
    const miopenTensorDescriptor_t dxDesc, const uint64_t solution_id,
    size_t* workSpaceSize);

MIOPEN_EXPORT miopenStatus_t miopenConvolutionBackwardDataCompileSolution(
    miopenHandle_t handle, const miopenTensorDescriptor_t dyDesc,
    const miopenTensorDescriptor_t wDesc,
    const miopenConvolutionDescriptor_t convDesc,
    const miopenTensorDescriptor_t dxDesc, const uint64_t solution_id);

MIOPEN_EXPORT miopenStatus_t miopenConvolutionBackwardDataImmediate(
    miopenHandle_t handle, const miopenTensorDescriptor_t dyDesc,
    const void* dy, const miopenTensorDescriptor_t wDesc, const void* w,
    const miopenConvolutionDescriptor_t convDesc,
    const miopenTensorDescriptor_t dxDesc, void* dx, void* workSpace,
    size_t workSpaceSize, const uint64_t solution_id);

MIOPEN_EXPORT miopenStatus_t miopenConvolutionBackwardWeightsGetSolutionCount(
    miopenHandle_t handle, const miopenTensorDescriptor_t dyDesc,
    const miopenTensorDescriptor_t xDesc,
    const miopenConvolutionDescriptor_t convDesc,
    const miopenTensorDescriptor_t dwDesc, size_t* solutionCount);

MIOPEN_EXPORT miopenStatus_t miopenConvolutionBackwardWeightsGetSolution(
    miopenHandle_t handle, const miopenTensorDescriptor_t dyDesc,
    const miopenTensorDescriptor_t xDesc,
    const miopenConvolutionDescriptor_t convDesc,
    const miopenTensorDescriptor_t dwDesc, const size_t maxSolutionCount,
    size_t* solutionCount, miopenConvSolution_t* solutions);

MIOPEN_EXPORT miopenStatus_t
miopenConvolutionBackwardWeightsGetSolutionWorkspaceSize(
    miopenHandle_t handle, const miopenTensorDescriptor_t dyDesc,
    const miopenTensorDescriptor_t xDesc,
    const miopenConvolutionDescriptor_t convDesc,
    const miopenTensorDescriptor_t dwDesc, const uint64_t solution_id,
    size_t* workSpaceSize);

MIOPEN_EXPORT miopenStatus_t miopenConvolutionBackwardWeightsCompileSolution(
    miopenHandle_t handle, const miopenTensorDescriptor_t dyDesc,
    const miopenTensorDescriptor_t xDesc,
    const miopenConvolutionDescriptor_t convDesc,
    const miopenTensorDescriptor_t dwDesc, const uint64_t solution_id);

MIOPEN_EXPORT miopenStatus_t miopenConvolutionBackwardWeightsImmediate(
    miopenHandle_t handle, const miopenTensorDescriptor_t dyDesc,
    const void* dy, const miopenTensorDescriptor_t xDesc, const void* x,
    const miopenConvolutionDescriptor_t convDesc,
    const miopenTensorDescriptor_t dwDesc, void* dw, void* workSpace,
    size_t workSpaceSize, const uint64_t solution_id);

MIOPEN_EXPORT miopenStatus_t miopenConvolutionForwardGetWorkSpaceSize(
    miopenHandle_t handle, const miopenTensorDescriptor_t wDesc,
    const miopenTensorDescriptor_t xDesc,
    const miopenConvolutionDescriptor_t convDesc,
    const miopenTensorDescriptor_t yDesc, size_t* workSpaceSize);

MIOPEN_EXPORT miopenStatus_t miopenFindConvolutionForwardAlgorithm(
    miopenHandle_t handle, const miopenTensorDescriptor_t xDesc, const void* x,
    const miopenTensorDescriptor_t wDesc, const void* w,
    const miopenConvolutionDescriptor_t convDesc,
    const miopenTensorDescriptor_t yDesc, void* y, const int requestAlgoCount,
    int* returnedAlgoCount, miopenConvAlgoPerf_t* perfResults, void* workSpace,
    size_t workSpaceSize, bool exhaustiveSearch);

MIOPEN_EXPORT miopenStatus_t miopenConvolutionForward(
    miopenHandle_t handle, const void* alpha,
    const miopenTensorDescriptor_t xDesc, const void* x,
    const miopenTensorDescriptor_t wDesc, const void* w,
    const miopenConvolutionDescriptor_t convDesc, miopenConvFwdAlgorithm_t algo,
    const void* beta, const miopenTensorDescriptor_t yDesc, void* y,
    void* workSpace, size_t workSpaceSize);

MIOPEN_EXPORT miopenStatus_t miopenConvolutionForwardBias(
    miopenHandle_t handle, const void* alpha,
    const miopenTensorDescriptor_t bDesc, const void* b, const void* beta,
    const miopenTensorDescriptor_t yDesc, void* y);

MIOPEN_EXPORT miopenStatus_t miopenConvolutionBackwardDataGetWorkSpaceSize(
    miopenHandle_t handle, const miopenTensorDescriptor_t dyDesc,
    const miopenTensorDescriptor_t wDesc,
    const miopenConvolutionDescriptor_t convDesc,
    const miopenTensorDescriptor_t dxDesc, size_t* workSpaceSize);

MIOPEN_EXPORT miopenStatus_t miopenFindConvolutionBackwardDataAlgorithm(
    miopenHandle_t handle, const miopenTensorDescriptor_t dyDesc,
    const void* dy, const miopenTensorDescriptor_t wDesc, const void* w,
    const miopenConvolutionDescriptor_t convDesc,
    const miopenTensorDescriptor_t dxDesc, void* dx, const int requestAlgoCount,
    int* returnedAlgoCount, miopenConvAlgoPerf_t* perfResults, void* workSpace,
    size_t workSpaceSize, bool exhaustiveSearch);

MIOPEN_EXPORT miopenStatus_t miopenConvolutionBackwardData(
    miopenHandle_t handle, const void* alpha,
    const miopenTensorDescriptor_t dyDesc, const void* dy,
    const miopenTensorDescriptor_t wDesc, const void* w,
    const miopenConvolutionDescriptor_t convDesc,
    miopenConvBwdDataAlgorithm_t algo, const void* beta,
    const miopenTensorDescriptor_t dxDesc, void* dx, void* workSpace,
    size_t workSpaceSize);

MIOPEN_EXPORT miopenStatus_t miopenConvolutionBackwardWeightsGetWorkSpaceSize(
    miopenHandle_t handle, const miopenTensorDescriptor_t dyDesc,
    const miopenTensorDescriptor_t xDesc,
    const miopenConvolutionDescriptor_t convDesc,
    const miopenTensorDescriptor_t dwDesc, size_t* workSpaceSize);

MIOPEN_EXPORT miopenStatus_t miopenFindConvolutionBackwardWeightsAlgorithm(
    miopenHandle_t handle, const miopenTensorDescriptor_t dyDesc,
    const void* dy, const miopenTensorDescriptor_t xDesc, const void* x,
    const miopenConvolutionDescriptor_t convDesc,
    const miopenTensorDescriptor_t dwDesc, void* dw, const int requestAlgoCount,
    int* returnedAlgoCount, miopenConvAlgoPerf_t* perfResults, void* workSpace,
    size_t workSpaceSize, bool exhaustiveSearch);

MIOPEN_EXPORT miopenStatus_t miopenConvolutionBackwardWeights(
    miopenHandle_t handle, const void* alpha,
    const miopenTensorDescriptor_t dyDesc, const void* dy,
    const miopenTensorDescriptor_t xDesc, const void* x,
    const miopenConvolutionDescriptor_t convDesc,
    miopenConvBwdWeightsAlgorithm_t algo, const void* beta,
    const miopenTensorDescriptor_t dwDesc, void* dw, void* workSpace,
    size_t workSpaceSize);

MIOPEN_EXPORT miopenStatus_t miopenConvolutionBackwardBias(
    miopenHandle_t handle, const void* alpha,
    const miopenTensorDescriptor_t dyDesc, const void* dy, const void* beta,
    const miopenTensorDescriptor_t dbDesc, void* db);

MIOPEN_EXPORT miopenStatus_t
miopenCreatePoolingDescriptor(miopenPoolingDescriptor_t* poolDesc);

MIOPEN_EXPORT miopenStatus_t miopenSetPoolingIndexType(
    miopenPoolingDescriptor_t poolDesc, miopenIndexType_t index_type);

MIOPEN_EXPORT miopenStatus_t miopenGetPoolingIndexType(
    miopenPoolingDescriptor_t poolDesc, miopenIndexType_t* index_type);

MIOPEN_EXPORT miopenStatus_t miopenSetPoolingWorkSpaceIndexMode(
    miopenPoolingDescriptor_t poolDesc,
    miopenPoolingWorkspaceIndexMode_t workspace_index);

MIOPEN_EXPORT miopenStatus_t miopenGetPoolingWorkSpaceIndexMode(
    miopenPoolingDescriptor_t poolDesc,
    miopenPoolingWorkspaceIndexMode_t* workspace_index);

MIOPEN_EXPORT miopenStatus_t miopenSet2dPoolingDescriptor(
    miopenPoolingDescriptor_t poolDesc, miopenPoolingMode_t mode,
    int windowHeight, int windowWidth, int pad_h, int pad_w, int stride_h,
    int stride_w);

MIOPEN_EXPORT miopenStatus_t miopenGet2dPoolingDescriptor(
    const miopenPoolingDescriptor_t poolDesc, miopenPoolingMode_t* mode,
    int* windowHeight, int* windowWidth, int* pad_h, int* pad_w, int* stride_h,
    int* stride_w);

MIOPEN_EXPORT miopenStatus_t miopenGetPoolingForwardOutputDim(
    const miopenPoolingDescriptor_t poolDesc,
    const miopenTensorDescriptor_t tensorDesc, int* n, int* c, int* h, int* w);

MIOPEN_EXPORT miopenStatus_t miopenSetNdPoolingDescriptor(
    miopenPoolingDescriptor_t poolDesc, const miopenPoolingMode_t mode,
    int nbDims, int* windowDimA, int* padA, int* stridesA);

MIOPEN_EXPORT miopenStatus_t miopenGetNdPoolingDescriptor(
    const miopenPoolingDescriptor_t poolDesc, int nbDimsRequested,
    miopenPoolingMode_t* mode, int* nbDims, int* windowDimA, int* padA,
    int* stridesA);

MIOPEN_EXPORT miopenStatus_t miopenGetPoolingNdForwardOutputDim(
    const miopenPoolingDescriptor_t poolDesc,
    const miopenTensorDescriptor_t tensorDesc, int dims, int* tensorDimArr);

MIOPEN_EXPORT miopenStatus_t miopenPoolingGetWorkSpaceSize(
    const miopenTensorDescriptor_t yDesc, size_t* workSpaceSize);

MIOPEN_EXPORT miopenStatus_t miopenPoolingGetWorkSpaceSizeV2(
    const miopenPoolingDescriptor_t poolDesc,
    const miopenTensorDescriptor_t yDesc, size_t* workSpaceSize);

MIOPEN_EXPORT miopenStatus_t miopenPoolingForward(
    miopenHandle_t handle, const miopenPoolingDescriptor_t poolDesc,
    const void* alpha, const miopenTensorDescriptor_t xDesc, const void* x,
    const void* beta, const miopenTensorDescriptor_t yDesc, void* y,
    bool do_backward, void* workSpace, size_t workSpaceSize);

MIOPEN_EXPORT miopenStatus_t miopenPoolingBackward(
    miopenHandle_t handle, const miopenPoolingDescriptor_t poolDesc,
    const void* alpha, const miopenTensorDescriptor_t yDesc, const void* y,
    const miopenTensorDescriptor_t dyDesc, const void* dy,
    const miopenTensorDescriptor_t xDesc, const void* x, const void* beta,
    const miopenTensorDescriptor_t dxDesc, void* dx, const void* workSpace);

MIOPEN_EXPORT miopenStatus_t
miopenDestroyPoolingDescriptor(miopenPoolingDescriptor_t poolDesc);

MIOPEN_EXPORT miopenStatus_t
miopenCreateLRNDescriptor(miopenLRNDescriptor_t* lrnDesc);

MIOPEN_EXPORT miopenStatus_t miopenSetLRNDescriptor(
    const miopenLRNDescriptor_t lrnDesc, miopenLRNMode_t mode,
    unsigned int lrnN, double lrnAlpha, double lrnBeta, double lrnK);

MIOPEN_EXPORT miopenStatus_t miopenGetLRNDescriptor(
    const miopenLRNDescriptor_t lrnDesc, miopenLRNMode_t* mode,
    unsigned int* lrnN, double* lrnAlpha, double* lrnBeta, double* lrnK);

MIOPEN_EXPORT miopenStatus_t miopenLRNGetWorkSpaceSize(
    const miopenTensorDescriptor_t yDesc, size_t* workSpaceSize);

MIOPEN_EXPORT miopenStatus_t miopenLRNForward(
    miopenHandle_t handle, const miopenLRNDescriptor_t lrnDesc,
    const void* alpha, const miopenTensorDescriptor_t xDesc, const void* x,
    const void* beta, const miopenTensorDescriptor_t yDesc, void* y,
    bool do_backward, void* workSpace);

MIOPEN_EXPORT miopenStatus_t miopenLRNBackward(
    miopenHandle_t handle, const miopenLRNDescriptor_t lrnDesc,
    const void* alpha, const miopenTensorDescriptor_t yDesc, const void* y,
    const miopenTensorDescriptor_t dyDesc, const void* dy,
    const miopenTensorDescriptor_t xDesc, const void* x, const void* beta,
    const miopenTensorDescriptor_t dxDesc, void* dx, const void* workSpace);

MIOPEN_EXPORT miopenStatus_t
miopenDestroyLRNDescriptor(miopenLRNDescriptor_t lrnDesc);

MIOPEN_EXPORT miopenStatus_t miopenDeriveBNTensorDescriptor(
    miopenTensorDescriptor_t derivedBnDesc,
    const miopenTensorDescriptor_t xDesc, miopenBatchNormMode_t bn_mode);

MIOPEN_EXPORT miopenStatus_t miopenBatchNormalizationForwardTraining(
    miopenHandle_t handle, miopenBatchNormMode_t bn_mode, void* alpha,
    void* beta, const miopenTensorDescriptor_t xDesc, const void* x,
    const miopenTensorDescriptor_t yDesc, void* y,
    const miopenTensorDescriptor_t bnScaleBiasMeanVarDesc, void* bnScale,
    void* bnBias, double expAvgFactor, void* resultRunningMean,
    void* resultRunningVariance, double epsilon, void* resultSaveMean,
    void* resultSaveInvVariance);

MIOPEN_EXPORT miopenStatus_t miopenBatchNormalizationForwardInference(
    miopenHandle_t handle, miopenBatchNormMode_t bn_mode, void* alpha,
    void* beta, const miopenTensorDescriptor_t xDesc, const void* x,
    const miopenTensorDescriptor_t yDesc, void* y,
    const miopenTensorDescriptor_t bnScaleBiasMeanVarDesc, void* bnScale,
    void* bnBias, void* estimatedMean, void* estimatedVariance, double epsilon);

MIOPEN_EXPORT miopenStatus_t miopenBatchNormalizationBackward(
    miopenHandle_t handle, miopenBatchNormMode_t bn_mode,
    const void* alphaDataDiff, const void* betaDataDiff,
    const void* alphaParamDiff, const void* betaParamDiff,
    const miopenTensorDescriptor_t xDesc, const void* x,
    const miopenTensorDescriptor_t dyDesc, const void* dy,
    const miopenTensorDescriptor_t dxDesc, void* dx,
    const miopenTensorDescriptor_t bnScaleBiasDiffDesc, const void* bnScale,
    void* resultBnScaleDiff, void* resultBnBiasDiff, double epsilon,
    const void* savedMean, const void* savedInvVariance);

MIOPEN_EXPORT miopenStatus_t
miopenCreateActivationDescriptor(miopenActivationDescriptor_t* activDesc);

MIOPEN_EXPORT miopenStatus_t miopenSetActivationDescriptor(
    const miopenActivationDescriptor_t activDesc, miopenActivationMode_t mode,
    double activAlpha, double activBeta, double activGamma);

MIOPEN_EXPORT miopenStatus_t miopenGetActivationDescriptor(
    const miopenActivationDescriptor_t activDesc, miopenActivationMode_t* mode,
    double* activAlpha, double* activBeta, double* activGamma);

MIOPEN_EXPORT miopenStatus_t miopenActivationForward(
    miopenHandle_t handle, const miopenActivationDescriptor_t activDesc,
    const void* alpha, const miopenTensorDescriptor_t xDesc, const void* x,
    const void* beta, const miopenTensorDescriptor_t yDesc, void* y);

MIOPEN_EXPORT miopenStatus_t miopenActivationBackward(
    miopenHandle_t handle, const miopenActivationDescriptor_t activDesc,
    const void* alpha, const miopenTensorDescriptor_t yDesc, const void* y,
    const miopenTensorDescriptor_t dyDesc, const void* dy,
    const miopenTensorDescriptor_t xDesc, const void* x, const void* beta,
    const miopenTensorDescriptor_t dxDesc, void* dx);

MIOPEN_EXPORT miopenStatus_t
miopenDestroyActivationDescriptor(miopenActivationDescriptor_t activDesc);

MIOPEN_EXPORT miopenStatus_t miopenSoftmaxForward(
    miopenHandle_t handle, const void* alpha,
    const miopenTensorDescriptor_t xDesc, const void* x, const void* beta,
    const miopenTensorDescriptor_t yDesc, void* y);

MIOPEN_EXPORT miopenStatus_t miopenSoftmaxBackward(
    miopenHandle_t handle, const void* alpha,
    const miopenTensorDescriptor_t yDesc, const void* y,
    const miopenTensorDescriptor_t dyDesc, const void* dy, const void* beta,
    const miopenTensorDescriptor_t dxDesc, void* dx);

MIOPEN_EXPORT miopenStatus_t miopenSoftmaxForward_V2(
    miopenHandle_t handle, const void* alpha,
    const miopenTensorDescriptor_t xDesc, const void* x, const void* beta,
    const miopenTensorDescriptor_t yDesc, void* y,
    miopenSoftmaxAlgorithm_t algorithm, miopenSoftmaxMode_t mode);

MIOPEN_EXPORT miopenStatus_t miopenSoftmaxBackward_V2(
    miopenHandle_t handle, const void* alpha,
    const miopenTensorDescriptor_t yDesc, const void* y,
    const miopenTensorDescriptor_t dyDesc, const void* dy, const void* beta,
    const miopenTensorDescriptor_t dxDesc, void* dx,
    miopenSoftmaxAlgorithm_t algorithm, miopenSoftmaxMode_t mode);

typedef enum {
  miopenVerticalFusion = 0,
  miopenHorizontalFusion = 1,
} miopenFusionDirection_t;

MIOPEN_EXPORT miopenStatus_t
miopenCreateFusionPlan(miopenFusionPlanDescriptor_t* fusePlanDesc,
                       const miopenFusionDirection_t fuseDirection,
                       const miopenTensorDescriptor_t inputDesc);

MIOPEN_EXPORT miopenStatus_t
miopenDestroyFusionPlan(miopenFusionPlanDescriptor_t fusePlanDesc);

MIOPEN_EXPORT miopenStatus_t miopenCompileFusionPlan(
    miopenHandle_t handle, miopenFusionPlanDescriptor_t fusePlanDesc);

MIOPEN_EXPORT miopenStatus_t
miopenFusionPlanGetOp(miopenFusionPlanDescriptor_t fusePlanDesc,
                      const int op_idx, miopenFusionOpDescriptor_t* op);

MIOPEN_EXPORT miopenStatus_t miopenFusionPlanGetWorkSpaceSize(
    miopenHandle_t handle, miopenFusionPlanDescriptor_t fusePlanDesc,
    size_t* workSpaceSize, miopenConvFwdAlgorithm_t algo);

MIOPEN_EXPORT miopenStatus_t miopenFusionPlanConvolutionGetAlgo(
    miopenFusionPlanDescriptor_t fusePlanDesc, const int requestAlgoCount,
    int* returnedAlgoCount, miopenConvFwdAlgorithm_t* returnedAlgos);

MIOPEN_EXPORT miopenStatus_t miopenFusionPlanConvolutionSetAlgo(
    miopenFusionPlanDescriptor_t fusePlanDesc, miopenConvFwdAlgorithm_t algo);

MIOPEN_EXPORT miopenStatus_t miopenCreateOpConvForward(
    miopenFusionPlanDescriptor_t fusePlanDesc,
    miopenFusionOpDescriptor_t* convOp, miopenConvolutionDescriptor_t convDesc,
    const miopenTensorDescriptor_t wDesc);

MIOPEN_EXPORT miopenStatus_t miopenCreateOpActivationForward(
    miopenFusionPlanDescriptor_t fusePlanDesc,
    miopenFusionOpDescriptor_t* activFwdOp, miopenActivationMode_t mode);

MIOPEN_EXPORT miopenStatus_t miopenCreateOpActivationBackward(
    miopenFusionPlanDescriptor_t fusePlanDesc,
    miopenFusionOpDescriptor_t* activBwdOp, miopenActivationMode_t mode);

MIOPEN_EXPORT miopenStatus_t miopenCreateOpBiasForward(
    miopenFusionPlanDescriptor_t fusePlanDesc,
    miopenFusionOpDescriptor_t* biasOp, const miopenTensorDescriptor_t bDesc);

MIOPEN_EXPORT miopenStatus_t miopenCreateOpBatchNormInference(
    miopenFusionPlanDescriptor_t fusePlanDesc, miopenFusionOpDescriptor_t* bnOp,
    const miopenBatchNormMode_t bn_mode,
    const miopenTensorDescriptor_t bnScaleBiasMeanVarDesc);

MIOPEN_EXPORT miopenStatus_t miopenCreateOpBatchNormForward(
    miopenFusionPlanDescriptor_t fusePlanDesc,
    miopenFusionOpDescriptor_t* bnFwdOp, const miopenBatchNormMode_t bn_mode,
    bool runningMeanVariance);

MIOPEN_EXPORT miopenStatus_t miopenCreateOpBatchNormBackward(
    miopenFusionPlanDescriptor_t fusePlanDesc,
    miopenFusionOpDescriptor_t* bnBwdOp, const miopenBatchNormMode_t bn_mode);

MIOPEN_EXPORT miopenStatus_t
miopenCreateOperatorArgs(miopenOperatorArgs_t* args);

MIOPEN_EXPORT miopenStatus_t
miopenDestroyOperatorArgs(miopenOperatorArgs_t args);

MIOPEN_EXPORT miopenStatus_t miopenSetOpArgsConvForward(
    miopenOperatorArgs_t args, const miopenFusionOpDescriptor_t convOp,
    const void* alpha, const void* beta, const void* w);

MIOPEN_EXPORT miopenStatus_t miopenSetOpArgsActivForward(
    miopenOperatorArgs_t args, const miopenFusionOpDescriptor_t activFwdOp,
    const void* alpha, const void* beta, double activAlpha, double activBeta,
    double activGamma);

MIOPEN_EXPORT miopenStatus_t miopenSetOpArgsActivBackward(
    miopenOperatorArgs_t args, const miopenFusionOpDescriptor_t activBwdOp,
    const void* alpha, const void* beta, const void* y, const void* reserved,
    double activAlpha, double activBeta, double activGamma);

MIOPEN_EXPORT miopenStatus_t miopenSetOpArgsBatchNormInference(
    miopenOperatorArgs_t args, const miopenFusionOpDescriptor_t bnOp,
    const void* alpha, const void* beta, const void* bnScale,
    const void* bnBias, const void* estimatedMean,
    const void* estimatedVariance, double epsilon);

MIOPEN_EXPORT miopenStatus_t miopenSetOpArgsBatchNormForward(
    miopenOperatorArgs_t args, const miopenFusionOpDescriptor_t bnOp,
    const void* alpha, const void* beta, const void* bnScale,
    const void* bnBias, void* savedMean, void* savedInvVariance,
    void* runningMean, void* runningVariance, double expAvgFactor,
    double epsilon);

MIOPEN_EXPORT miopenStatus_t miopenSetOpArgsBatchNormBackward(
    miopenOperatorArgs_t args, const miopenFusionOpDescriptor_t bnOp,
    const void* alpha, const void* beta, const void* x, const void* bnScale,
    const void* bnBias, void* resultBnScaleDiff, void* resultBnBiasDiff,
    const void* savedMean, const void* savedInvVariance);

MIOPEN_EXPORT miopenStatus_t miopenSetOpArgsBiasForward(
    miopenOperatorArgs_t args, const miopenFusionOpDescriptor_t biasOp,
    const void* alpha, const void* beta, const void* bias);

MIOPEN_EXPORT miopenStatus_t miopenExecuteFusionPlan(
    const miopenHandle_t handle,
    const miopenFusionPlanDescriptor_t fusePlanDesc,
    const miopenTensorDescriptor_t inputDesc, const void* input,
    const miopenTensorDescriptor_t outputDesc, void* output,
    miopenOperatorArgs_t args);

typedef enum {
  miopenRNNRELU = 0,
  miopenRNNTANH = 1,
  miopenLSTM = 2,
  miopenGRU = 3,
} miopenRNNMode_t;

typedef enum {
  miopenRNNlinear = 0,
  miopenRNNskip = 1,
} miopenRNNInputMode_t;

typedef enum {
  miopenRNNdefault = 0,
  miopenRNNfundamental = 1,
} miopenRNNAlgo_t;

typedef enum {
  miopenRNNunidirection = 0,
  miopenRNNbidirection = 1,
} miopenRNNDirectionMode_t;

typedef enum {
  miopenRNNNoBias = 0,
  miopenRNNwithBias = 1,
} miopenRNNBiasMode_t;

typedef enum {
  miopenRNNAlgoGEMM = 0,
} miopenRNNGEMMalgoMode_t;

MIOPEN_EXPORT miopenStatus_t
miopenCreateRNNDescriptor(miopenRNNDescriptor_t* rnnDesc);

MIOPEN_EXPORT miopenStatus_t miopenGetRNNDescriptor(
    miopenRNNDescriptor_t rnnDesc, miopenRNNMode_t* rnnMode,
    miopenRNNAlgo_t* algoMode, miopenRNNInputMode_t* inputMode,
    miopenRNNDirectionMode_t* dirMode, miopenRNNBiasMode_t* biasMode,
    int* hiddenSize, int* layer);

MIOPEN_EXPORT miopenStatus_t miopenGetRNNDescriptor_V2(
    miopenRNNDescriptor_t rnnDesc, int* hiddenSize, int* layer,
    miopenDropoutDescriptor_t* dropoutDesc, miopenRNNInputMode_t* inputMode,
    miopenRNNDirectionMode_t* dirMode, miopenRNNMode_t* rnnMode,
    miopenRNNBiasMode_t* biasMode, miopenRNNAlgo_t* algoMode,
    miopenDataType_t* dataType);

MIOPEN_EXPORT miopenStatus_t
miopenDestroyRNNDescriptor(miopenRNNDescriptor_t rnnDesc);

MIOPEN_EXPORT miopenStatus_t miopenSetRNNDescriptor(
    miopenRNNDescriptor_t rnnDesc, const int hsize, const int nlayers,
    miopenRNNInputMode_t inMode, miopenRNNDirectionMode_t direction,
    miopenRNNMode_t rnnMode, miopenRNNBiasMode_t biasMode, miopenRNNAlgo_t algo,
    miopenDataType_t dataType);

MIOPEN_EXPORT miopenStatus_t miopenSetRNNDescriptor_V2(
    miopenRNNDescriptor_t rnnDesc, const int hsize, const int nlayers,
    miopenDropoutDescriptor_t dropoutDesc, miopenRNNInputMode_t inMode,
    miopenRNNDirectionMode_t direction, miopenRNNMode_t rnnMode,
    miopenRNNBiasMode_t biasMode, miopenRNNAlgo_t algo,
    miopenDataType_t dataType);

MIOPEN_EXPORT miopenStatus_t miopenGetRNNWorkspaceSize(
    miopenHandle_t handle, const miopenRNNDescriptor_t rnnDesc,
    const int sequenceLen, const miopenTensorDescriptor_t* xDesc,
    size_t* numBytes);

MIOPEN_EXPORT miopenStatus_t miopenGetRNNTrainingReserveSize(
    miopenHandle_t handle, miopenRNNDescriptor_t rnnDesc, const int sequenceLen,
    const miopenTensorDescriptor_t* xDesc, size_t* numBytes);

MIOPEN_EXPORT miopenStatus_t miopenGetRNNParamsSize(
    miopenHandle_t handle, miopenRNNDescriptor_t rnnDesc,
    miopenTensorDescriptor_t xDesc, size_t* numBytes, miopenDataType_t dtype);

MIOPEN_EXPORT miopenStatus_t miopenGetRNNParamsDescriptor(
    miopenHandle_t handle, miopenRNNDescriptor_t rnnDesc,
    miopenTensorDescriptor_t xDesc, miopenTensorDescriptor_t wDesc,
    miopenDataType_t dtype);

MIOPEN_EXPORT miopenStatus_t miopenGetRNNInputTensorSize(
    miopenHandle_t handle, miopenRNNDescriptor_t rnnDesc, const int seqLen,
    miopenTensorDescriptor_t* xDesc, size_t* numBytes);

MIOPEN_EXPORT miopenStatus_t miopenGetRNNHiddenTensorSize(
    miopenHandle_t handle, miopenRNNDescriptor_t rnnDesc, const int seqLen,
    miopenTensorDescriptor_t* xDesc, size_t* numBytes);

MIOPEN_EXPORT miopenStatus_t miopenGetRNNLayerParamSize(
    miopenHandle_t handle, miopenRNNDescriptor_t rnnDesc, const int layer,
    miopenTensorDescriptor_t xDesc, const int paramID, size_t* numBytes);

MIOPEN_EXPORT miopenStatus_t
miopenGetRNNLayerBiasSize(miopenHandle_t handle, miopenRNNDescriptor_t rnnDesc,
                          const int layer, const int biasID, size_t* numBytes);

MIOPEN_EXPORT miopenStatus_t miopenGetRNNLayerParam(
    miopenHandle_t handle, miopenRNNDescriptor_t rnnDesc, const int layer,
    miopenTensorDescriptor_t xDesc, miopenTensorDescriptor_t wDesc,
    const void* w, const int paramID, miopenTensorDescriptor_t paramDesc,
    void* layerParam);

MIOPEN_EXPORT miopenStatus_t miopenGetRNNLayerBias(
    miopenHandle_t handle, miopenRNNDescriptor_t rnnDesc, const int layer,
    miopenTensorDescriptor_t xDesc, miopenTensorDescriptor_t wDesc,
    const void* w, const int biasID, miopenTensorDescriptor_t biasDesc,
    void* layerBias);

MIOPEN_EXPORT miopenStatus_t miopenGetRNNLayerParamOffset(
    miopenRNNDescriptor_t rnnDesc, const int layer,
    miopenTensorDescriptor_t xDesc, const int paramID,
    miopenTensorDescriptor_t paramDesc, size_t* layerParamOffset);

MIOPEN_EXPORT miopenStatus_t miopenGetRNNLayerBiasOffset(
    miopenRNNDescriptor_t rnnDesc, const int layer,
    miopenTensorDescriptor_t xDesc, const int biasID,
    miopenTensorDescriptor_t biasDesc, size_t* layerBiasOffset);

MIOPEN_EXPORT miopenStatus_t miopenSetRNNLayerParam(
    miopenHandle_t handle, miopenRNNDescriptor_t rnnDesc, const int layer,
    miopenTensorDescriptor_t xDesc, miopenTensorDescriptor_t wDesc, void* w,
    const int paramID, miopenTensorDescriptor_t paramDesc,
    const void* layerParam);

MIOPEN_EXPORT miopenStatus_t miopenSetRNNLayerBias(
    miopenHandle_t handle, miopenRNNDescriptor_t rnnDesc, const int layer,
    miopenTensorDescriptor_t xDesc, miopenTensorDescriptor_t wDesc, void* w,
    const int biasID, miopenTensorDescriptor_t biasDesc, const void* layerBias);

MIOPEN_EXPORT miopenStatus_t miopenRNNForwardTraining(
    miopenHandle_t handle, const miopenRNNDescriptor_t rnnDesc,
    const int sequenceLen, const miopenTensorDescriptor_t* xDesc, const void* x,
    const miopenTensorDescriptor_t hxDesc, const void* hx,
    const miopenTensorDescriptor_t cxDesc, const void* cx,
    const miopenTensorDescriptor_t wDesc, const void* w,
    const miopenTensorDescriptor_t* yDesc, void* y,
    const miopenTensorDescriptor_t hyDesc, void* hy,
    const miopenTensorDescriptor_t cyDesc, void* cy, void* workSpace,
    size_t workSpaceNumBytes, void* reserveSpace, size_t reserveSpaceNumBytes);

MIOPEN_EXPORT miopenStatus_t miopenRNNBackwardData(
    miopenHandle_t handle, const miopenRNNDescriptor_t rnnDesc,
    const int sequenceLen, const miopenTensorDescriptor_t* yDesc, const void* y,
    const miopenTensorDescriptor_t* dyDesc, const void* dy,
    const miopenTensorDescriptor_t dhyDesc, const void* dhy,
    const miopenTensorDescriptor_t dcyDesc, const void* dcy,
    const miopenTensorDescriptor_t wDesc, const void* w,
    const miopenTensorDescriptor_t hxDesc, const void* hx,
    const miopenTensorDescriptor_t cxDesc, const void* cx,
    const miopenTensorDescriptor_t* dxDesc, void* dx,
    const miopenTensorDescriptor_t dhxDesc, void* dhx,
    const miopenTensorDescriptor_t dcxDesc, void* dcx, void* workSpace,
    size_t workSpaceNumBytes, void* reserveSpace, size_t reserveSpaceNumBytes);

MIOPEN_EXPORT miopenStatus_t miopenRNNBackwardWeights(
    miopenHandle_t handle, const miopenRNNDescriptor_t rnnDesc,
    const int sequenceLen, const miopenTensorDescriptor_t* xDesc, const void* x,
    const miopenTensorDescriptor_t hxDesc, const void* hx,
    const miopenTensorDescriptor_t* yDesc, const void* y,
    const miopenTensorDescriptor_t dwDesc, void* dw, void* workSpace,
    size_t workSpaceNumBytes, const void* reserveSpace,
    size_t reserveSpaceNumBytes);

MIOPEN_EXPORT miopenStatus_t miopenRNNForwardInference(
    miopenHandle_t handle, miopenRNNDescriptor_t rnnDesc, const int sequenceLen,
    const miopenTensorDescriptor_t* xDesc, const void* x,
    const miopenTensorDescriptor_t hxDesc, const void* hx,
    const miopenTensorDescriptor_t cxDesc, const void* cx,
    const miopenTensorDescriptor_t wDesc, const void* w,
    const miopenTensorDescriptor_t* yDesc, void* y,
    const miopenTensorDescriptor_t hyDesc, void* hy,
    const miopenTensorDescriptor_t cyDesc, void* cy, void* workSpace,
    size_t workSpaceNumBytes);

typedef enum {
  MIOPEN_CTC_LOSS_ALGO_DETERMINISTIC = 0,
} miopenCTCLossAlgo_t;

MIOPEN_EXPORT miopenStatus_t
miopenCreateCTCLossDescriptor(miopenCTCLossDescriptor_t* ctcLossDesc);

MIOPEN_EXPORT miopenStatus_t miopenGetCTCLossDescriptor(
    miopenCTCLossDescriptor_t ctcLossDesc, miopenDataType_t* dataType,
    int* blank_label_id, bool* apply_softmax_layer);

MIOPEN_EXPORT miopenStatus_t
miopenDestroyCTCLossDescriptor(miopenCTCLossDescriptor_t ctcLossDesc);

MIOPEN_EXPORT miopenStatus_t miopenSetCTCLossDescriptor(
    miopenCTCLossDescriptor_t ctcLossDesc, miopenDataType_t dataType,
    const int blank_label_id, bool apply_softmax_layer);

MIOPEN_EXPORT miopenStatus_t miopenGetCTCLossWorkspaceSize(
    miopenHandle_t handle, const miopenTensorDescriptor_t probsDesc,
    const miopenTensorDescriptor_t gradientsDesc, const int* labels,
    const int* labelLengths, const int* inputLengths, miopenCTCLossAlgo_t algo,
    const miopenCTCLossDescriptor_t ctcLossDesc, size_t* workSpaceSize);

MIOPEN_EXPORT miopenStatus_t miopenCTCLoss(
    miopenHandle_t handle, const miopenTensorDescriptor_t probsDesc,
    const void* probs, const int* labels, const int* labelLengths,
    const int* inputLengths, void* losses,
    const miopenTensorDescriptor_t gradientsDesc, void* gradients,
    miopenCTCLossAlgo_t algo, const miopenCTCLossDescriptor_t ctcLossDesc,
    void* workSpace, size_t workSpaceSize);

typedef enum {
  MIOPEN_RNG_PSEUDO_XORWOW = 0,
} miopenRNGType_t;

MIOPEN_EXPORT miopenStatus_t
miopenCreateDropoutDescriptor(miopenDropoutDescriptor_t* dropoutDesc);

MIOPEN_EXPORT miopenStatus_t
miopenDestroyDropoutDescriptor(miopenDropoutDescriptor_t dropoutDesc);

MIOPEN_EXPORT miopenStatus_t miopenDropoutGetReserveSpaceSize(
    const miopenTensorDescriptor_t xDesc, size_t* reserveSpaceSizeInBytes);

MIOPEN_EXPORT miopenStatus_t
miopenDropoutGetStatesSize(miopenHandle_t handle, size_t* stateSizeInBytes);

MIOPEN_EXPORT miopenStatus_t miopenGetDropoutDescriptor(
    miopenDropoutDescriptor_t dropoutDesc, miopenHandle_t handle,
    float* dropout, void** states, unsigned long long* seed, bool* use_mask,
    bool* state_evo, miopenRNGType_t* rng_mode);

MIOPEN_EXPORT miopenStatus_t miopenRestoreDropoutDescriptor(
    miopenDropoutDescriptor_t dropoutDesc, miopenHandle_t handle, float dropout,
    void* states, size_t stateSizeInBytes, unsigned long long seed,
    bool use_mask, bool state_evo, miopenRNGType_t rng_mode);

MIOPEN_EXPORT miopenStatus_t miopenSetDropoutDescriptor(
    miopenDropoutDescriptor_t dropoutDesc, miopenHandle_t handle, float dropout,
    void* states, size_t stateSizeInBytes, unsigned long long seed,
    bool use_mask, bool state_evo, miopenRNGType_t rng_mode);

MIOPEN_EXPORT miopenStatus_t miopenDropoutForward(
    miopenHandle_t handle, const miopenDropoutDescriptor_t dropoutDesc,
    const miopenTensorDescriptor_t noise_shape,
    const miopenTensorDescriptor_t xDesc, const void* x,
    const miopenTensorDescriptor_t yDesc, void* y, void* reserveSpace,
    size_t reserveSpaceSizeInBytes);

MIOPEN_EXPORT miopenStatus_t miopenDropoutBackward(
    miopenHandle_t handle, const miopenDropoutDescriptor_t dropoutDesc,
    const miopenTensorDescriptor_t noise_shape,
    const miopenTensorDescriptor_t dyDesc, const void* dy,
    const miopenTensorDescriptor_t dxDesc, void* dx, void* reserveSpace,
    size_t reserveSpaceSizeInBytes);
