typedef enum hipfftResult_t {
  HIPFFT_SUCCESS = 0,
  HIPFFT_INVALID_PLAN = 1,
  HIPFFT_ALLOC_FAILED = 2,
  HIPFFT_INVALID_TYPE = 3,
  HIPFFT_INVALID_VALUE = 4,
  HIPFFT_INTERNAL_ERROR = 5,
  HIPFFT_EXEC_FAILED = 6,
  HIPFFT_SETUP_FAILED = 7,
  HIPFFT_INVALID_SIZE = 8,
  HIPFFT_UNALIGNED_DATA = 9,
  HIPFFT_INCOMPLETE_PARAMETER_LIST = 10,
  HIPFFT_INVALID_DEVICE = 11,
  HIPFFT_PARSE_ERROR = 12,
  HIPFFT_NO_WORKSPACE = 13,
  HIPFFT_NOT_IMPLEMENTED = 14,
  HIPFFT_NOT_SUPPORTED = 16,
} hipfftResult;

typedef enum hipfftType_t {
  HIPFFT_R2C = 0x2a,
  HIPFFT_C2R = 0x2c,
  HIPFFT_C2C = 0x29,
  HIPFFT_D2Z = 0x6a,
  HIPFFT_Z2D = 0x6c,
  HIPFFT_Z2Z = 0x69,
} hipfftType;

typedef enum hipfftLibraryPropertyType_t {
  HIPFFT_MAJOR_VERSION,
  HIPFFT_MINOR_VERSION,
  HIPFFT_PATCH_LEVEL,
} hipfftLibraryPropertyType;

HIPFFT_EXPORT hipfftResult hipfftCreate(hipfftHandle* plan);

HIPFFT_EXPORT hipfftResult hipfftMakePlanMany64(
    hipfftHandle plan, int rank, long long int* n, long long int* inembed,
    long long int istride, long long int idist, long long int* onembed,
    long long int ostride, long long int odist, hipfftType type,
    long long int batch, size_t* workSize);

HIPFFT_EXPORT hipfftResult hipfftGetSize(hipfftHandle plan, size_t* workSize);

HIPFFT_EXPORT hipfftResult hipfftSetAutoAllocation(hipfftHandle plan,
                                                   int autoAllocate);

HIPFFT_EXPORT hipfftResult hipfftSetWorkArea(hipfftHandle plan, void* workArea);

HIPFFT_EXPORT hipfftResult hipfftExecC2C(hipfftHandle plan,
                                         hipfftComplex* idata,
                                         hipfftComplex* odata, int direction);

HIPFFT_EXPORT hipfftResult hipfftExecR2C(hipfftHandle plan, hipfftReal* idata,
                                         hipfftComplex* odata);

HIPFFT_EXPORT hipfftResult hipfftExecC2R(hipfftHandle plan,
                                         hipfftComplex* idata,
                                         hipfftReal* odata);

HIPFFT_EXPORT hipfftResult hipfftExecZ2Z(hipfftHandle plan,
                                         hipfftDoubleComplex* idata,
                                         hipfftDoubleComplex* odata,
                                         int direction);

HIPFFT_EXPORT hipfftResult hipfftExecD2Z(hipfftHandle plan,
                                         hipfftDoubleReal* idata,
                                         hipfftDoubleComplex* odata);

HIPFFT_EXPORT hipfftResult hipfftExecZ2D(hipfftHandle plan,
                                         hipfftDoubleComplex* idata,
                                         hipfftDoubleReal* odata);

HIPFFT_EXPORT hipfftResult hipfftSetStream(hipfftHandle plan,
                                           hipStream_t stream);

HIPFFT_EXPORT hipfftResult hipfftDestroy(hipfftHandle plan);

HIPFFT_EXPORT hipfftResult hipfftGetProperty(hipfftLibraryPropertyType type,
                                             int* value);
