HIPFFT_EXPORT hipfftResult hipfftCreate(hipfftHandle* plan) {
  return DynamicCall<decltype(hipfftCreate), &hipfftCreate>("hipfftCreate",
                                                            plan);
}

HIPFFT_EXPORT hipfftResult hipfftMakePlanMany64(
    hipfftHandle plan, int rank, long long int* n, long long int* inembed,
    long long int istride, long long int idist, long long int* onembed,
    long long int ostride, long long int odist, hipfftType type,
    long long int batch, size_t* workSize) {
  return DynamicCall<decltype(hipfftMakePlanMany64), &hipfftMakePlanMany64>(
      "hipfftMakePlanMany64", plan, rank, n, inembed, istride, idist, onembed,
      ostride, odist, type, batch, workSize);
}

HIPFFT_EXPORT hipfftResult hipfftGetSize(hipfftHandle plan, size_t* workSize) {
  return DynamicCall<decltype(hipfftGetSize), &hipfftGetSize>("hipfftGetSize",
                                                              plan, workSize);
}

HIPFFT_EXPORT hipfftResult hipfftSetAutoAllocation(hipfftHandle plan,
                                                   int autoAllocate) {
  return DynamicCall<decltype(hipfftSetAutoAllocation),
                     &hipfftSetAutoAllocation>("hipfftSetAutoAllocation", plan,
                                               autoAllocate);
}

HIPFFT_EXPORT hipfftResult hipfftSetWorkArea(hipfftHandle plan,
                                             void* workArea) {
  return DynamicCall<decltype(hipfftSetWorkArea), &hipfftSetWorkArea>(
      "hipfftSetWorkArea", plan, workArea);
}

HIPFFT_EXPORT hipfftResult hipfftExecC2C(hipfftHandle plan,
                                         hipfftComplex* idata,
                                         hipfftComplex* odata, int direction) {
  return DynamicCall<decltype(hipfftExecC2C), &hipfftExecC2C>(
      "hipfftExecC2C", plan, idata, odata, direction);
}

HIPFFT_EXPORT hipfftResult hipfftExecR2C(hipfftHandle plan, hipfftReal* idata,
                                         hipfftComplex* odata) {
  return DynamicCall<decltype(hipfftExecR2C), &hipfftExecR2C>(
      "hipfftExecR2C", plan, idata, odata);
}

HIPFFT_EXPORT hipfftResult hipfftExecC2R(hipfftHandle plan,
                                         hipfftComplex* idata,
                                         hipfftReal* odata) {
  return DynamicCall<decltype(hipfftExecC2R), &hipfftExecC2R>(
      "hipfftExecC2R", plan, idata, odata);
}

HIPFFT_EXPORT hipfftResult hipfftExecZ2Z(hipfftHandle plan,
                                         hipfftDoubleComplex* idata,
                                         hipfftDoubleComplex* odata,
                                         int direction) {
  return DynamicCall<decltype(hipfftExecZ2Z), &hipfftExecZ2Z>(
      "hipfftExecZ2Z", plan, idata, odata, direction);
}

HIPFFT_EXPORT hipfftResult hipfftExecD2Z(hipfftHandle plan,
                                         hipfftDoubleReal* idata,
                                         hipfftDoubleComplex* odata) {
  return DynamicCall<decltype(hipfftExecD2Z), &hipfftExecD2Z>(
      "hipfftExecD2Z", plan, idata, odata);
}

HIPFFT_EXPORT hipfftResult hipfftExecZ2D(hipfftHandle plan,
                                         hipfftDoubleComplex* idata,
                                         hipfftDoubleReal* odata) {
  return DynamicCall<decltype(hipfftExecZ2D), &hipfftExecZ2D>(
      "hipfftExecZ2D", plan, idata, odata);
}

HIPFFT_EXPORT hipfftResult hipfftSetStream(hipfftHandle plan,
                                           hipStream_t stream) {
  return DynamicCall<decltype(hipfftSetStream), &hipfftSetStream>(
      "hipfftSetStream", plan, stream);
}

HIPFFT_EXPORT hipfftResult hipfftDestroy(hipfftHandle plan) {
  return DynamicCall<decltype(hipfftDestroy), &hipfftDestroy>("hipfftDestroy",
                                                              plan);
}

HIPFFT_EXPORT hipfftResult hipfftGetProperty(hipfftLibraryPropertyType type,
                                             int* value) {
  return DynamicCall<decltype(hipfftGetProperty), &hipfftGetProperty>(
      "hipfftGetProperty", type, value);
}
