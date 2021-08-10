HIPFFT_EXPORT hipfftResult
hipfftPlan1d(hipfftHandle* plan, int nx, hipfftType type,
             int batch /* deprecated - use hipfftPlanMany */) {
  return DynamicCall<decltype(hipfftPlan1d), &hipfftPlan1d>(
      "hipfftPlan1d", plan, nx, type, batch);
}

HIPFFT_EXPORT hipfftResult hipfftPlan2d(hipfftHandle* plan, int nx, int ny,
                                        hipfftType type) {
  return DynamicCall<decltype(hipfftPlan2d), &hipfftPlan2d>("hipfftPlan2d",
                                                            plan, nx, ny, type);
}

HIPFFT_EXPORT hipfftResult hipfftPlan3d(hipfftHandle* plan, int nx, int ny,
                                        int nz, hipfftType type) {
  return DynamicCall<decltype(hipfftPlan3d), &hipfftPlan3d>(
      "hipfftPlan3d", plan, nx, ny, nz, type);
}

HIPFFT_EXPORT hipfftResult hipfftPlanMany(hipfftHandle* plan, int rank, int* n,
                                          int* inembed, int istride, int idist,
                                          int* onembed, int ostride, int odist,
                                          hipfftType type, int batch) {
  return DynamicCall<decltype(hipfftPlanMany), &hipfftPlanMany>(
      "hipfftPlanMany", plan, rank, n, inembed, istride, idist, onembed,
      ostride, odist, type, batch);
}

HIPFFT_EXPORT hipfftResult hipfftGetSize(hipfftHandle plan, size_t* workSize) {
  return DynamicCall<decltype(hipfftGetSize), &hipfftGetSize>("hipfftGetSize",
                                                              plan, workSize);
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
