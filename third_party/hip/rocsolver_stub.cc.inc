ROCSOLVER_EXPORT rocblas_status rocsolver_spotrf(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n, float* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* info) {
  return DynamicCall<decltype(rocsolver_spotrf), &rocsolver_spotrf>(
      "rocsolver_spotrf", handle, uplo, n, A, lda, info);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dpotrf(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n, double* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* info) {
  return DynamicCall<decltype(rocsolver_dpotrf), &rocsolver_dpotrf>(
      "rocsolver_dpotrf", handle, uplo, n, A, lda, info);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_cpotrf(
    rocblas_handle handle, const rocblas_fill uplo, const rocblas_int n,
    rocblas_float_complex* A, const rocblas_int lda, rocblas_int* info) {
  return DynamicCall<decltype(rocsolver_cpotrf), &rocsolver_cpotrf>(
      "rocsolver_cpotrf", handle, uplo, n, A, lda, info);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zpotrf(
    rocblas_handle handle, const rocblas_fill uplo, const rocblas_int n,
    rocblas_double_complex* A, const rocblas_int lda, rocblas_int* info) {
  return DynamicCall<decltype(rocsolver_zpotrf), &rocsolver_zpotrf>(
      "rocsolver_zpotrf", handle, uplo, n, A, lda, info);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_spotrf_batched(
    rocblas_handle handle, const rocblas_fill uplo, const rocblas_int n,
    float* const A[], const rocblas_int lda, rocblas_int* info,
    const rocblas_int batch_count) {
  return DynamicCall<decltype(rocsolver_spotrf_batched),
                     &rocsolver_spotrf_batched>(
      "rocsolver_spotrf_batched", handle, uplo, n, A, lda, info, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dpotrf_batched(
    rocblas_handle handle, const rocblas_fill uplo, const rocblas_int n,
    double* const A[], const rocblas_int lda, rocblas_int* info,
    const rocblas_int batch_count) {
  return DynamicCall<decltype(rocsolver_dpotrf_batched),
                     &rocsolver_dpotrf_batched>(
      "rocsolver_dpotrf_batched", handle, uplo, n, A, lda, info, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_cpotrf_batched(
    rocblas_handle handle, const rocblas_fill uplo, const rocblas_int n,
    rocblas_float_complex* const A[], const rocblas_int lda, rocblas_int* info,
    const rocblas_int batch_count) {
  return DynamicCall<decltype(rocsolver_cpotrf_batched),
                     &rocsolver_cpotrf_batched>(
      "rocsolver_cpotrf_batched", handle, uplo, n, A, lda, info, batch_count);
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zpotrf_batched(
    rocblas_handle handle, const rocblas_fill uplo, const rocblas_int n,
    rocblas_double_complex* const A[], const rocblas_int lda, rocblas_int* info,
    const rocblas_int batch_count) {
  return DynamicCall<decltype(rocsolver_zpotrf_batched),
                     &rocsolver_zpotrf_batched>(
      "rocsolver_zpotrf_batched", handle, uplo, n, A, lda, info, batch_count);
}
