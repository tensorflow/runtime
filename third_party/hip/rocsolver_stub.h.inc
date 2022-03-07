ROCSOLVER_EXPORT rocblas_status rocsolver_spotrf(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n, float* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_dpotrf(rocblas_handle handle,
                                                 const rocblas_fill uplo,
                                                 const rocblas_int n, double* A,
                                                 const rocblas_int lda,
                                                 rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_cpotrf(
    rocblas_handle handle, const rocblas_fill uplo, const rocblas_int n,
    rocblas_float_complex* A, const rocblas_int lda, rocblas_int* info);

ROCSOLVER_EXPORT rocblas_status rocsolver_zpotrf(
    rocblas_handle handle, const rocblas_fill uplo, const rocblas_int n,
    rocblas_double_complex* A, const rocblas_int lda, rocblas_int* info);
