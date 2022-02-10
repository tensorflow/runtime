ROCBLAS_EXPORT rocblas_status rocblas_create_handle(rocblas_handle* handle) {
  return DynamicCall<decltype(rocblas_create_handle), &rocblas_create_handle>(
      "rocblas_create_handle", handle);
}

ROCBLAS_EXPORT rocblas_status rocblas_destroy_handle(rocblas_handle handle) {
  return DynamicCall<decltype(rocblas_destroy_handle), &rocblas_destroy_handle>(
      "rocblas_destroy_handle", handle);
}

ROCBLAS_EXPORT rocblas_status rocblas_set_stream(rocblas_handle handle,
                                                 hipStream_t stream) {
  return DynamicCall<decltype(rocblas_set_stream), &rocblas_set_stream>(
      "rocblas_set_stream", handle, stream);
}

ROCBLAS_EXPORT rocblas_status rocblas_get_stream(rocblas_handle handle,
                                                 hipStream_t* stream) {
  return DynamicCall<decltype(rocblas_get_stream), &rocblas_get_stream>(
      "rocblas_get_stream", handle, stream);
}

ROCBLAS_EXPORT rocblas_status rocblas_set_pointer_mode(
    rocblas_handle handle, rocblas_pointer_mode pointer_mode) {
  return DynamicCall<decltype(rocblas_set_pointer_mode),
                     &rocblas_set_pointer_mode>("rocblas_set_pointer_mode",
                                                handle, pointer_mode);
}

ROCBLAS_EXPORT rocblas_status rocblas_get_pointer_mode(
    rocblas_handle handle, rocblas_pointer_mode* pointer_mode) {
  return DynamicCall<decltype(rocblas_get_pointer_mode),
                     &rocblas_get_pointer_mode>("rocblas_get_pointer_mode",
                                                handle, pointer_mode);
}

ROCBLAS_EXPORT rocblas_status rocblas_gemm_ex(
    rocblas_handle handle, rocblas_operation transA, rocblas_operation transB,
    rocblas_int m, rocblas_int n, rocblas_int k, const void* alpha,
    const void* a, rocblas_datatype a_type, rocblas_int lda, const void* b,
    rocblas_datatype b_type, rocblas_int ldb, const void* beta, const void* c,
    rocblas_datatype c_type, rocblas_int ldc, void* d, rocblas_datatype d_type,
    rocblas_int ldd, rocblas_datatype compute_type, rocblas_gemm_algo algo,
    int32_t solution_index, uint32_t flags) {
  return DynamicCall<decltype(rocblas_gemm_ex), &rocblas_gemm_ex>(
      "rocblas_gemm_ex", handle, transA, transB, m, n, k, alpha, a, a_type, lda,
      b, b_type, ldb, beta, c, c_type, ldc, d, d_type, ldd, compute_type, algo,
      solution_index, flags);
}

ROCBLAS_EXPORT rocblas_status rocblas_gemm_strided_batched_ex(
    rocblas_handle handle, rocblas_operation transA, rocblas_operation transB,
    rocblas_int m, rocblas_int n, rocblas_int k, const void* alpha,
    const void* a, rocblas_datatype a_type, rocblas_int lda,
    rocblas_stride stride_a, const void* b, rocblas_datatype b_type,
    rocblas_int ldb, rocblas_stride stride_b, const void* beta, const void* c,
    rocblas_datatype c_type, rocblas_int ldc, rocblas_stride stride_c, void* d,
    rocblas_datatype d_type, rocblas_int ldd, rocblas_stride stride_d,
    rocblas_int batch_count, rocblas_datatype compute_type,
    rocblas_gemm_algo algo, int32_t solution_index, uint32_t flags) {
  return DynamicCall<decltype(rocblas_gemm_strided_batched_ex),
                     &rocblas_gemm_strided_batched_ex>(
      "rocblas_gemm_strided_batched_ex", handle, transA, transB, m, n, k, alpha,
      a, a_type, lda, stride_a, b, b_type, ldb, stride_b, beta, c, c_type, ldc,
      stride_c, d, d_type, ldd, stride_d, batch_count, compute_type, algo,
      solution_index, flags);
}

ROCBLAS_EXPORT rocblas_status rocblas_axpy_ex(
    rocblas_handle handle, rocblas_int n, const void* alpha,
    rocblas_datatype alpha_type, const void* x, rocblas_datatype x_type,
    rocblas_int incx, void* y, rocblas_datatype y_type, rocblas_int incy,
    rocblas_datatype execution_type) {
  return DynamicCall<decltype(rocblas_axpy_ex), &rocblas_axpy_ex>(
      "rocblas_axpy_ex", handle, n, alpha, alpha_type, x, x_type, incx, y,
      y_type, incy, execution_type);
}

ROCBLAS_EXPORT rocblas_status rocblas_strsm_batched(
    rocblas_handle handle, rocblas_side side, rocblas_fill uplo,
    rocblas_operation transA, rocblas_diagonal diag, rocblas_int m,
    rocblas_int n, const float* alpha, const float* const A[], rocblas_int lda,
    float* const B[], rocblas_int ldb, rocblas_int batch_count) {
  return DynamicCall<decltype(rocblas_strsm_batched), &rocblas_strsm_batched>(
      "rocblas_strsm_batched", handle, side, uplo, transA, diag, m, n, alpha, A,
      lda, B, ldb, batch_count);
}

ROCBLAS_EXPORT rocblas_status rocblas_dtrsm_batched(
    rocblas_handle handle, rocblas_side side, rocblas_fill uplo,
    rocblas_operation transA, rocblas_diagonal diag, rocblas_int m,
    rocblas_int n, const double* alpha, const double* const A[],
    rocblas_int lda, double* const B[], rocblas_int ldb,
    rocblas_int batch_count) {
  return DynamicCall<decltype(rocblas_dtrsm_batched), &rocblas_dtrsm_batched>(
      "rocblas_dtrsm_batched", handle, side, uplo, transA, diag, m, n, alpha, A,
      lda, B, ldb, batch_count);
}

ROCBLAS_EXPORT rocblas_status rocblas_ctrsm_batched(
    rocblas_handle handle, rocblas_side side, rocblas_fill uplo,
    rocblas_operation transA, rocblas_diagonal diag, rocblas_int m,
    rocblas_int n, const rocblas_float_complex* alpha,
    const rocblas_float_complex* const A[], rocblas_int lda,
    rocblas_float_complex* const B[], rocblas_int ldb,
    rocblas_int batch_count) {
  return DynamicCall<decltype(rocblas_ctrsm_batched), &rocblas_ctrsm_batched>(
      "rocblas_ctrsm_batched", handle, side, uplo, transA, diag, m, n, alpha, A,
      lda, B, ldb, batch_count);
}

ROCBLAS_EXPORT rocblas_status rocblas_ztrsm_batched(
    rocblas_handle handle, rocblas_side side, rocblas_fill uplo,
    rocblas_operation transA, rocblas_diagonal diag, rocblas_int m,
    rocblas_int n, const rocblas_double_complex* alpha,
    const rocblas_double_complex* const A[], rocblas_int lda,
    rocblas_double_complex* const B[], rocblas_int ldb,
    rocblas_int batch_count) {
  return DynamicCall<decltype(rocblas_ztrsm_batched), &rocblas_ztrsm_batched>(
      "rocblas_ztrsm_batched", handle, side, uplo, transA, diag, m, n, alpha, A,
      lda, B, ldb, batch_count);
}
