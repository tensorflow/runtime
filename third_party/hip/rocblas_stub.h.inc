#define ROCBLAS_EXPORT __attribute__((visibility("default")))

typedef enum rocblas_operation_ {
  rocblas_operation_none = 111,
  rocblas_operation_transpose = 112,
  rocblas_operation_conjugate_transpose = 113,
} rocblas_operation;

typedef enum rocblas_fill_ {
  rocblas_fill_upper = 121,
  rocblas_fill_lower = 122,
  rocblas_fill_full = 123,
} rocblas_fill;

typedef enum rocblas_diagonal_ {
  rocblas_diagonal_non_unit = 131,
  rocblas_diagonal_unit = 132,
} rocblas_diagonal;

typedef enum rocblas_side_ {
  rocblas_side_left = 141,
  rocblas_side_right = 142,
  rocblas_side_both = 143,
} rocblas_side;

typedef enum rocblas_status_ {
  rocblas_status_success = 0,
  rocblas_status_invalid_handle = 1,
  rocblas_status_not_implemented = 2,
  rocblas_status_invalid_pointer = 3,
  rocblas_status_invalid_size = 4,
  rocblas_status_memory_error = 5,
  rocblas_status_internal_error = 6,
  rocblas_status_perf_degraded = 7,
  rocblas_status_size_query_mismatch = 8,
  rocblas_status_size_increased = 9,
  rocblas_status_size_unchanged = 10,
  rocblas_status_invalid_value = 11,
  rocblas_status_continue = 12,
  rocblas_status_check_numerics_fail = 13,
} rocblas_status;

typedef enum rocblas_datatype_ {
  rocblas_datatype_f16_r = 150,
  rocblas_datatype_f32_r = 151,
  rocblas_datatype_f64_r = 152,
  rocblas_datatype_f16_c = 153,
  rocblas_datatype_f32_c = 154,
  rocblas_datatype_f64_c = 155,
  rocblas_datatype_i8_r = 160,
  rocblas_datatype_u8_r = 161,
  rocblas_datatype_i32_r = 162,
  rocblas_datatype_u32_r = 163,
  rocblas_datatype_i8_c = 164,
  rocblas_datatype_u8_c = 165,
  rocblas_datatype_i32_c = 166,
  rocblas_datatype_u32_c = 167,
  rocblas_datatype_bf16_r = 168,
  rocblas_datatype_bf16_c = 169,
} rocblas_datatype;

typedef enum rocblas_pointer_mode_ {
  rocblas_pointer_mode_host = 0,
  rocblas_pointer_mode_device = 1,
} rocblas_pointer_mode;

typedef enum rocblas_gemm_algo_ {
  rocblas_gemm_algo_standard = 0x0,
} rocblas_gemm_algo;

typedef enum rocblas_gemm_flags_ {
  rocblas_gemm_flags_none = 0x0,
  rocblas_gemm_flags_pack_int8x4 = 0x1,
} rocblas_gemm_flags;

ROCBLAS_EXPORT rocblas_status rocblas_create_handle(rocblas_handle* handle);

ROCBLAS_EXPORT rocblas_status rocblas_destroy_handle(rocblas_handle handle);

ROCBLAS_EXPORT rocblas_status rocblas_set_stream(rocblas_handle handle,
                                                 hipStream_t stream);

ROCBLAS_EXPORT rocblas_status rocblas_get_stream(rocblas_handle handle,
                                                 hipStream_t* stream);

ROCBLAS_EXPORT rocblas_status rocblas_set_pointer_mode(
    rocblas_handle handle, rocblas_pointer_mode pointer_mode);

ROCBLAS_EXPORT rocblas_status rocblas_get_pointer_mode(
    rocblas_handle handle, rocblas_pointer_mode* pointer_mode);

ROCBLAS_EXPORT rocblas_status rocblas_gemm_ex(
    rocblas_handle handle, rocblas_operation transA, rocblas_operation transB,
    rocblas_int m, rocblas_int n, rocblas_int k, const void* alpha,
    const void* a, rocblas_datatype a_type, rocblas_int lda, const void* b,
    rocblas_datatype b_type, rocblas_int ldb, const void* beta, const void* c,
    rocblas_datatype c_type, rocblas_int ldc, void* d, rocblas_datatype d_type,
    rocblas_int ldd, rocblas_datatype compute_type, rocblas_gemm_algo algo,
    int32_t solution_index, uint32_t flags);

ROCBLAS_EXPORT rocblas_status rocblas_gemm_strided_batched_ex(
    rocblas_handle handle, rocblas_operation transA, rocblas_operation transB,
    rocblas_int m, rocblas_int n, rocblas_int k, const void* alpha,
    const void* a, rocblas_datatype a_type, rocblas_int lda,
    rocblas_stride stride_a, const void* b, rocblas_datatype b_type,
    rocblas_int ldb, rocblas_stride stride_b, const void* beta, const void* c,
    rocblas_datatype c_type, rocblas_int ldc, rocblas_stride stride_c, void* d,
    rocblas_datatype d_type, rocblas_int ldd, rocblas_stride stride_d,
    rocblas_int batch_count, rocblas_datatype compute_type,
    rocblas_gemm_algo algo, int32_t solution_index, uint32_t flags);

ROCBLAS_EXPORT rocblas_status rocblas_axpy_ex(
    rocblas_handle handle, rocblas_int n, const void* alpha,
    rocblas_datatype alpha_type, const void* x, rocblas_datatype x_type,
    rocblas_int incx, void* y, rocblas_datatype y_type, rocblas_int incy,
    rocblas_datatype execution_type);
