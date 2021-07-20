typedef enum rocfft_status_e {
  rocfft_status_success,
  rocfft_status_failure,
  rocfft_status_invalid_arg_value,
  rocfft_status_invalid_dimensions,
  rocfft_status_invalid_array_type,
  rocfft_status_invalid_strides,
  rocfft_status_invalid_distance,
  rocfft_status_invalid_offset,
  rocfft_status_invalid_work_buffer,
} rocfft_status;

typedef enum rocfft_transform_type_e {
  rocfft_transform_type_complex_forward,
  rocfft_transform_type_complex_inverse,
  rocfft_transform_type_real_forward,
  rocfft_transform_type_real_inverse,
} rocfft_transform_type;

typedef enum rocfft_precision_e {
  rocfft_precision_single,
  rocfft_precision_double,
} rocfft_precision;

typedef enum rocfft_result_placement_e {
  rocfft_placement_inplace,
  rocfft_placement_notinplace,
} rocfft_result_placement;

typedef enum rocfft_array_type_e {
  rocfft_array_type_complex_interleaved,
  rocfft_array_type_complex_planar,
  rocfft_array_type_real,
  rocfft_array_type_hermitian_interleaved,
  rocfft_array_type_hermitian_planar,
  rocfft_array_type_unset,
} rocfft_array_type;

rocfft_status rocfft_setup();

rocfft_status rocfft_cleanup();

rocfft_status rocfft_plan_create(rocfft_plan* plan,
                                 rocfft_result_placement placement,
                                 rocfft_transform_type transform_type,
                                 rocfft_precision precision, size_t dimensions,
                                 const size_t* lengths,
                                 size_t number_of_transforms,
                                 const rocfft_plan_description description);

rocfft_status rocfft_execute(const rocfft_plan plan, void* in_buffer[],
                             void* out_buffer[], rocfft_execution_info info);

rocfft_status rocfft_plan_destroy(rocfft_plan plan);

rocfft_status rocfft_plan_description_set_data_layout(
    rocfft_plan_description description, const rocfft_array_type in_array_type,
    const rocfft_array_type out_array_type, const size_t* in_offsets,
    const size_t* out_offsets, const size_t in_strides_size,
    const size_t* in_strides, const size_t in_distance,
    const size_t out_strides_size, const size_t* out_strides,
    const size_t out_distance);

rocfft_status rocfft_get_version_string(char* buf, size_t len);

rocfft_status rocfft_plan_get_work_buffer_size(const rocfft_plan plan,
                                               size_t* size_in_bytes);

rocfft_status rocfft_plan_get_print(const rocfft_plan plan);

rocfft_status rocfft_plan_description_create(
    rocfft_plan_description* description);

rocfft_status rocfft_plan_description_destroy(
    rocfft_plan_description description);

rocfft_status rocfft_execution_info_create(rocfft_execution_info* info);

rocfft_status rocfft_execution_info_destroy(rocfft_execution_info info);

rocfft_status rocfft_execution_info_set_work_buffer(rocfft_execution_info info,
                                                    void* work_buffer,
                                                    const size_t size_in_bytes);

rocfft_status rocfft_execution_info_set_stream(rocfft_execution_info info,
                                               void* stream);
