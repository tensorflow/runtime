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

hiprtcResult hiprtcVersion(int* major, int* minor);

hiprtcResult hiprtcAddNameExpression(hiprtcProgram prog,
                                     const char* name_expression);

hiprtcResult hiprtcCompileProgram(hiprtcProgram prog, int numOptions,
                                  const char** options);

hiprtcResult hiprtcCreateProgram(hiprtcProgram* prog, const char* src,
                                 const char* name, int numHeaders,
                                 const char** headers,
                                 const char** includeNames);

hiprtcResult hiprtcDestroyProgram(hiprtcProgram* prog);

hiprtcResult hiprtcGetLoweredName(hiprtcProgram prog,
                                  const char* name_expression,
                                  const char** lowered_name);

hiprtcResult hiprtcGetProgramLog(hiprtcProgram prog, char* log);

hiprtcResult hiprtcGetProgramLogSize(hiprtcProgram prog, size_t* logSizeRet);

hiprtcResult hiprtcGetCode(hiprtcProgram prog, char* code);

hiprtcResult hiprtcGetCodeSize(hiprtcProgram prog, size_t* codeSizeRet);
