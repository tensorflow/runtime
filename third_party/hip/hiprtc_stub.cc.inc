hiprtcResult hiprtcVersion(int* major, int* minor) {
  return DynamicCall<decltype(hiprtcVersion), &hiprtcVersion>("hiprtcVersion",
                                                              major, minor);
}

hiprtcResult hiprtcAddNameExpression(hiprtcProgram prog,
                                     const char* name_expression) {
  return DynamicCall<decltype(hiprtcAddNameExpression),
                     &hiprtcAddNameExpression>("hiprtcAddNameExpression", prog,
                                               name_expression);
}

hiprtcResult hiprtcCompileProgram(hiprtcProgram prog, int numOptions,
                                  const char** options) {
  return DynamicCall<decltype(hiprtcCompileProgram), &hiprtcCompileProgram>(
      "hiprtcCompileProgram", prog, numOptions, options);
}

hiprtcResult hiprtcCreateProgram(hiprtcProgram* prog, const char* src,
                                 const char* name, int numHeaders,
                                 const char** headers,
                                 const char** includeNames) {
  return DynamicCall<decltype(hiprtcCreateProgram), &hiprtcCreateProgram>(
      "hiprtcCreateProgram", prog, src, name, numHeaders, headers,
      includeNames);
}

hiprtcResult hiprtcDestroyProgram(hiprtcProgram* prog) {
  return DynamicCall<decltype(hiprtcDestroyProgram), &hiprtcDestroyProgram>(
      "hiprtcDestroyProgram", prog);
}

hiprtcResult hiprtcGetLoweredName(hiprtcProgram prog,
                                  const char* name_expression,
                                  const char** lowered_name) {
  return DynamicCall<decltype(hiprtcGetLoweredName), &hiprtcGetLoweredName>(
      "hiprtcGetLoweredName", prog, name_expression, lowered_name);
}

hiprtcResult hiprtcGetProgramLog(hiprtcProgram prog, char* log) {
  return DynamicCall<decltype(hiprtcGetProgramLog), &hiprtcGetProgramLog>(
      "hiprtcGetProgramLog", prog, log);
}

hiprtcResult hiprtcGetProgramLogSize(hiprtcProgram prog, size_t* logSizeRet) {
  return DynamicCall<decltype(hiprtcGetProgramLogSize),
                     &hiprtcGetProgramLogSize>("hiprtcGetProgramLogSize", prog,
                                               logSizeRet);
}

hiprtcResult hiprtcGetCode(hiprtcProgram prog, char* code) {
  return DynamicCall<decltype(hiprtcGetCode), &hiprtcGetCode>("hiprtcGetCode",
                                                              prog, code);
}

hiprtcResult hiprtcGetCodeSize(hiprtcProgram prog, size_t* codeSizeRet) {
  return DynamicCall<decltype(hiprtcGetCodeSize), &hiprtcGetCodeSize>(
      "hiprtcGetCodeSize", prog, codeSizeRet);
}
