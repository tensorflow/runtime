package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

cc_library(
    name = "cudnn_frontend",
    hdrs = glob([
        "include/*.h",
        "include/contrib/nlohmann/json/json.hpp",
    ]),
    defines = ["NV_CUDNN_DISABLE_EXCEPTION"],
    includes = ["include"],
)
