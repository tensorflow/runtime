package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

# Note: run_lit.sh does not exist in open-source environment.
exports_files(["run_lit.sh"])

# LLVM lit configurations for lit test in OSS environment.
filegroup(
    name = "litcfgs",
    srcs = glob(["lit*py"]),
)
