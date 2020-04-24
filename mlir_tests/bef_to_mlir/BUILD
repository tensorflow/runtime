load("@tf_runtime//mlir_tests:lit.bzl", "glob_lit_tests")

licenses(["notice"])

glob_lit_tests(
    data = [":test_utilities"],
    #=== GOOGLE_PIPER: tf_runtime/mlir_tests:run_lit.sh ===#
    test_file_exts = [
        "mlir",
    ],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "@llvm-project//llvm:FileCheck",
        #=== GOOGLE_PIPER: llvm-project/mlir:run_lit.sh ===#
        "@tf_runtime//tools:tfrt_opt",
        "@tf_runtime//tools:tfrt_translate",
    ],
)
