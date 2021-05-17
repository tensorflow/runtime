# Description:
#   Eigen is a C++ template library for linear algebra: vectors,
#   matrices, and related algorithms.

licenses([
    # Note: Eigen is an MPL2 library that includes GPL v3 and LGPL v2.1+ code.
    #       We've taken special care to not reference any restricted code.
    "reciprocal",  # MPL2
    "notice",  # Portions BSD
])

exports_files(["COPYING.MPL2"])

# Files known to be under MPL2 license.
filegroup(
    name = "eigen_header_files",
    srcs = glob(
        [
            "Eigen/**",
            "unsupported/Eigen/CXX11/**",
            "unsupported/Eigen/FFT",
            "unsupported/Eigen/KroneckerProduct",
            "unsupported/Eigen/src/FFT/**",
            "unsupported/Eigen/src/KroneckerProduct/**",
            "unsupported/Eigen/MatrixFunctions",
            "unsupported/Eigen/SpecialFunctions",
            "unsupported/Eigen/src/MatrixFunctions/**",
            "unsupported/Eigen/src/SpecialFunctions/**",
        ],
        exclude = [
            # Guarantees that any non-MPL2 file added to the list above will fail to
            # compile.
            "Eigen/src/Core/util/NonMPL2.h",
            "Eigen/**/CMakeLists.txt",
        ],
    ),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "eigen3",
    hdrs = [":eigen_header_files"],
    defines = [
        # This define (mostly) guarantees we don't link any problematic
        # code. We use it, but we do not rely on it, as evidenced above.
        "EIGEN_MPL2_ONLY",
        "EIGEN_MAX_ALIGN_BYTES=64",
    ],
    visibility = ["//visibility:public"],
)
