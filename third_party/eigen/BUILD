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

# Helper to exclude all files with extensions in glob for top-level headers.
ALL_FILES_WITH_EXTENSIONS = glob(["**/*.*"])

# Top-level headers, excluding anything in one of the  ../src/.. directories.
EIGEN_HEADERS = glob(
    [
        "Eigen/*",
        "unsupported/Eigen/*",
        "unsupported/Eigen/CXX11/*",
    ],
    exclude = [
        "**/src/**",
    ] + ALL_FILES_WITH_EXTENSIONS,
)

# Internal eigen headers, known to be under an MPL2 license.
EIGEN_MPL2_SOURCES = glob(
    [
        "Eigen/**/src/**/*.h",
        "unsupported/Eigen/**/src/**/*.h",
    ],
    exclude = [
        # This guarantees that any file depending on non MPL2 licensed code
        # will not compile.
        "Eigen/src/Core/util/NonMPL2.h",
    ],
)

# Files known to be under MPL2 license.
filegroup(
    name = "eigen_header_files",
    srcs = EIGEN_HEADERS + EIGEN_MPL2_SOURCES,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "eigen3",
    srcs = EIGEN_MPL2_SOURCES,
    hdrs = EIGEN_HEADERS,
    defines = [
        # This define (mostly) guarantees we don't link any problematic
        # code. We use it, but we do not rely on it, as evidenced above.
        "EIGEN_MPL2_ONLY",
        "EIGEN_MAX_ALIGN_BYTES=64",
    ],
    visibility = ["//visibility:public"],
)
