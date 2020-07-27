// Copyright 2020 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- contraction_kernel.cc ------------------------------------*- C++ -*-===//
//
// Runtime check for disabling custom contraction kernels.
//
//===----------------------------------------------------------------------===//

#include "tfrt/common/compat/eigen/contraction_kernel.h"

#include <mutex>

// We need a pair of compile time and runtime flags to disable compilation of
// custom contraction kernels for unsupported architectures (e.g. Android,
// iOS, ARM and PPC CPUs, etc...), and to be able to fallback on default Eigen
// contraction kernel at runtime.
//
// MKL-DNN contraction kernel might generate different results depending on the
// available CPU instructions (avx2, avx512, etc...) and this might be
// undesirable for tests based on golden data. Whether or not having such tests
// is a good idea is a separate topic.
//
// Example:
//   bazel test --test_env=TFRT_DISABLE_EIGEN_MKLDNN_CONTRACTION_KERNEL=true \
//       //path/to/a/test:target

#if defined(TFRT_EIGEN_USE_CUSTOM_CONTRACTION_KERNEL)

namespace Eigen {
namespace internal {

bool UseCustomContractionKernelsTFRT() {
  static bool use_custom_contraction_kernel = true;

  static std::once_flag initialized;
  std::call_once(initialized, [&] {
    char* flag = std::getenv("TFRT_DISABLE_EIGEN_MKLDNN_CONTRACTION_KERNEL");
    if (flag && (strcmp(flag, "true") == 0 || strcmp(flag, "1") == 0)) {
      use_custom_contraction_kernel = false;
    }
  });
  return use_custom_contraction_kernel;
}

}  // namespace internal
}  // namespace Eigen
#endif
