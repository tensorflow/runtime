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

//===- test_static_registration.cc ----------------------------------------===//
//
// This file uses a static constructor to automatically register all of the
// cuda_test kernels.
//
//===----------------------------------------------------------------------===//

#include "tfrt/gpu/kernels/cuda_opdefs/cuda_test_ops.h"

namespace tfrt {
namespace cuda {

// Static initialization for dialect registration.
static ::mlir::DialectRegistration<CUDATestDialect> cuda_registration;

}  // namespace cuda
}  // namespace tfrt
