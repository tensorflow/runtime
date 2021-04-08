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

// This file uses a static constructor to automatically register all of the
// kernels in this directory.  This can be used to simplify clients that don't
// care about selective registration of kernels.

#include "tfrt/host_context/kernel_registry.h"
#include "tfrt/tensor/conversion_registry.h"
#include "tfrt/tensor/coo_host_tensor.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/dense_host_tensor_kernels.h"
#include "tfrt/tensor/scalar_host_tensor.h"
#include "tfrt/tensor/string_host_tensor.h"
#include "tfrt/tensor/string_host_tensor_kernels.h"
#include "tfrt/tensor/tensor_shape.h"

namespace tfrt {

// This is the entrypoint to the library.
static void Register(KernelRegistry* registry) {
  RegisterTensorShapeKernels(registry);
  RegisterDenseHostTensorKernels(registry);
  RegisterCooHostTensorKernels(registry);
  RegisterStringHostTensorKernels(registry);
}

TFRT_STATIC_KERNEL_REGISTRATION(Register);

// TODO(fishx): Create a macro for this registration.
static bool host_conversion_fn_registration = []() {
  AddStaticTensorConversionFn(RegisterCooHostTensorConversionFn);
  AddStaticTensorConversionFn(RegisterDenseHostTensorConversionFn);
  AddStaticTensorConversionFn(RegisterStringHostTensorConversionFn);
  AddStaticTensorConversionFn(RegisterScalarHostTensorConversionFn);
  return true;
}();

}  // namespace tfrt
