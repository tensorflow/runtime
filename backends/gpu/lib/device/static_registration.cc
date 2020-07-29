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

//===- static_registration.cc ---------------------------------------------===//
//
// This file uses a static constructor to automatically register gpu device.
//
//===----------------------------------------------------------------------===//

#include "tfrt/gpu/device/conversion_function.h"
#include "tfrt/host_context/device.h"
#include "tfrt/tensor/conversion_registry.h"

namespace tfrt {

static DeviceTypeRegistration register_device_type_gpu("gpu");

// TODO(fishx): Create a macro for this registration.
static bool gpu_conversion_fn_registration = []() {
  AddStaticTensorConversionFn(gpu::RegisterGpuTensorConversionFn);
  return true;
}();

}  // namespace tfrt
