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

//===- float_kernels.cc ---------------------------------------------------===//
//
// This file implements host executor kernels for floating point types.
//
//===----------------------------------------------------------------------===//

#include "tfrt/basic_kernels/basic_kernels.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/error_util.h"

namespace tfrt {

//===----------------------------------------------------------------------===//
// float kernels
//===----------------------------------------------------------------------===//

static Chain TFRTPrintF32(Argument<float> arg, AsyncKernelFrame* frame) {
  printf("f32 = %f\n", *arg);
  fflush(stdout);
  return Chain();
}

static Chain TFRTPrintF64(Argument<double> arg, AsyncKernelFrame* frame) {
  printf("f64 = %f\n", *arg);
  fflush(stdout);
  return Chain();
}

template <typename T>
static T TFRTConstant(Attribute<T> arg) {
  return *arg;
}

template <typename T>
static T TFRTAdd(Argument<T> arg0, Argument<T> arg1) {
  return *arg0 + *arg1;
}

template <typename T>
static T TFRTMinimum(T v1, T v2) {
  return std::min(v1, v2);
}

template <typename T>
static Expected<T> TFRTDiv(T arg0, T arg1) {
  if (arg1 == 0) {
    return MakeStringError("Divide by zero");
  }
  return arg0 / arg1;
}

template <typename T>
static T TFRTMultiply(T arg0, T arg1) {
  return arg0 * arg1;
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

template <typename T>
void RegisterFloatKernelsForType(KernelRegistry* registry,
                                 const std::string& suffix) {
  registry->AddKernel("tfrt.constant." + suffix, TFRT_KERNEL(TFRTConstant<T>));
  registry->AddKernel("tfrt.add." + suffix, TFRT_KERNEL(TFRTAdd<T>));
  registry->AddKernel("tfrt.minimum." + suffix, TFRT_KERNEL(TFRTMinimum<T>));
  registry->AddKernel("tfrt.div." + suffix, TFRT_KERNEL(TFRTDiv<T>));
  registry->AddKernel("tfrt.multiply." + suffix, TFRT_KERNEL(TFRTMultiply<T>));
}

void RegisterFloatKernels(KernelRegistry* registry) {
  registry->AddKernel("tfrt.print.f32", TFRT_KERNEL(TFRTPrintF32));
  registry->AddKernel("tfrt.print.f64", TFRT_KERNEL(TFRTPrintF64));

  RegisterFloatKernelsForType<float>(registry, "f32");
  RegisterFloatKernelsForType<double>(registry, "f64");
}

}  // namespace tfrt
