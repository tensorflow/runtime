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
// f32 float kernels
//===----------------------------------------------------------------------===//

// TODO(rmlarsen): Avoid code duplication.
static float TFRTConstantF32(Attribute<float> arg) { return *arg; }

static float TFRTAddF32(Argument<float> arg0, Argument<float> arg1) {
  return *arg0 + *arg1;
}

static Chain TFRTPrintF32(Argument<float> arg, KernelFrame* frame) {
  printf("f32 = %f\n", *arg);
  fflush(stdout);
  return Chain();
}

//===----------------------------------------------------------------------===//
// f64 float kernels
//===----------------------------------------------------------------------===//

static double TFRTConstantF64(Attribute<double> arg) { return *arg; }

static double TFRTAddF64(Argument<double> arg0, Argument<double> arg1) {
  return *arg0 + *arg1;
}

static Chain TFRTPrintF64(Argument<double> arg, KernelFrame* frame) {
  printf("f64 = %f\n", *arg);
  fflush(stdout);
  return Chain();
}

//===----------------------------------------------------------------------===//
// float kernels
//===----------------------------------------------------------------------===//

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

void RegisterFloatKernels(KernelRegistry* registry) {
  registry->AddKernel("tfrt.constant.f32", TFRT_KERNEL(TFRTConstantF32));
  registry->AddKernel("tfrt.add.f32", TFRT_KERNEL(TFRTAddF32));
  registry->AddKernel("tfrt.print.f32", TFRT_KERNEL(TFRTPrintF32));

  registry->AddKernel("tfrt.constant.f64", TFRT_KERNEL(TFRTConstantF64));
  registry->AddKernel("tfrt.add.f64", TFRT_KERNEL(TFRTAddF64));
  registry->AddKernel("tfrt.print.f64", TFRT_KERNEL(TFRTPrintF64));
  registry->AddKernel("tfrt.minimum.f64", TFRT_KERNEL(TFRTMinimum<double>));
  registry->AddKernel("tfrt.div.f64", TFRT_KERNEL(TFRTDiv<double>));
  registry->AddKernel("tfrt.multiply.f64", TFRT_KERNEL(TFRTMultiply<double>));
}

}  // namespace tfrt
