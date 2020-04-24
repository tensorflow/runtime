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
static float HexConstantF32(Attribute<float> arg) { return *arg; }

static float HexAddF32(Argument<float> arg0, Argument<float> arg1) {
  return *arg0 + *arg1;
}

static Chain HexPrintF32(Argument<float> arg, KernelFrame* frame) {
  printf("f32 = %f\n", *arg);
  fflush(stdout);
  return Chain();
}

//===----------------------------------------------------------------------===//
// f64 float kernels
//===----------------------------------------------------------------------===//

static double HexConstantF64(Attribute<double> arg) { return *arg; }

static double HexAddF64(Argument<double> arg0, Argument<double> arg1) {
  return *arg0 + *arg1;
}

static Chain HexPrintF64(Argument<double> arg, KernelFrame* frame) {
  printf("f64 = %f\n", *arg);
  fflush(stdout);
  return Chain();
}

//===----------------------------------------------------------------------===//
// float kernels
//===----------------------------------------------------------------------===//

template <typename T>
static T HexMinimum(T v1, T v2) {
  return std::min(v1, v2);
}

template <typename T>
static Expected<T> HexDiv(T arg0, T arg1) {
  if (arg1 == 0) {
    return MakeStringError("Divide by zero");
  }
  return arg0 / arg1;
}

template <typename T>
static T HexMultiply(T arg0, T arg1) {
  return arg0 * arg1;
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void RegisterFloatKernels(KernelRegistry* registry) {
  registry->AddKernel("hex.constant.f32", TFRT_KERNEL(HexConstantF32));
  registry->AddKernel("hex.add.f32", TFRT_KERNEL(HexAddF32));
  registry->AddKernel("hex.print.f32", TFRT_KERNEL(HexPrintF32));

  registry->AddKernel("hex.constant.f64", TFRT_KERNEL(HexConstantF64));
  registry->AddKernel("hex.add.f64", TFRT_KERNEL(HexAddF64));
  registry->AddKernel("hex.print.f64", TFRT_KERNEL(HexPrintF64));
  registry->AddKernel("hex.minimum.f64", TFRT_KERNEL(HexMinimum<double>));
  registry->AddKernel("hex.div.f64", TFRT_KERNEL(HexDiv<double>));
  registry->AddKernel("hex.multiply.f64", TFRT_KERNEL(HexMultiply<double>));
}

}  // namespace tfrt
