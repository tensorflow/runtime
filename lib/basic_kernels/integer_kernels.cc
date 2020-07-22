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

//===- integer_kernels.cc -------------------------------------------------===//
//
// This file implements a host executor kernels for integer types.
//
//===----------------------------------------------------------------------===//

#include "llvm_derived/Support/raw_ostream.h"
#include "tfrt/basic_kernels/basic_kernels.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/host_context/sync_kernel_utils.h"
#include "tfrt/support/error_util.h"

namespace tfrt {
namespace {

//===----------------------------------------------------------------------===//
// tfrt integer constant kernels
//===----------------------------------------------------------------------===//

template <typename T>
T TFRTConstant(Attribute<T> arg) {
  return arg.get();
}

//===----------------------------------------------------------------------===//
// tfrt add kernels
//===----------------------------------------------------------------------===//
template <typename T>
T TFRTAdd(T arg0, T arg1) {
  return arg0 + arg1;
}

//===----------------------------------------------------------------------===//
// tfrt minus kernels
//===----------------------------------------------------------------------===//
template <typename T>
T TFRTMinus(T arg0, T arg1) {
  return arg0 - arg1;
}

//===----------------------------------------------------------------------===//
// tfrt equal kernels
//===----------------------------------------------------------------------===//
template <typename T>
bool TFRTEqual(T arg0, T arg1) {
  return arg0 == arg1;
}

//===----------------------------------------------------------------------===//
// tfrt less equal kernels
//===----------------------------------------------------------------------===//
template <typename T>
bool TFRTLessEqual(T arg0, T arg1) {
  return arg0 <= arg1;
}

//===----------------------------------------------------------------------===//
// tfrt div kernel. Returns quotient and remainder
//===----------------------------------------------------------------------===//
template <typename T>
Expected<std::pair<T, T>> TFRTDiv(T arg0, T arg1) {
  if (arg1 == 0) {
    return MakeStringError("Divide by zero");
  }
  return std::pair<T, T>{arg0 / arg1, arg0 % arg1};
}

//===----------------------------------------------------------------------===//
// tfrt print integer kernels
//===----------------------------------------------------------------------===//

Chain TFRTPrintI1(bool arg) {
  tfrt::outs() << "int1 = " << static_cast<int32_t>(arg) << '\n';
  tfrt::outs().flush();
  return Chain();
}

Chain TFRTPrintI32(int32_t arg) {
  tfrt::outs() << "int32 = " << arg << '\n';
  tfrt::outs().flush();
  return Chain();
}

Chain TFRTPrintI64(int64_t arg) {
  tfrt::outs() << "int64 = " << arg << '\n';
  tfrt::outs().flush();
  return Chain();
}

//===----------------------------------------------------------------------===//
// tfrt cast kernels
//===----------------------------------------------------------------------===//

template <typename Tin, typename Tout>
static Tout TFRTCast(Tin value) {
  return static_cast<Tout>(value);
}

}  // namespace

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void RegisterIntegerKernels(KernelRegistry* registry) {
  registry->AddKernel("tfrt.constant.i32", TFRT_KERNEL(TFRTConstant<int32_t>));
  registry->AddKernel("tfrt.constant.i64", TFRT_KERNEL(TFRTConstant<int64_t>));

  registry->AddKernel("tfrt.add.i32", TFRT_KERNEL(TFRTAdd<int32_t>));
  registry->AddKernel("tfrt.add.i64", TFRT_KERNEL(TFRTAdd<int64_t>));

  registry->AddKernel("tfrt.minus.i32", TFRT_KERNEL(TFRTMinus<int32_t>));
  registry->AddKernel("tfrt.minus.i64", TFRT_KERNEL(TFRTMinus<int64_t>));

  registry->AddKernel("tfrt.equal.i32", TFRT_KERNEL(TFRTEqual<int32_t>));
  registry->AddKernel("tfrt.equal.i64", TFRT_KERNEL(TFRTEqual<int64_t>));

  registry->AddKernel("tfrt.lessequal.i32",
                      TFRT_KERNEL(TFRTLessEqual<int32_t>));
  registry->AddKernel("tfrt.lessequal.i64",
                      TFRT_KERNEL(TFRTLessEqual<int64_t>));

  registry->AddKernel("tfrt.div.i32", TFRT_KERNEL(TFRTDiv<int32_t>));
  registry->AddKernel("tfrt.div.i64", TFRT_KERNEL(TFRTDiv<int64_t>));

  registry->AddKernel("tfrt.print.i1", TFRT_KERNEL(TFRTPrintI1));
  registry->AddKernel("tfrt.print.i32", TFRT_KERNEL(TFRTPrintI32));
  registry->AddKernel("tfrt.print.i64", TFRT_KERNEL(TFRTPrintI64));

  registry->AddKernel("tfrt.cast.i64_to_f32",
                      TFRT_KERNEL(TFRTCast<int64_t, float>));
  registry->AddKernel("tfrt.cast.f32_to_i64",
                      TFRT_KERNEL(TFRTCast<float, int64_t>));
  registry->AddKernel("tfrt.cast.i64_to_f64",
                      TFRT_KERNEL(TFRTCast<int64_t, double>));
  registry->AddKernel("tfrt.cast.f64_to_i64",
                      TFRT_KERNEL(TFRTCast<double, int64_t>));

  // Register synchronous kernels.
  registry->AddSyncKernel("tfrt.constant_s.i32",
                          TFRT_SYNC_KERNEL(TFRTConstant<int32_t>));
  registry->AddSyncKernel("tfrt.constant_s.i64",
                          TFRT_SYNC_KERNEL(TFRTConstant<int64_t>));

  registry->AddSyncKernel("tfrt.add_s.i32", TFRT_SYNC_KERNEL(TFRTAdd<int32_t>));
  registry->AddSyncKernel("tfrt.add_s.i64", TFRT_SYNC_KERNEL(TFRTAdd<int64_t>));
}

}  // namespace tfrt
