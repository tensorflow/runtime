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
#include "tfrt/support/error_util.h"

namespace tfrt {
namespace {

//===----------------------------------------------------------------------===//
// hex integer constant kernels
//===----------------------------------------------------------------------===//

template <typename T>
T HexConstant(Attribute<T> arg) {
  return arg.get();
}

//===----------------------------------------------------------------------===//
// hex add kernels
//===----------------------------------------------------------------------===//
template <typename T>
T HexAdd(T arg0, T arg1) {
  return arg0 + arg1;
}

//===----------------------------------------------------------------------===//
// hex minus kernels
//===----------------------------------------------------------------------===//
template <typename T>
T HexMinus(T arg0, T arg1) {
  return arg0 - arg1;
}

//===----------------------------------------------------------------------===//
// hex equal kernels
//===----------------------------------------------------------------------===//
template <typename T>
bool HexEqual(T arg0, T arg1) {
  return arg0 == arg1;
}

//===----------------------------------------------------------------------===//
// hex less equal kernels
//===----------------------------------------------------------------------===//
template <typename T>
bool HexLessEqual(T arg0, T arg1) {
  return arg0 <= arg1;
}

//===----------------------------------------------------------------------===//
// hex div kernel. Returns quotient and remainder
//===----------------------------------------------------------------------===//
template <typename T>
Expected<std::pair<T, T>> HexDiv(T arg0, T arg1) {
  if (arg1 == 0) {
    return MakeStringError("Divide by zero");
  }
  return std::pair<T, T>{arg0 / arg1, arg0 % arg1};
}

//===----------------------------------------------------------------------===//
// hex print integer kernels
//===----------------------------------------------------------------------===//

Chain HexPrintI1(bool arg) {
  tfrt::outs() << "int1 = " << static_cast<int32_t>(arg) << '\n';
  tfrt::outs().flush();
  return Chain();
}

Chain HexPrintI32(int32_t arg) {
  tfrt::outs() << "int32 = " << arg << '\n';
  tfrt::outs().flush();
  return Chain();
}

Chain HexPrintI64(int64_t arg) {
  tfrt::outs() << "int64 = " << arg << '\n';
  tfrt::outs().flush();
  return Chain();
}

//===----------------------------------------------------------------------===//
// hex cast kernels
//===----------------------------------------------------------------------===//

template <typename Tin, typename Tout>
static Tout HexCast(Tin value) {
  return static_cast<Tout>(value);
}

}  // namespace

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void RegisterIntegerKernels(KernelRegistry* registry) {
  registry->AddKernel("hex.constant.i32", TFRT_KERNEL(HexConstant<int32_t>));
  registry->AddKernel("hex.constant.i64", TFRT_KERNEL(HexConstant<int64_t>));

  registry->AddKernel("hex.add.i32", TFRT_KERNEL(HexAdd<int32_t>));
  registry->AddKernel("hex.add.i64", TFRT_KERNEL(HexAdd<int64_t>));

  registry->AddKernel("hex.minus.i32", TFRT_KERNEL(HexMinus<int32_t>));
  registry->AddKernel("hex.minus.i64", TFRT_KERNEL(HexMinus<int64_t>));

  registry->AddKernel("hex.equal.i32", TFRT_KERNEL(HexEqual<int32_t>));
  registry->AddKernel("hex.equal.i64", TFRT_KERNEL(HexEqual<int64_t>));

  registry->AddKernel("hex.lessequal.i32", TFRT_KERNEL(HexLessEqual<int32_t>));
  registry->AddKernel("hex.lessequal.i64", TFRT_KERNEL(HexLessEqual<int64_t>));

  registry->AddKernel("hex.div.i32", TFRT_KERNEL(HexDiv<int32_t>));
  registry->AddKernel("hex.div.i64", TFRT_KERNEL(HexDiv<int64_t>));

  registry->AddKernel("hex.print.i1", TFRT_KERNEL(HexPrintI1));
  registry->AddKernel("hex.print.i32", TFRT_KERNEL(HexPrintI32));
  registry->AddKernel("hex.print.i64", TFRT_KERNEL(HexPrintI64));

  registry->AddKernel("hex.cast.i64_to_f64",
                      TFRT_KERNEL(HexCast<int64_t, double>));
  registry->AddKernel("hex.cast.f64_to_i64",
                      TFRT_KERNEL(HexCast<double, int64_t>));
}

}  // namespace tfrt
