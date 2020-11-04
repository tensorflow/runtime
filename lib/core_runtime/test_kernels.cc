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

//===- core_runtime/test_kernels.cc ---------------------------------------===//
//
// This library contains test kernels needed by core runtime unit tests.
//
//===----------------------------------------------------------------------===//

#include "llvm_derived/Support/raw_ostream.h"
#include "tfrt/core_runtime/kernels.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/core_runtime/tensor_handle.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/tensor/tensor.h"

namespace tfrt {

static void TensorToTensorHandleWithErrorMetadata(
    Argument<Tensor> arg, Result<TensorHandle> tensorhandle_output,
    KernelErrorHandler handler, const ExecutionContext& exec_ctx) {
  auto metadata = handler.EmitError("invalid tensor metadata");
  // TODO(b/158775215): TensorHandle should take device from Tensor argument.
  tensorhandle_output.Emplace(
      exec_ctx.host()->GetHostDeviceRef(),
      AsyncValueRef<TensorMetadata>(std::move(metadata)),
      AsyncValueRef<Tensor>(FormRef(arg.value())));
}

static void corert_op_attrs_print(Argument<OpAttrs> attrs,
                                  Argument<Chain> in_chain,
                                  Result<Chain> out_chain) {
  attrs->Print(tfrt::outs());
  tfrt::outs().flush();
  out_chain.Set(in_chain);
}

static void corert_op_attrs_freeze(Argument<OpAttrs> attrs,
                                   Argument<Chain> in_chain,
                                   Result<OpAttrsRef> result,
                                   Result<Chain> out_chain) {
  result.Emplace(attrs->freeze());
  out_chain.Set(in_chain);
}

static void corert_op_attrs_ref_print(Argument<OpAttrsRef> frozen,
                                      Argument<Chain> in_chain,
                                      Result<Chain> out_chain) {
  frozen->Print(tfrt::outs());
  tfrt::outs().flush();
  out_chain.Set(in_chain);
}

void RegisterCoreRuntimeTestKernels(KernelRegistry* registry) {
  registry->AddKernel("tfrt_test.tensorhandle_with_error_metadata",
                      TFRT_KERNEL(TensorToTensorHandleWithErrorMetadata));
  registry->AddKernel("tfrt_test.corert.op_attrs_print",
                      TFRT_KERNEL(corert_op_attrs_print));
  registry->AddKernel("tfrt_test.corert.op_attrs_freeze",
                      TFRT_KERNEL(corert_op_attrs_freeze));
  registry->AddKernel("tfrt_test.corert.op_attrs_ref_print",
                      TFRT_KERNEL(corert_op_attrs_ref_print));
}
}  // namespace tfrt
