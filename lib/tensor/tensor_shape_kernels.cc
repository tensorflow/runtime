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

//===- tensor_shape_kernels.cc --------------------------------------------===//
//
// This file implements a few kernels that manipulate TensorShapes.  These may
// themselves eventually be useful, but for right now they are primarily
// intended for testing.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/raw_ostream.h"
#include "llvm_derived/Support/raw_ostream.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/tensor/tensor_shape.h"

namespace tfrt {

// We define this kernel the hard way, since it is variadic.
static void TsBuildShape(AsyncKernelFrame* frame) {
  frame->AssertArity(/*nargs*/ 0, /*nattributes*/ 1, /*nresults*/ 1);

  ArrayRef<ssize_t> elements = frame->GetArrayAttributeAt<ssize_t>(0).data();
  frame->EmplaceResult<TensorShape>(elements);
}

static void TsPrintShape(Argument<TensorShape> arg) {
  tfrt::outs() << "shape = " << *arg << '\n';
  tfrt::outs().flush();
}

static bool TsEqualShape(Argument<TensorShape> lhs, Argument<TensorShape> rhs) {
  return *lhs == *rhs;
}

static ssize_t TsGetDimension(const TensorShape& shape, int idx) {
  return shape.GetDimensionSize(idx);
}

template <size_t Rank>
static FixedRankShape<Rank> TsAsFixedRankShape(Argument<TensorShape> arg) {
  return FixedRankShape<Rank>(*arg);
}

template <size_t Rank>
static void TsPrintFixedRankShape(Argument<FixedRankShape<Rank>> arg) {
  tfrt::outs() << "fixed_rank_shape = " << *arg << '\n';
  tfrt::outs().flush();
}

void RegisterTensorShapeKernels(KernelRegistry* registry) {
  registry->AddKernel("ts.build_shape", TsBuildShape);
  registry->AddKernel("ts.print_shape", TFRT_KERNEL(TsPrintShape));
  registry->AddKernel("ts.equal_shape", TFRT_KERNEL(TsEqualShape));
  registry->AddKernel("ts.get_dimension", TFRT_KERNEL(TsGetDimension));
  registry->AddKernel("ts.as_fixed_rank_shape.1",
                      TFRT_KERNEL(TsAsFixedRankShape<1>));
  registry->AddKernel("ts.as_fixed_rank_shape.2",
                      TFRT_KERNEL(TsAsFixedRankShape<2>));
  registry->AddKernel("ts.as_fixed_rank_shape.3",
                      TFRT_KERNEL(TsAsFixedRankShape<3>));

  registry->AddKernel("ts.print_fixed_rank_shape.1",
                      TFRT_KERNEL(TsPrintFixedRankShape<1>));
  registry->AddKernel("ts.print_fixed_rank_shape.2",
                      TFRT_KERNEL(TsPrintFixedRankShape<2>));
  registry->AddKernel("ts.print_fixed_rank_shape.3",
                      TFRT_KERNEL(TsPrintFixedRankShape<3>));
}

}  // namespace tfrt
