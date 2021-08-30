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

// This file implements a few kernels that manipulate TensorShapes.  These may
// themselves eventually be useful, but for right now they are primarily
// intended for testing.

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm_derived/Support/raw_ostream.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/host_context/sync_kernel_utils.h"
#include "tfrt/tensor/tensor_shape.h"

namespace tfrt {

// We define this kernel the hard way, since it is variadic.
static void TsBuildShape(AsyncKernelFrame* frame) {
  frame->AssertArity(/*nargs*/ 0, /*nattributes*/ 1, /*nresults*/ 1);

  ArrayRef<Index> elements = frame->GetArrayAttributeAt<Index>(0).data();
  frame->EmplaceResult<TensorShape>(elements);
}

// Builds a `TensorShape` from an array attribute.
static TensorShape TsSyncBuildShape(ArrayAttribute<Index> shape) {
  return TensorShape(shape.data());
}

static Chain TsPrintShape(Argument<TensorShape> arg) {
  tfrt::outs() << "shape = " << *arg << '\n';
  tfrt::outs().flush();
  return Chain();
}

static bool TsEqualShape(Argument<TensorShape> lhs, Argument<TensorShape> rhs) {
  return *lhs == *rhs;
}

static Index TsGetDimension(const TensorShape& shape, int idx) {
  return shape.GetDimensionSize(idx);
}

static Index TsGetNumElements(const TensorShape& shape) {
  return shape.GetNumElements();
}

template <size_t Rank>
static FixedRankShape<Rank> TsAsFixedRankShape(Argument<TensorShape> arg) {
  return FixedRankShape<Rank>(*arg);
}

template <size_t Rank>
static Chain TsPrintFixedRankShape(Argument<FixedRankShape<Rank>> arg) {
  tfrt::outs() << "fixed_rank_shape = " << *arg << '\n';
  tfrt::outs().flush();
  return Chain();
}

static PartialTensorShape TsBuildPartialShape(ArrayAttribute<Index> shape) {
  return PartialTensorShape(shape.data());
}

static PartialTensorShape TsBuildUnrankedPartialShape() {
  return PartialTensorShape(llvm::None);
}

static Chain TsPrintPartialShape(Argument<PartialTensorShape> arg) {
  tfrt::outs() << "partial_tensor_shape = " << *arg << '\n';
  tfrt::outs().flush();
  return Chain();
}

static Expected<TensorShape> TsToShape(const PartialTensorShape& arg) {
  return arg.ToTensorShape();
}

static PartialTensorShape TsToPartialShape(const TensorShape& arg) {
  // TODO(haoliang): Ideally we should be able to build a PartialTensorShape
  // directly from a TensorShape via a constructor method, so that we can avoid
  // allocating the temporary dimensions array.
  llvm::SmallVector<Index, 4> dims;
  dims.reserve(arg.GetRank());
  arg.GetDimensions(&dims);
  return PartialTensorShape(llvm::makeArrayRef(dims));
}

void RegisterTensorShapeKernels(KernelRegistry* registry) {
  registry->AddKernel("ts.build_shape", TsBuildShape);
  registry->AddKernel("ts.print_shape", TFRT_KERNEL(TsPrintShape));
  registry->AddKernel("ts.equal_shape", TFRT_KERNEL(TsEqualShape));
  registry->AddKernel("ts.get_dimension", TFRT_KERNEL(TsGetDimension));
  registry->AddKernel("ts.get_num_elements", TFRT_KERNEL(TsGetNumElements));
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
  registry->AddKernel("ts.build_partial_shape",
                      TFRT_KERNEL(TsBuildPartialShape));
  registry->AddKernel("ts.build_unranked_partial_shape",
                      TFRT_KERNEL(TsBuildUnrankedPartialShape));
  registry->AddKernel("ts.print_partial_shape",
                      TFRT_KERNEL(TsPrintPartialShape));
  registry->AddKernel("ts.to_shape", TFRT_KERNEL(TsToShape));
  registry->AddKernel("ts.to_partial_shape", TFRT_KERNEL(TsToPartialShape));

  // Register sync kernels.
  registry->AddSyncKernel("ts_sync.build_shape",
                          TFRT_SYNC_KERNEL(TsSyncBuildShape));
  registry->AddSyncKernel("ts_sync.build_partial_shape",
                          TFRT_SYNC_KERNEL(TsBuildPartialShape));
  registry->AddSyncKernel("ts_sync.to_partial_shape",
                          TFRT_SYNC_KERNEL(TsToPartialShape));
}

}  // namespace tfrt
