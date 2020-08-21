/*
 * Copyright 2020 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//===- range_dataset.cc -----------------------------------------*- C++ -*-===//
//
// This file implements RangeDataset class which yields a step-separated range
// of values.
//
//===----------------------------------------------------------------------===//

#include "range_dataset.h"

namespace tfrt {
namespace data {

//===----------------------------------------------------------------------===//
// RangeDatasetIterator methods
//===----------------------------------------------------------------------===//
IterationResult RangeDatasetIterator::GetNext(
    const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();
  bool has_next = (dataset_->step_ > 0 && next_ < dataset_->stop_) ||
                  (dataset_->step_ < 0 && next_ > dataset_->stop_);
  if (!has_next) {
    return IterationResult::Eof(host, 1);
  }

  SmallVector<RCReference<AsyncValue>, 4> values;
  switch (dataset_->element_type_.kind()) {
#define DTYPE_NUMERIC(ENUM)                                            \
  case DType::ENUM:                                                    \
    values.push_back(                                                  \
        MakeAvailableAsyncValueRef<TypeForDTypeKind<DType::ENUM>>(     \
            host, static_cast<TypeForDTypeKind<DType::ENUM>>(next_))); \
    break;

#include "tfrt/dtype/dtype.def"  // NOLINT
#undef DTYPE_NUMERIC
    default:
      return IterationResult::Error(
          MakeErrorAsyncValueRef(host, "Unsupported data type"), 1);
  }

  next_ += dataset_->step_;

  return IterationResult::Values(std::move(values), host);
}

//===----------------------------------------------------------------------===//
// RangeDataset methods
//===----------------------------------------------------------------------===//
RCReference<Iterator> RangeDataset::MakeIterator() {
  return TakeRef(host_->Construct<RangeDatasetIterator>(FormRef(this)));
}

}  // namespace data
}  // namespace tfrt
