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

//===- map_dataset.cc -------------------------------------------*- C++ -*-===//
//
// This file implements MapDataset class which wraps around another Dataset
// instance and transforms the element before returning it to the caller.
//
//===----------------------------------------------------------------------===//

#include "map_dataset.h"

namespace tfrt {
namespace data {

//===----------------------------------------------------------------------===//
// MapDataset methods
//===----------------------------------------------------------------------===//
RCReference<Iterator> MapDataset::MakeIterator() {
  return TakeRef(host_->Construct<MapDatasetIterator>(FormRef(this)));
}

//===----------------------------------------------------------------------===//
// MapDatasetIterator methods
//===----------------------------------------------------------------------===//
IterationResult MapDatasetIterator::GetNext(const ExecutionContext& exec_ctx) {
  auto input = input_iterator_->GetNext(exec_ctx);
  const Function* map_fn = parent_dataset_->map_fn_.get();

  auto values = std::move(input.values);
  auto eof = std::move(input.eof);

  // IDEA(donglin): consider extending RCArray to support CopyRef() without
  // doing shallow copy.
  auto additional_fn_args = parent_dataset_->additional_fn_args_.CopyRef();
  auto result =
      EnqueueFunction(map_fn, std::move(additional_fn_args),
                      RCArray<AsyncValue>(std::move(values)), exec_ctx);
  return IterationResult::Pending(std::move(result), std::move(eof));
}

}  // namespace data
}  // namespace tfrt
