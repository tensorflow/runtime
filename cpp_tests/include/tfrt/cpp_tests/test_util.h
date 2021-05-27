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

// This file defines utilities related to setting up unit tests.
#ifndef TFRT_CPP_TESTS_TEST_UTIL_H_
#define TFRT_CPP_TESTS_TEST_UTIL_H_

#include "tfrt/dtype/dtype.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/dense_host_tensor_view.h"
#include "tfrt/tensor/tensor_metadata.h"

namespace tfrt {

inline std::unique_ptr<HostContext> CreateHostContext() {
  auto decoded_diagnostic_handler = [&](const DecodedDiagnostic& diag) {
    abort();
  };
  std::unique_ptr<ConcurrentWorkQueue> work_queue =
      CreateSingleThreadedWorkQueue();
  std::unique_ptr<HostAllocator> host_allocator = CreateMallocAllocator();
  return std::make_unique<HostContext>(decoded_diagnostic_handler,
                                       std::move(host_allocator),
                                       std::move(work_queue));
}

template <typename T>
DenseHostTensor CreateDummyTensor(ArrayRef<ssize_t> dims,
                                  HostContext* host_ctx) {
  const TensorMetadata metadata(GetDType<T>(), dims);
  auto dht =
      DenseHostTensor::CreateUninitialized(metadata, host_ctx).getValue();
  MutableDHTArrayView<T> view(&dht);
  for (int i = 0, s = dht.NumElements(); i < s; i++) {
    view[i] = i;
  }
  return dht;
}

}  // namespace tfrt

#endif  // TFRT_CPP_TESTS_TEST_UTIL_H_
