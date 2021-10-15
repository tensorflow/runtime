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

//===- concurrent_work_queue.cc - Work Queue Abstraction ------------------===//
//
// This file defines the generic interface for concurrent work queue
// abstractions.

#include "tfrt/host_context/concurrent_work_queue.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {

using llvm::StringMap;

namespace {

using WorkQueueFactoryMap = StringMap<WorkQueueFactory>;

WorkQueueFactoryMap* GetWorkQueueFactories() {
  static WorkQueueFactoryMap* factories = new WorkQueueFactoryMap;
  return factories;
}

}  // namespace

ConcurrentWorkQueue::~ConcurrentWorkQueue() = default;

void RegisterWorkQueueFactory(string_view name, WorkQueueFactory factory) {
  auto p = GetWorkQueueFactories()->try_emplace(name, std::move(factory));
  (void)p;
  assert(p.second && "Factory already registered");
}

std::unique_ptr<ConcurrentWorkQueue> CreateWorkQueue(string_view config) {
  size_t colon = config.find(':');
  string_view name =
      colon == string_view::npos ? config : config.substr(0, colon);
  string_view args = colon == string_view::npos ? "" : config.substr(colon + 1);
  auto* factories = GetWorkQueueFactories();
  auto it = factories->find(name);
  if (it == factories->end()) {
    return nullptr;
  }
  return it->second(args);
}

}  // namespace tfrt
