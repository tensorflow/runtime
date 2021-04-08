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

// Function Cache
//
// This file declares FunctionCache, which caches the programs that are
// registered and instantiated from remote requests.

#ifndef TFRT_DISTRIBUTED_RUNTIME_FUNCTION_CACHE_H_
#define TFRT_DISTRIBUTED_RUNTIME_FUNCTION_CACHE_H_

#include <unordered_map>

#include "tfrt/bef_converter/bef_buffer.h"
#include "tfrt/bef_executor/bef_file.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/resource_context.h"

namespace tfrt {

class HostContext;

// TODO(bramandia): Replace with TFRT FunctionLibrary once available.
class FunctionCache {
 public:
  explicit FunctionCache(HostContext* host_context) : host_(host_context) {}

  // Register the given program. A program can have multiple functions in it.
  // The program_name serves as both unique ID of this program.
  Error Register(const std::string& program_name, BEFBuffer bef_buffer);

  // Create BEFFile corresponding to the program with the given name.
  // A struct representing a BEFFile and the respective buffer.
  struct CachedBEF {
    CachedBEF() {}

    RCReference<BEFFile> bef_file;
    bool require_distributed_context = false;
    bool require_preallocated_outputs = false;
  };
  CachedBEF* Prepare(const std::string& program_name);

 private:
  HostContext* host_;

  mutex cached_bef_mutex_;
  // Map from the program name to the CachedBEF.
  std::unordered_map<std::string, std::pair<BEFBuffer, CachedBEF>> cached_bef_
      TFRT_GUARDED_BY(cached_bef_mutex_);
};

}  // namespace tfrt
#endif  // TFRT_DISTRIBUTED_RUNTIME_FUNCTION_CACHE_H_
