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

//===- function_cache.cc - Function Cache ----------------*- C++ -*--------===//
//
// Contains implementation of FunctionCache class.

#include "tfrt/distributed_runtime/function_cache.h"

#include "llvm/ADT/DenseMap.h"
#include "tfrt/bef_converter/bef_buffer.h"
#include "tfrt/bef_executor/bef_file.h"
#include "tfrt/distributed_runtime/callback_registry.h"
#include "tfrt/distributed_runtime/fabric_communicator.h"
#include "tfrt/distributed_runtime/remote_object_manager.h"
#include "tfrt/distributed_runtime/server_context.h"
#include "tfrt/host_context/function.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {

Error FunctionCache::Register(const std::string& program_name,
                              BEFBuffer bef_buffer) {
  RCReference<BEFFile> bef_file =
      tfrt::BEFFile::Open(bef_buffer, host_->GetKernelRegistry(),
                          host_->diag_handler(), host_->allocator());

  if (!bef_file) {
    return llvm::make_error<MalformattedMlirFileErrorInfo>(
        StrCat("Failed to open lowered BEF for function ", program_name, "."));
  }
  const Function* fn = bef_file->GetFunction(program_name);
  int arg_index = 0;
  bool require_distributed_context = false;
  if (fn->num_arguments() > 0 &&
      fn->argument_types()[0].GetName() == "!tfrt_dist.dist_context") {
    require_distributed_context = true;
    arg_index++;
  }
  // If the next argument is RemoteObjectId, this is the RemoteObjectId outputs.
  // We have to pass all the output remote object IDs.
  bool require_preallocated_outputs = false;
  if (fn->num_arguments() > arg_index &&
      fn->argument_types()[arg_index].GetName() ==
          "!tfrt_dist.remote_object_id") {
    require_preallocated_outputs = true;
  }
  mutex_lock lock(cached_bef_mutex_);
  if (cached_bef_.find(program_name) != cached_bef_.end()) {
    return llvm::make_error<RemoteFunctionAlreadyExistsErrorInfo>(
        StrCat("Program ", program_name, " already registered."));
  }
  auto& cached = cached_bef_[program_name];
  cached.first = std::move(bef_buffer);
  cached.second.bef_file = bef_file.CopyRef();
  cached.second.require_distributed_context = require_distributed_context;
  cached.second.require_preallocated_outputs = require_preallocated_outputs;
  return Error::success();
}

FunctionCache::CachedBEF* FunctionCache::Prepare(
    const std::string& program_name) {
  mutex_lock lock(cached_bef_mutex_);
  auto iter = cached_bef_.find(program_name);
  if (iter != cached_bef_.end()) {
    return &iter->second.second;
  }
  return nullptr;
}

}  // namespace tfrt
