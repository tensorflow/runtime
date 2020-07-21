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

//===- bef_file.h -----------------------------------------------*- C++ -*-===//
//
// This file declares constants used when interfacing with the "Binary Executor
// Format" (BEF) files.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_BEF_EXECUTOR_BEF_FILE_H_
#define TFRT_BEF_EXECUTOR_BEF_FILE_H_

#include <functional>

#include "tfrt/support/forward_decls.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {

class AsyncValue;
class DecodedDiagnostic;
class Function;
class HostAllocator;
class KernelRegistry;

// Instances of this class represent a BEF file in memory.  The in-memory
// representation of BEF files is HostContext independent, allowing reuse across
// multiple contexts if desired.
class BEFFile : public ReferenceCounted<BEFFile> {
 public:
  typedef std::function<void(DecodedDiagnostic)> ErrorHandler;

  // Open and read a BEF file, setting up our internal state and returning a
  // pointer to our initialized object on success.  On failure, an error
  // message is emitted to the error_handler and nullptr is returned.
  //
  // TODO: This should (optionally) manage ownership of the underlying data
  // passed in, taking a closure to run when the lifetime of the BEFFile is
  // done.
  static RCReference<BEFFile> Open(ArrayRef<uint8_t> file,
                                   KernelRegistry* registry,
                                   ErrorHandler error_handler,
                                   HostAllocator* host_allocator);

  // Get a list of functions out of the BEF file.
  void GetFunctionList(SmallVectorImpl<const Function*>* result) const;

  // Return the Function record with the specified name, or null if it isn't
  // found in this BEF file.
  const Function* GetFunction(string_view function_name) const;

  virtual ~BEFFile() = 0;
};

// Execute SyncBEFFunction synchronously. Return excution error in the Error
// return value.
//
// TODO(jingdong): Remove this function once we implement
// SyncBEFFunction::Execute() that takes and returns AsyncValue. This is
// required for now, as we want to be able to call
// SyncBEFFunction::SyncExecute() without exposing SyncBEFFunction in the header
// file.
class ExecutionContext;
class Value;
Error ExecuteSyncBEFFunction(const Function& func,
                             const ExecutionContext& exec_ctx,
                             ArrayRef<Value*> arguments,
                             ArrayRef<Value*> results);

}  // namespace tfrt

#endif  // TFRT_BEF_EXECUTOR_BEF_FILE_H_
