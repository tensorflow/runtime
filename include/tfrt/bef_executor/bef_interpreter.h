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

// Synchronous function interpreter
//
// This file declares the BEFInterpreter class.

#ifndef TFRT_BEF_EXECUTOR_BEF_INTERPRETER_H_
#define TFRT_BEF_EXECUTOR_BEF_INTERPRETER_H_

#include <memory>

#include "tfrt/support/forward_decls.h"

namespace tfrt {

class Function;
class ExecutionContext;
class Value;

class BEFInterpreterImpl;

/// A BEFInterpreter runs a BEF function containing a stream of synchronous
/// kernels. Multiple interpreters can be active at one time, e.g. due to
/// concurrent control flow constructs.
//
// BEFInterpreter is thread-compatible.
class BEFInterpreter final {
 public:
  // `func` must be a SyncBEFFunction. It must out-live the BEFInterpreter
  // object, as BEFInterpreter only keeps a reference to `func`.
  explicit BEFInterpreter(const Function& func);
  ~BEFInterpreter();

  Error Execute(const ExecutionContext& exec_ctx, ArrayRef<Value*> arguments,
                ArrayRef<Value*> results);

 private:
  std::unique_ptr<BEFInterpreterImpl> impl_;
};

}  // namespace tfrt

#endif  // TFRT_BEF_EXECUTOR_BEF_INTERPRETER_H_
