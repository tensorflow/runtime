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

//===- bef_interpreter.cc--------------------------------------------------===//
//
// This file implements the Interpreter for BEF files.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstdio>

#include "bef_file_impl.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/location.h"
#include "tfrt/host_context/sync_kernel_frame.h"
#include "tfrt/support/bef_encoding.h"
#include "tfrt/support/bef_reader.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {

/// A BEFInterpreter runs a BEF function containing a stream of synchronous
/// kernels. Multiple interpreters can be active at one time, e.g. due to
/// concurrent control flow constructs.
//
// BEFInterpreter is thread-compatible.
class BEFInterpreter final {
 public:
  // `func` must out-live the BEFInterpreter object, as BEFInterpreter only
  // keeps a reference to `func`.
  explicit BEFInterpreter(const SyncBEFFunction& func);

  Error Execute(const ExecutionContext& exec_ctx, ArrayRef<Value*> arguments,
                ArrayRef<Value*> results);

 private:
  // Set up the registers for the function computation.
  void SetupRegisters(ArrayRef<Value*> arguments, ArrayRef<Value*> results);

  struct Register {
    uint32_t user_count;
    Value* value;
  };

  const SyncBEFFunction& func_;

  SmallVector<Register, 16> registers_;

  // Store local Values used in the computation.
  SmallVector<Value, 16> local_values_;
};

//===----------------------------------------------------------------------===//
// Core interpreter logic
//===----------------------------------------------------------------------===//

BEFInterpreter::BEFInterpreter(const SyncBEFFunction& func) : func_{func} {
  auto register_infos = func_.register_infos();

  size_t num_registers = register_infos.size();

  // Set up local values.
  local_values_.resize(num_registers - func.num_arguments() -
                       func.num_results());

  registers_.reserve(num_registers);
  auto local_value_index = 0;

  // Set up Value pointer in registers for local values, as this is not changed
  // during the function evaluation.
  for (auto& reg_info : register_infos) {
    auto& reg = registers_.emplace_back();

    if (reg_info.is_arg_or_result) {
#ifndef NDEBUG
      // Initialize argument or result register Value in the debug mode.
      reg.value = nullptr;
#endif
    } else {
      reg.value = &local_values_[local_value_index];
      ++local_value_index;
    }
  }
}

void BEFInterpreter::SetupRegisters(ArrayRef<Value*> arguments,
                                    ArrayRef<Value*> results) {
  auto register_infos = func_.register_infos();

  // Set up argument Value
  for (size_t i = 0; i < arguments.size(); ++i) {
    assert(register_infos[i].is_arg_or_result);
    assert(!registers_[i].value);
    registers_[i].value = arguments[i];
  }

  // Set up result Value
  auto result_index = 0;
  for (auto reg_idx : func_.result_regs()) {
    assert(register_infos[reg_idx].is_arg_or_result);
    assert(!registers_[reg_idx].value);

    registers_[reg_idx].value = results[result_index];
    ++result_index;
  }

  // Set up register user counts
  for (size_t reg_index = 0, end = register_infos.size(); reg_index != end;
       ++reg_index) {
    registers_[reg_index].user_count = register_infos[reg_index].user_count;
  }
}

Error BEFInterpreter::Execute(const ExecutionContext& exec_ctx,
                              ArrayRef<Value*> arguments,
                              ArrayRef<Value*> results) {
  assert(arguments.size() == func_.num_arguments() &&
         "incorrect number of arguments passed to function call");
  assert(results.size() == func_.num_results() &&
         "incorrect number of results passed to function call");

  SetupRegisters(arguments, results);

  SyncKernelFrameBuilder kernel_frame(exec_ctx);
  SmallVector<Value*, 8> retired_values;

  for (auto kernel_offset : func_.kernel_offsets()) {
    BEFKernel kernel(func_.kernels().data() +
                     kernel_offset / kKernelEntryAlignment);

    // Find the kernel function.
    SyncKernelImplementation kernel_fn =
        func_.bef_file()->GetSyncKernel(kernel.kernel_code());
    assert(kernel_fn != nullptr);

    int entry_offset = 0;

    // Set up arguments.
    auto arguments =
        kernel.GetKernelEntries(entry_offset, kernel.num_arguments());
    for (auto reg_idx : arguments) {
      auto& reg = registers_[reg_idx];

      kernel_frame.AddArg(reg.value);

      --reg.user_count;
      assert(reg.user_count >= 0);
      if (reg.user_count == 0) {
        retired_values.emplace_back(reg.value);
      }
    }

    // Set up attributes.
    entry_offset += arguments.size();
    auto attributes =
        kernel.GetKernelEntries(entry_offset, kernel.num_attributes());
    for (auto attribute_offset : attributes) {
      // We pass the pointer here because this attribute could be an array of
      // size 0.
      kernel_frame.AddAttribute(func_.bef_file()->attribute_section_.data() +
                                attribute_offset);
    }

    // Set up function attributes.
    entry_offset += attributes.size();
    auto functions =
        kernel.GetKernelEntries(entry_offset, kernel.num_functions());
    for (auto fn_idx : functions) {
      // Functions are passed as their corresponding `Function`.
      kernel_frame.AddAttribute(func_.bef_file()->functions_[fn_idx].get());
    }

    // Set up results.
    entry_offset += functions.size();
    auto results = kernel.GetKernelEntries(entry_offset, kernel.num_results());
    for (auto reg_index : results) {
      auto& reg = registers_[reg_index];
      kernel_frame.AddResult(reg.value);
      // If there is no use for the result, mark it as retired.
      if (reg.user_count == 0) {
        retired_values.emplace_back(reg.value);
      }
    }

    kernel_fn(&kernel_frame);

    // Free values that are no longer needed.
    for (auto value : retired_values) {
      value->reset();
    }

    // Check for error.
    if (auto error = kernel_frame.TakeError()) {
      return error;
    }

    // Reset state for the next kernel evaluation.
    kernel_frame.Reset();
    retired_values.clear();
  }

#ifndef NDEBUG
  // In debug mode, reset all the argument and result registers to make
  // debugging easier.
  for (size_t i = 0; i < arguments.size(); ++i) {
    registers_[i].value = nullptr;
  }

  for (auto reg_idx : func_.result_regs()) {
    registers_[reg_idx].value = nullptr;
  }

  for (auto& value : local_values_) {
    assert(!value.HasValue());
  }
#endif

  return Error::success();
}

//===----------------------------------------------------------------------===//
// SyncBEFFunction implementation
//===----------------------------------------------------------------------===//

// Execute SyncBEFFunction synchronously. Return excution error in the Error
// result.
Error SyncBEFFunction::SyncExecute(const ExecutionContext& exec_ctx,
                                   ArrayRef<Value*> arguments,
                                   ArrayRef<Value*> results) const {
  BEFInterpreter interpreter{*this};
  return interpreter.Execute(exec_ctx, arguments, results);
}

}  // namespace tfrt
