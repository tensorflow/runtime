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
  struct KernelEntry {
    SyncKernelImplementation kernel_fn;
    // KernelEntry starting location in BEF.
    const uint32_t* kernel_start;
    // All attributes, including, function attributes.
    // This refers to a segment in attribute_pool_.
    ArrayRef<const void*> attributes;
    // Registers that are retired after the execution of this kernel.
    // This refers to a segment in retired_register_pool_.
    ArrayRef<Value*> retired_regs;
  };

  // Set up the data for each kernel.
  void SetupKernelEntries();
  // Set up the registers for the function computation.
  void SetupRegisters(ArrayRef<Value*> arguments, ArrayRef<Value*> results);

  const SyncBEFFunction& func_;

  // All registers used in the function.
  SmallVector<Value*, 16> registers_;

  // Store local Values used in the computation.
  SmallVector<Value, 16> local_values_;

  // All kernel entries in the function.
  SmallVector<KernelEntry, 16> kernel_entries_;

  // Registers that are retired at each kernel.
  SmallVector<Value*, 16> retired_register_pool_;
  // Attributes used in all kernels.
  SmallVector<const void*, 16> attribute_pool_;
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
      reg = nullptr;
#endif
    } else {
      reg = &local_values_[local_value_index];
      ++local_value_index;
    }
  }

  SetupKernelEntries();
}

void BEFInterpreter::SetupKernelEntries() {
  SmallVector<int, 16> user_counts;

  auto register_infos = func_.register_infos();
  user_counts.reserve(register_infos.size());

  // Initialize the user counts for each register.
  for (auto& reg_info : register_infos) {
    user_counts.emplace_back() = reg_info.user_count;
  }

  retired_register_pool_.reserve(local_values_.size());

  // Prepare all kernel entries for this function.
  for (auto kernel_offset : func_.kernel_offsets()) {
    auto& kernel_entry = kernel_entries_.emplace_back();

    // Get the KernelEntry starting location in BEF.
    kernel_entry.kernel_start =
        func_.kernels().data() + kernel_offset / kKernelEntryAlignment;

    BEFKernel kernel(kernel_entry.kernel_start);

    // Get the kernel function.
    kernel_entry.kernel_fn =
        func_.bef_file()->GetSyncKernel(kernel.kernel_code());
    assert(kernel_entry.kernel_fn != nullptr);

    int retired_reg_start = retired_register_pool_.size();

    // Collect retired registers from arguments.
    auto arguments = kernel.GetArguments();
    for (auto reg_idx : arguments) {
      auto& user_count = user_counts[reg_idx];

      --user_count;
      assert(user_count >= 0);
      if (user_count == 0) {
        auto* value = registers_[reg_idx];
        assert(value);
        retired_register_pool_.emplace_back(value);
      }
    }

    // Collect retired registers from results.
    auto results = kernel.GetResults();
    for (auto reg_index : results) {
      // If there is no use for the result, mark it as retired.
      if (user_counts[reg_index] == 0) {
        auto* value = registers_[reg_index];
        assert(value);
        retired_register_pool_.emplace_back(value);
      }
    }

    // Set the retired registers for this kernel.
    kernel_entry.retired_regs =
        llvm::makeArrayRef(retired_register_pool_.begin() + retired_reg_start,
                           retired_register_pool_.end());

    // Collect the attributes
    int attribute_start = attribute_pool_.size();
    auto attributes = kernel.GetAttributes();
    for (auto attribute_offset : attributes) {
      // We pass the pointer here because this attribute could be an array of
      // size 0.
      attribute_pool_.emplace_back(func_.bef_file()->attribute_section_.data() +
                                   attribute_offset);
    }

    // Collect the function attributes.
    auto functions = kernel.GetFunctions();
    for (auto fn_idx : functions) {
      // Functions are passed as their corresponding `Function`.
      attribute_pool_.emplace_back(func_.bef_file()->functions_[fn_idx].get());
    }

    // Set the attributes for this kernel.
    kernel_entry.attributes = llvm::makeArrayRef(
        attribute_pool_.begin() + attribute_start, attribute_pool_.end());
  }
}

void BEFInterpreter::SetupRegisters(ArrayRef<Value*> arguments,
                                    ArrayRef<Value*> results) {
  // Set up argument Value
  for (size_t i = 0; i < arguments.size(); ++i) {
    assert(func_.register_infos()[i].is_arg_or_result);
    assert(!registers_[i]);
    registers_[i] = arguments[i];
  }

  // Set up result Value
  auto result_index = 0;
  for (auto reg_idx : func_.result_regs()) {
    assert(func_.register_infos()[reg_idx].is_arg_or_result);
    assert(!registers_[reg_idx]);

    registers_[reg_idx] = results[result_index];
    ++result_index;
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

  SyncKernelFrameBuilder kernel_frame(registers_, exec_ctx);

  // Walk through each kernel entry and invoke each kernel sequentially.
  for (auto& kernel_entry : kernel_entries_) {
    BEFKernel kernel(kernel_entry.kernel_start);

    kernel_frame.SetArguments(kernel.GetArguments());
    kernel_frame.SetAttributes(kernel_entry.attributes);
    kernel_frame.SetResults(kernel.GetResults());

    kernel_entry.kernel_fn(&kernel_frame);

    // Free values that are no longer needed.
    for (auto value : kernel_entry.retired_regs) {
      value->reset();
    }

    // Check for error.
    if (auto error = kernel_frame.TakeError()) {
      return error;
    }
  }

#ifndef NDEBUG
  // In debug mode, reset all the argument and result registers to make
  // debugging easier.
  for (size_t i = 0; i < arguments.size(); ++i) {
    registers_[i] = nullptr;
  }

  for (auto reg_idx : func_.result_regs()) {
    registers_[reg_idx] = nullptr;
  }

  // Check all local values are freed.
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
