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

//===- bef_executor.cc ----------------------------------------------------===//
//
// This file implements the Executor for BEF files.
//
//===----------------------------------------------------------------------===//

#include <atomic>
#include <cstdint>
#include <cstdio>

#include "bef_file_impl.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "tfrt/host_context/async_value.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/kernel_frame.h"
#include "tfrt/host_context/location.h"
#include "tfrt/support/bef_encoding.h"
#include "tfrt/support/bef_reader.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/ref_count.h"
#include "tfrt/tracing/tracing.h"

#ifdef DEBUG_BEF_EXECUTOR
#define DEBUG_PRINT(...) fprintf(stderr, __VA_ARGS__)
#else
#define DEBUG_PRINT(...)
#endif

namespace tfrt {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

namespace {

AsyncValue* GetRegisterValue(const BEFFileImpl::RegisterInfo& reg) {
  return reg.value.load(std::memory_order_acquire);
}

AsyncValue* GetOrCreateRegisterValue(BEFFileImpl::RegisterInfo* reg,
                                     HostContext* host) {
  // In the normal case, just load the pointer and return it.
  AsyncValue* value = reg->value.load(std::memory_order_acquire);
  if (value) return value;

  // If it doesn't exist, we create an IndirectAsyncValue for this.  We have to
  // be a bit careful though because a concurrent task could swap in the actual
  // result while we're working on this.
  auto* indirect_value = host->MakeIndirectAsyncValue().release();

  AsyncValue* existing = nullptr;
  // Speculatively set refcount in the expectation that compare_exchange
  // succeeds (see b/142802684). Specifically:
  // Add user_count refs to indirect_value. Corresponding DropRefs will occur
  // as it's used. indirect_value starts with 1 reference, and setting this
  // register will count as an additional use (+1), so add user_count refs,
  // bringing its refcount to (user_count + 1).
  indirect_value->AddRef(reg->user_count);
  if (!reg->value.compare_exchange_strong(existing, indirect_value,
                                          std::memory_order_release,
                                          std::memory_order_acquire)) {
    // If result_reg already got a result, then we don't need the
    // IndirectAsyncValue after all. Decrease refcount back to 0.
    indirect_value->DropRef(reg->user_count + 1);
    return existing;
  } else {
    return indirect_value;
  }
}

// This method makes kernels with error input immediately ready for processing
// by setting their arguments_not_ready count to 1. This allows faster error
// propagation than having these kernels wait for all inputs to be available.
// And it also saves memory by reducing lifetime of error values.
//
// Because this is a slow path that is run only when input value has
// error, we want it out of line.
LLVM_ATTRIBUTE_NOINLINE
void SetKernelsWithErrorInputReady(
    MutableArrayRef<BEFFileImpl::KernelInfo> kernel_infos,
    ArrayRef<uint32_t> kernels_with_error_input) {
  for (auto kernel_id : kernels_with_error_input) {
    auto& arguments_not_ready = kernel_infos[kernel_id].arguments_not_ready;
    int not_ready_count = arguments_not_ready.load(std::memory_order_acquire);
    while (not_ready_count > 1) {
      if (arguments_not_ready.compare_exchange_weak(not_ready_count, 1,
                                                    std::memory_order_release,
                                                    std::memory_order_acquire))
        break;
    }
  }
}

AsyncValue* SetRegisterValue(BEFFileImpl::RegisterInfo* reg,
                             AsyncValue* new_value,
                             bool* register_already_set) {
  assert(reg->user_count > 0 &&
         "No need to set register value if it is not being used by anyone.");
  // Atomically set reg->value to new_value.
  AsyncValue* existing = nullptr;
  // Speculatively set refcount in the expectation that compare_exchange
  // succeeds (see b/142802684). Specifically:
  // Add user_count refs to new_value. Corresponding DropRefs will occur as
  // it's used. new_value already has +1 reference. So add (user_count - 1)
  // more refs, bringing its effective refcount to +(user_count).
  //
  // Setting a register counts as an additional use (+1), but we are setting
  // the register right now (-1), so we can skip that AddRef/DropRef pair.
  new_value->AddRef(reg->user_count - 1);
  if (!reg->value.compare_exchange_strong(existing, new_value,
                                          std::memory_order_release,
                                          std::memory_order_acquire)) {
    // If there was already a value in it, it must be a IndirectAsyncValue. We
    // set the IndirectAsyncValue to point to the result.
    auto indirect_value = cast<IndirectAsyncValue>(existing);

    // Speculative AddRef above proved unneeded, so revert it.
    new_value->DropRef(reg->user_count - 1);

    // Give our +1 reference to 'new_value' to the indirect_value, since we are
    // not storing it in our register file.
    indirect_value->ForwardTo(TakeRef(new_value));

    // Setting a register counts as an additional use. Signal caller to DropRef
    // after the caller finished using the returned pointer.
    *register_already_set = true;

    return existing;
  }

  *register_already_set = false;
  return new_value;
}

}  // namespace

class BEFLocationHandler final : public LocationHandler,
                                 public ReferenceCounted<BEFLocationHandler> {
 public:
  BEFLocationHandler(HostContext* host, BEFFileImpl* bef_file)
      : host_{host}, bef_file_(FormRef(bef_file)) {}

  void Destroy() { host_->Destruct(this); }

  DecodedLocation DecodeLocation(Location loc) const override {
    return bef_file_->DecodeLocation(loc.data);
  }

  BEFFileImpl* BefFile() const { return bef_file_.get(); }

 private:
  friend class ReferenceCounted<BEFLocationHandler>;
  HostContext* host_;
  RCReference<BEFFileImpl> bef_file_;
};

/// A BEFExecutor runs a BEF function containing a stream of asynchronous
/// kernels. Multiple executors can be active at one time, e.g. due to
/// concurrent control flow constructs.
class BEFExecutor final : public ReferenceCounted<BEFExecutor> {
 public:
  static void Execute(ExecutionContext exec_ctx, const BEFFunction& fn,
                      ArrayRef<AsyncValue*> arguments,
                      MutableArrayRef<RCReference<AsyncValue>> results);

  /// When the last reference to the BEFExecutor is dropped, we deallocate
  /// ourself.  The memory for this class is managed through the HostAllocator
  /// managed by the HostContext.
  void Destroy() {
    auto host = this->GetHost();
    this->~BEFExecutor();
    host->Deallocate<BEFExecutor>(this);
  }

 private:
  BEFExecutor(ExecutionContext exec_ctx, BEFFileImpl* bef_file);
  ~BEFExecutor();

  void Execute(bool has_arguments_pseudo_kernel);

 private:
  void DecrementArgumentsNotReadyCounts(SmallVectorImpl<unsigned>* kernel_ids);
  void ProcessArgumentsPseudoKernel(SmallVectorImpl<unsigned>* kernel_ids);
  void ProcessUsedBys(const BEFKernel& kernel, int kernel_id, int result_number,
                      AsyncValue* result, int* entry_offset,
                      SmallVectorImpl<unsigned>* kernel_ids);
  void MaybeAddRefForResult(AsyncValue* result);
  HostContext* GetHost() const { return exec_ctx_.host(); }
  BEFFileImpl* BefFile() const { return location_handler_->BefFile(); }

  ArrayRef<uint32_t> kernels() { return function_info_.kernels; }

  MutableArrayRef<BEFFileImpl::RegisterInfo> register_infos() {
    return function_info_.register_infos.mutable_array();
  }

  MutableArrayRef<BEFFileImpl::KernelInfo> kernel_infos() {
    return function_info_.kernel_infos.mutable_array();
  }

 private:
  friend class ReferenceCounted<BEFExecutor>;

  /// The execution context for this BEFExecutor.
  ExecutionContext exec_ctx_;

  /// Decoded BEFFunction
  BEFFileImpl::FunctionInfo function_info_;

  /// Make sure location handler is alive as long as there is pending execution.
  RCReference<BEFLocationHandler> location_handler_;
};

//===----------------------------------------------------------------------===//
// Core executor logic
//===----------------------------------------------------------------------===//

// Enqueue the users of the result for later processing. If a result has no
// users, it will be skipped. If the kernel immediately completed a result, then
// we can mark all kernels using it as ready to go, otherwise we need to enqueue
// them on their unavailable operands.
void BEFExecutor::ProcessUsedBys(const BEFKernel& kernel, int kernel_id,
                                 int result_number, AsyncValue* result,
                                 int* entry_offset,
                                 SmallVectorImpl<unsigned>* kernel_ids) {
  // Find used_by entries for this result.
  auto num_used_bys = kernel.num_used_bys(result_number);
  // Skip current result if there is no user.
  if (num_used_bys == 0) {
    MaybeAddRefForResult(result);
    return;
  }

  auto used_bys = kernel.GetKernelEntries(*entry_offset, num_used_bys);
  // Move entry offset to used_bys for next result.
  *entry_offset += num_used_bys;

  assert(!used_bys.empty());

  auto state = result->state();

  // If this result has error, then we can accelerate error propagation by
  // making any using kernel ready.
  //
  // This check is done intentionally after checking for IsConcrete()
  // so that in the normal path we call AsyncValue::state() only once.
  if (state.IsError()) {
#ifdef DEBUG_BEF_EXECUTOR
    if (kernel_id >= 0) {
      std::string error_message;
      llvm::raw_string_ostream os(error_message);
      os << result->GetError();
      DEBUG_PRINT(
          "Kernel %d %s got error: %s\n", kernel_id,
          location_handler_->BefFile()->GetKernelName(kernel.kernel_code()),
          os.str().c_str());
    }
#endif
    SetKernelsWithErrorInputReady(kernel_infos(), used_bys);
  }

  // If this result is already available (because the kernel produced its
  // result synchronously, or because the worker thread beat our thread)
  // then we can immediately process any using kernel as part of our visit
  // here. Just add it to the worklist for processing, to avoid recursion.
  if (state.IsAvailable()) {
    kernel_ids->append(used_bys.begin(), used_bys.end());
    return;
  }

  // Otherwise, the kernel is going to produce its result asynchronously -
  // we process the user whenever the value becomes available.

  // Keep this executor alive until the kernel runs.
  AddRef();

  // We could just add one "AndThen" closure for each use, which would be
  // fine in the case where results have one use, but it would result in
  // needless work for cases where there are multiple users of the same
  // result.
  //
  // It is better to make an SmallVector of things to update when the
  // result becomes available, but that isn't great because it makes the
  // capture list too large for the inline representation of an
  // std::function - forcing an allocation in the path of the common case.
  // Handle this by just handling the two cases explicitly.
  if (used_bys.size() == 1) {
    // Single result case - build the SmallVector inside of the closure to
    // reduce the size of the capture list.

    auto used_by = used_bys.front();

    result->AndThen([this, used_by]() {
      // When the result becomes available, we process the using kernel.
      SmallVector<unsigned, 4> using_kernel_id;
      using_kernel_id.push_back(used_by);
      this->DecrementArgumentsNotReadyCounts(&using_kernel_id);
      this->DropRef();
    });
    return;
  }

  // Otherwise, build a list of values outside of the capture list and
  // process it in one go.
  SmallVector<unsigned, 8> using_kernel_ids;
  // As in BEFExecutor's constructor, we reserve some extra space to
  // accommodate growth for users of results of these kernels.
  using_kernel_ids.reserve(used_bys.size() + 4);
  using_kernel_ids.append(used_bys.begin(), used_bys.end());

  // Process the whole batch when this result becomes available.
  result->AndThen(
      [this, using_kernel_ids = std::move(using_kernel_ids)]() mutable {
        this->DecrementArgumentsNotReadyCounts(&using_kernel_ids);
        this->DropRef();
      });
}

// Process the arguments pseudo kernel and enqueue the users of these arguments.
void BEFExecutor::ProcessArgumentsPseudoKernel(
    SmallVectorImpl<unsigned>* kernel_ids) {
  assert(!kernel_ids->empty());
  assert(kernel_ids->back() == 0);
  // Remove the first kernel that is argument pseudo kernel.
  kernel_ids->pop_back();

  BEFKernel kernel(kernels().data());

  assert(kernel.num_arguments() == 0);
  assert(kernel.num_attributes() == 0);
  assert(kernel.num_functions() == 0);
  assert(kernel.num_results() != 0);

  MutableArrayRef<BEFFileImpl::RegisterInfo> register_array = register_infos();

  // The kernel body of argument pseudo kernel contains only results and
  // used_bys.
  auto results = kernel.GetKernelEntries(0, kernel.num_results());
  // Move offset to the start of used_bys.
  int used_by_offset = results.size();
  for (int result_number = 0; result_number < results.size(); ++result_number) {
    auto& result_register = register_array[results[result_number]];
    // TODO(chky): mlir_to_bef should not emit used args.
    if (result_register.user_count == 0) continue;

    AsyncValue* result = GetRegisterValue(result_register);
    assert(result && "Argument AsyncValue is not set.");

    // Process users of this result.
    ProcessUsedBys(kernel, /*kernel_id=*/-1, result_number, result,
                   &used_by_offset, kernel_ids);
  }
}

// Extends the lifetime of location_handler_ as long as there are unavailable
// results. This is to ensure location_handler_ remains valid in asynchronous
// kernels. This approach works because reporting an error requires at least one
// unavailable result.
void BEFExecutor::MaybeAddRefForResult(AsyncValue* result) {
  if (!result->IsAvailable()) {
    result->AndThen([handler = location_handler_.CopyRef()]() {});
  }
}

/// Decrement arguments_not_ready counters for the specified kernels by one,
/// executing them if they are now ready to run. This processes the kernels
/// from the end of the vector to the start - worklist style.
void BEFExecutor::DecrementArgumentsNotReadyCounts(
    SmallVectorImpl<unsigned>* kernel_ids) {
  KernelFrameBuilder kernel_frame(exec_ctx_);
  kernel_frame.SetAttributeSection(BefFile()->attribute_section_);

  MutableArrayRef<BEFFileImpl::KernelInfo> kernel_array = kernel_infos();
  MutableArrayRef<BEFFileImpl::RegisterInfo> register_array = register_infos();

  while (!kernel_ids->empty()) {
    auto kernel_id = kernel_ids->pop_back_val();
    assert(kernel_id < kernel_array.size() && "invalid kernel ID");

    // Decrement the count and see if we're ready to run.  If not, then we're
    // done with the kernel.
    if (kernel_array[kernel_id].arguments_not_ready.fetch_sub(1) != 1) continue;

    assert(kernel_array[kernel_id].offset % kKernelEntryAlignment == 0);
    BEFKernel kernel(kernels().data() +
                     kernel_array[kernel_id].offset / kKernelEntryAlignment);

    // Keep track of whether we saw any error arguments. If so, we propagate the
    // error to the results automatically. Initialize it with the cancel async
    // value if the execution has been canceled.
    AsyncValue* any_error_argument = exec_ctx_.GetCancelAsyncValue();

    // Process the kernel record to get information about what argument
    // registers, result registers, and attributes should be passed.
    kernel_frame.Reset();

    // Find the kernel implementation of this kernel.
    AsyncKernelImplementation kernel_fn =
        BefFile()->GetAsyncKernel(kernel.kernel_code());

    // Check the low bit of special_metadata, which indicates if the kernel
    // is non-strict.
    bool is_nonstrict_kernel =
        static_cast<bool>(kernel.special_metadata() &
                          static_cast<uint32_t>(SpecialAttribute::kNonStrict));
    DEBUG_PRINT("Run %skernel %u %s\n",
                is_nonstrict_kernel ? "non-strict " : "", kernel_id,
                BefFile()->GetKernelName(kernel.kernel_code()));

    // Set up operands.
    int entry_offset = 0;
    auto arguments =
        kernel.GetKernelEntries(entry_offset, kernel.num_arguments());
    for (auto reg_idx : arguments) {
      BEFFileImpl::RegisterInfo& reg = register_array[reg_idx];

      // The argument register may not be available if this is a non-strict
      // kernel that is starting before all operands are available. In that
      // case, we use an IndirectAsyncValue so it can be resolved later.
      AsyncValue* value = GetOrCreateRegisterValue(&reg, GetHost());
      // TODO(b/142757465): remove arguments_and_results_ vector in
      // AsyncKernelFrame.
      kernel_frame.AddArg(value);
      if (value->IsError()) any_error_argument = value;
    }

    // TODO(b/142757465): remove arguments_and_results_ vector in
    // AsyncKernelFrame.
    kernel_frame.SetNumResults(kernel.num_results());

    // Set up attributes.
    entry_offset += arguments.size();
    auto attributes =
        kernel.GetKernelEntries(entry_offset, kernel.num_attributes());
    for (auto attribute_offset : attributes) {
      // We pass the pointer here because this attribute could be an array of
      // size 0.
      kernel_frame.AddAttribute(BefFile()->attribute_section_.data() +
                                attribute_offset);
    }

    // Set up functions.
    entry_offset += attributes.size();
    auto functions =
        kernel.GetKernelEntries(entry_offset, kernel.num_functions());
    for (auto fn_idx : functions) {
      // Functions are passed as their corresponding `Function`.
      kernel_frame.AddAttribute(BefFile()->functions_[fn_idx].get());
    }

    // If all arguments are good or if the kernel is non-strict, run the
    // function.
    if (any_error_argument == nullptr || is_nonstrict_kernel) {
      // Get the location to pass down to the kernels so they can report an
      // error.
      kernel_frame.SetLocation(
          {location_handler_.get(), kernel.kernel_location()});

      // kernel_fn should populate results in kernel_frame with pointers to
      // AsyncValue before it returns.
      {
        TFRT_TRACE_KERNEL_SCOPE(BefFile()->GetKernelName(kernel.kernel_code()));
        kernel_fn(&kernel_frame);
      }
    } else {
      // Otherwise, automatically propagate errors to the result values.
      for (size_t i = 0, e = kernel_frame.GetNumResults(); i != e; ++i) {
        kernel_frame.SetResultAt(i, FormRef(any_error_argument));
      }
    }

    // Now that the kernel had a chance to look at the arguments, we're done
    // with them, so they can potentially be deallocated if this was the last
    // kernel to use them.
    for (auto* arg : kernel_frame.GetArguments()) arg->DropRef();

    // The following loop iterates over all results of the kernel. If a result
    // has no users, it will be skipped. If the kernel immediately completed a
    // result, then we can mark all kernels using it as ready to go, otherwise
    // we need to enqueue them on their unavailable operands.

    // Move entry offset to start of results.
    entry_offset += functions.size();
    auto results = kernel.GetKernelEntries(entry_offset, kernel.num_results());
    // Move entry offset to start of all used_bys.
    entry_offset += results.size();
    for (int result_number = 0; result_number < results.size();
         ++result_number) {
      auto& result_register = register_array[results[result_number]];

      // This kernel is not a pesudo kernel, assert the result register is
      // either unset or an IndirectAsyncValue.
      assert(GetRegisterValue(result_register) == nullptr ||
             GetRegisterValue(result_register)->IsUnresolvedIndirect());

      // Copy back the result AsyncValue to this result register.
      AsyncValue* result = kernel_frame.GetResultAt(result_number);
      assert(result && "Kernel did not set result AsyncValue");
      if (result_register.user_count == 0) {
        MaybeAddRefForResult(result);
        // If no one uses this result, skip storing the value in the register.
        // We must drop our +1 ref.
        result->DropRef();
        continue;
      }

      bool register_already_set;
      auto* register_value =
          SetRegisterValue(&result_register, result, &register_already_set);
      // Process users of this result.
      ProcessUsedBys(kernel, kernel_id, result_number, register_value,
                     &entry_offset, kernel_ids);

      // DropRef since we no longer need the IndirectAsyncValue in the register.
      if (register_already_set) register_value->DropRef();
    }
  }
}

//===----------------------------------------------------------------------===//
// Executor Setup
//===----------------------------------------------------------------------===//

BEFExecutor::BEFExecutor(ExecutionContext exec_ctx, BEFFileImpl* bef_file)
    : exec_ctx_(std::move(exec_ctx)),
      location_handler_(TakeRef(exec_ctx_.host()->Construct<BEFLocationHandler>(
          exec_ctx_.host(), bef_file))) {}

BEFExecutor::~BEFExecutor() {}

void BEFExecutor::Execute(bool has_arguments_pseudo_kernel) {
  MutableArrayRef<BEFFileImpl::KernelInfo> kernel_array = kernel_infos();

  // InitializeKernels initialized each KernelInfo::arguments_not_ready to one
  // plus the number of arguments. This means that as we walk the list to drop
  // the argument count, if we hit zero then it is time for us to trigger the
  // computation. This arrangement is nice because any sync or async kernel that
  // immediately produces results will immediately unblock subsequent kernels to
  // be run by the primary host thread, which results in zero thread hops, clean
  // top-down execution semantics (very cache friendly), and results in all the
  // atomics staying in that cores' cache.
  SmallVector<unsigned, 16> kernel_ids_to_visit;
  // If a kernel's result has multiple uses, DecrementArgumentsNotReadyCounts
  // pops one kernel_id and pushes multiple user kernel_ids, increasing the size
  // of kernel_ids_to_visit. We reserve some extra space to accommodate this
  // growth.
  kernel_ids_to_visit.reserve(kernel_array.size() + 4);
  // Reverse indices in kernel_ids_to_visit because
  // DecrementArgumentsNotReadyCounts processes its argument from back to front.
  for (unsigned i = 0, e = kernel_array.size(); i != e; ++i) {
    kernel_ids_to_visit.push_back(e - i - 1);
  }

  // The first kernel can be a pseudo kernel provides the arguments, which gets
  // special handling.
  if (has_arguments_pseudo_kernel) {
    ProcessArgumentsPseudoKernel(&kernel_ids_to_visit);
  }

  DecrementArgumentsNotReadyCounts(&kernel_ids_to_visit);
}

// Set RegisterInfo::value for argument registers.
static void InitializeArgumentRegisters(
    ArrayRef<AsyncValue*> arguments,
    MutableArrayRef<BEFFileImpl::RegisterInfo> register_infos) {
  for (size_t i = 0, e = register_infos.size(); i != e; ++i) {
    if (i < arguments.size()) {
      AsyncValue* value = arguments[i];
      // Add user_count refs to the arg. Corresponding DropRefs will occur as
      // this arg is used.
      value->AddRef(register_infos[i].user_count);
      register_infos[i].value = value;
    }
  }
}

void BEFExecutor::Execute(ExecutionContext exec_ctx, const BEFFunction& fn,
                          ArrayRef<AsyncValue*> arguments,
                          MutableArrayRef<RCReference<AsyncValue>> results) {
  DEBUG_PRINT("Execute function %s start\n",
              fn.name().empty() ? "(unknown)" : fn.name().str().c_str());

  BEFFileImpl* bef_file = fn.bef_file();
  assert(arguments.size() == fn.argument_types().size() &&
         "incorrect number of arguments passed to function call");
  assert(results.size() == fn.result_types().size() &&
         "incorrect number of results passed to function call");

  HostContext* host = exec_ctx.host();
  auto* exec_ptr = host->Allocate<BEFExecutor>();
  auto* exec = new (exec_ptr) BEFExecutor(std::move(exec_ctx), bef_file);

  size_t location_offset;
  SmallVector<size_t, 4> result_regs;
  bool success = bef_file->ReadFunction(fn.function_offset(), fn.result_types(),
                                        &location_offset, &exec->function_info_,
                                        &result_regs, host->allocator());
  if (!success) return;
  assert(result_regs.size() == fn.result_types().size());

  MutableArrayRef<BEFFileImpl::RegisterInfo> register_array =
      exec->register_infos();
  InitializeArgumentRegisters(arguments, register_array);

  // Kick off BEF execution starting from ready kernels.
  exec->Execute(!arguments.empty());

  // Populate the function result AsyncValues (results).
  //
  // Due to the presence of async kernels, the result registers may not contain
  // an AsyncValue yet at this point. If a result register contains an
  // AsyncValue, we use it as the result. Otherwise, we make a
  // IndirectAsyncValue as the function result and store the IndirectAsyncValue
  // in the result register. When the actual AsyncValue is available, we set the
  // IndirectAsyncValue to point to the actual value.
  for (size_t i = 0, e = results.size(); i != e; ++i) {
    assert(!results[i] && "result AsyncValue is not nullptr");
    BEFFileImpl::RegisterInfo& result_reg = register_array[result_regs[i]];
    AsyncValue* value = GetOrCreateRegisterValue(&result_reg, host);
    results[i] = TakeRef(value);
  }

  // The executor is created with a refcount of 1 to keep it alive during its
  // own execution. Now that we're done with it, drop our reference to allow it
  // to be deleted whenever the last async results become available.
  exec->DropRef();

  DEBUG_PRINT("Execute function %s end\n",
              fn.name().empty() ? "(unknown)" : fn.name().str().c_str());
}

//===----------------------------------------------------------------------===//
// BEFFunction implementation
//===----------------------------------------------------------------------===//

/// Execute a function with the specified CPU context.
void BEFFunction::Execute(
    const ExecutionContext& exec_ctx, ArrayRef<AsyncValue*> arguments,
    MutableArrayRef<RCReference<AsyncValue>> results) const {
  BEFExecutor::Execute(exec_ctx, *this, arguments, results);
}

// To keep this function alive, we have to keep the underlying BEF file alive.
void BEFFunction::AddRef() const { bef_file_->AddRef(); }

// To keep this function alive, we have to keep the underlying BEF file alive.
void BEFFunction::DropRef() const { bef_file_->DropRef(); }

}  // namespace tfrt
