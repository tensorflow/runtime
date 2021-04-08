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

// This file implements the Executor for BEF files.

#include <atomic>
#include <cstdint>
#include <cstdio>

#include "bef_file_impl.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "tfrt/host_context/async_dispatch.h"
#include "tfrt/host_context/async_value.h"
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

// TSan ignores the memory synchronization ordering for the load operation if
// the comparison fails, and it leads to false positives, because it uses
// std::memory_order_release to check the ordering between previous write
// to the `REG` and loading of that value by a failed compare_exchange_strong
// operation into `EXISTING`.
#if defined(__has_feature) && __has_feature(thread_sanitizer)
#define COMPARE_EXCHANGE_STRONG(REG, EXISTING, NEW_VALUE) \
  reg->value.compare_exchange_strong(EXISTING, NEW_VALUE, \
                                     /*order=*/std::memory_order_acq_rel)
#else
#define COMPARE_EXCHANGE_STRONG(REG, EXISTING, NEW_VALUE)                   \
  reg->value.compare_exchange_strong(EXISTING, NEW_VALUE,                   \
                                     /*success=*/std::memory_order_release, \
                                     /*failure=*/std::memory_order_acquire)
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
  auto* indirect_value = MakeIndirectAsyncValue(host).release();

  AsyncValue* existing = nullptr;
  // Speculatively set refcount in the expectation that compare_exchange
  // succeeds (see b/142802684). Specifically:
  // Add user_count refs to indirect_value. Corresponding DropRefs will occur
  // as it's used. indirect_value starts with 1 reference, and setting this
  // register will count as an additional use (+1), so add user_count refs,
  // bringing its refcount to (user_count + 1).
  indirect_value->AddRef(reg->user_count);
  if (!COMPARE_EXCHANGE_STRONG(reg, existing, indirect_value)) {
    // If result_reg already got a result, then we don't need the
    // IndirectAsyncValue after all. Decrease refcount back to 0.
    indirect_value->DropRef(reg->user_count + 1);
    return existing;
  } else {
    return indirect_value;
  }
}

// Take one reference to `new_value` and set it in the register. The final
// AsyncValue inside this register may be different from `new_value` in case
// that there is an existing indirect async value.
LLVM_ATTRIBUTE_ALWAYS_INLINE void SetRegisterValue(
    BEFFileImpl::RegisterInfo* reg, RCReference<AsyncValue> result) {
  assert(reg->user_count > 0 &&
         "No need to set register value if it is not being used by anyone.");
  // Atomically set reg->value to new_value.
  AsyncValue* existing = nullptr;
  // Speculatively set refcount in the expectation that compare_exchange
  // succeeds (see b/142802684). Specifically:
  // Add user_count refs to new_value. Corresponding DropRefs will occur as
  // it's used.
  //
  // Note that new_value already has +1 reference. So add (user_count - 1)
  // more refs, bringing its effective refcount to +(user_count).
  auto* new_value = result.release();
  new_value->AddRef(reg->user_count - 1);

  if (!COMPARE_EXCHANGE_STRONG(reg, existing, new_value)) {
    // If there was already a value in it, it must be a IndirectAsyncValue. We
    // set the IndirectAsyncValue to point to the result.
    auto indirect_value = cast<IndirectAsyncValue>(existing);

    // Speculative AddRef above proved unneeded, so revert it.
    new_value->DropRef(reg->user_count - 1);

    // Give our +1 reference to 'new_value' to the indirect_value, since we are
    // not storing it in our register file.
    indirect_value->ForwardTo(TakeRef(new_value));

    // If it is an indirect async value, its refcount is created to be
    // (user_count + 1) in GetOrCreateRegisterValue(). The additional reference
    // is to ensure it is alive for the caller thread (ie. producer). Now that
    // the producer has done with this async value, we can drop the reference.
    //
    // Please Refer to async_value.md#setting-a-register-counts-as-a-use for
    // detailed explantations.
    existing->DropRef();
  }
}

llvm::ArrayRef<unsigned> GetNextUsedBys(const BEFKernel& kernel,
                                        int result_number, int* entry_offset) {
  // Find used_by entries for this result.
  auto num_used_bys = kernel.num_used_bys(result_number);
  auto used_bys = kernel.GetKernelEntries(*entry_offset, num_used_bys);
  // Move entry offset to used_bys for next result.
  *entry_offset += num_used_bys;

  return used_bys;
}

// ReadyKernelQueue is used for managing ready-to-run kernels in one sequential
// path.
class ReadyKernelQueue {
 public:
  // Constructs an empty queue with `stream_id`.
  ReadyKernelQueue(int stream_id,
                   MutableArrayRef<BEFFileImpl::KernelInfo> kernel_array)
      : stream_id_(stream_id), kernel_array_(kernel_array) {}

  // Constructs a queue using `kernel_ids`. The stream id is randomly picked
  // from the `kernel_ids`.
  ReadyKernelQueue(MutableArrayRef<BEFFileImpl::KernelInfo> kernel_array,
                   std::vector<unsigned> kernel_ids)
      : kernel_array_(kernel_array) {
    assert(!kernel_ids.empty());

    // Using the stream id of the first kernel.
    stream_id_ = kernel_array[kernel_ids[0]].stream_id;

    // Move kernels with the same stream id to `inline_kernel_ids_` and others
    // to `outline_kernel_ids_`. This recursively partitions the kernel_ids by
    // stream_id, by filtering out kernels that match stream_id on each
    // iteration, dumping the remaining kernels into outline_kernels, then later
    // creating another queue for outline_kernels, filtering out kernels that
    // match the next stream_id, etc.
    //
    // TODO(chky): Consider partitioning kernel_ids for each stream_id instead
    // of partitioning into only inline and outline groups.
    inline_kernel_ids_ = std::move(kernel_ids);
    auto outline_iter = std::partition(
        inline_kernel_ids_.begin(), inline_kernel_ids_.end(),
        [&](unsigned kernel_id) {
          assert(kernel_id < kernel_array_.size());
          return kernel_array_[kernel_id].stream_id == stream_id_;
        });
    outline_kernel_ids_.assign(outline_iter, inline_kernel_ids_.end());
    inline_kernel_ids_.resize(
        std::distance(inline_kernel_ids_.begin(), outline_iter));
  }

  // Decrement the ready counts for `kernel_ids` and put them in the queue.
  // Depending on their stream_id, they will be either put in the inline queue
  // for inline execution or outline queue for launching to a separate thread.
  LLVM_ATTRIBUTE_ALWAYS_INLINE void DecrementReadyCountAndEnqueue(
      ArrayRef<unsigned> kernel_ids) {
    // TODO(b/173798236): Consider introducing a randomization logic here in
    // mode to trigger errors in tests that relies on the implicit order.
    for (unsigned kernel_id : kernel_ids) {
      assert(kernel_id < kernel_array_.size());
      auto& kernel_info = kernel_array_[kernel_id];
      if (kernel_info.arguments_not_ready.fetch_sub(1) == 1) {
        if (kernel_info.stream_id == stream_id_) {
          inline_kernel_ids_.push_back(kernel_id);
        } else {
          outline_kernel_ids_.push_back(kernel_id);
        }
      }
    }
  }

  // `inline_kernel_ids` contains the kernels to be executed in the same thread.
  std::vector<unsigned>& inline_kernel_ids() { return inline_kernel_ids_; }

  // `outline_kernel_ids` contains the kernels to be launched to a separate
  // thread.
  std::vector<unsigned>& outline_kernel_ids() { return outline_kernel_ids_; }

  MutableArrayRef<BEFFileImpl::KernelInfo> kernel_array() const {
    return kernel_array_;
  }

  // The stream id for this sequence.
  int stream_id() const { return stream_id_; }

 private:
  int stream_id_;
  MutableArrayRef<BEFFileImpl::KernelInfo> kernel_array_;

  std::vector<unsigned> inline_kernel_ids_;
  std::vector<unsigned> outline_kernel_ids_;
};

}  // namespace

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

  void Execute();

 private:
  // Iteratively process ready kernels in `ready_kernel_queue` and inserts ready
  // users back for next round of processing, until there are no more ready
  // kernels.
  void ProcessReadyKernels(ReadyKernelQueue& ready_kernel_queue);

  // Process the first pseudo kernel and populate its ready users in
  // `ready_kernel_queue`.
  void ProcessArgumentsPseudoKernel(ReadyKernelQueue& ready_kernel_queue);

  // Process a single kernel specified by `kernel_id`, and populate the ready
  // users in `ready_kernel_queue`.
  void ProcessReadyKernel(unsigned kernel_id, KernelFrameBuilder* kernel_frame,
                          ReadyKernelQueue& ready_kernel_queue);

  // Enqueue the `users` of the `result` for later processing. If the result has
  // no users, it will be skipped. If the result is immediately available, then
  // we push them to `ready_kernel_queue`, otherwise we need to enqueue them
  // into this unavailable result. This function also publish the `result` to
  // the corresponding `result_register` so that the subscribers can use it.
  void ProcessUsedBysAndSetRegister(llvm::ArrayRef<unsigned> users,
                                    ReadyKernelQueue& ready_kernel_queue,
                                    RCReference<AsyncValue> result,
                                    BEFFileImpl::RegisterInfo* result_register);

  // Enqueue the `users` for the pseudo kernel's `result` for later processing.
  // Different from ProcessUsedBysAndSetRegister(), it does not take ownership
  // of the `result` and does not publish it to the corresponding register,
  // because the `result` is a function argument and the register has been set
  // up in BEFExecutor::Execute().
  void ProcessPseudoKernelUsedBys(llvm::ArrayRef<unsigned> users,
                                  ReadyKernelQueue& ready_kernel_queue,
                                  AsyncValue* result);

  // Enqueue `kernel_ids` to the concurrent work queue so that they can be
  // executed in a dfferent thread in parallel.
  void EnqueueReadyKernels(std::vector<unsigned> kernel_ids);

  HostContext* GetHost() const { return exec_ctx_.host(); }
  BEFFileImpl* BefFile() const { return bef_file_.get(); }

  ArrayRef<uint32_t> kernels() { return function_info_.kernels; }

  MutableArrayRef<BEFFileImpl::RegisterInfo> register_infos() {
    return function_info_.register_infos.mutable_array();
  }

  MutableArrayRef<BEFFileImpl::KernelInfo> kernel_infos() {
    return function_info_.kernel_infos.mutable_array();
  }

  void DebugPrintError(const BEFKernel& kernel, unsigned kernel_id,
                       AsyncValue* result);

 private:
  friend class ReferenceCounted<BEFExecutor>;

  /// The execution context for this BEFExecutor.
  ExecutionContext exec_ctx_;

  /// Decoded BEFFunction
  BEFFileImpl::FunctionInfo function_info_;

  RCReference<BEFFileImpl> bef_file_;
};

//===----------------------------------------------------------------------===//
// Core executor logic
//===----------------------------------------------------------------------===//

constexpr int kPseudoKernelId = 0;

// Enqueue the `users` of the `result` for later processing. If the result has
// no users, it will be skipped. If the result is immediately available, then we
// push them to `ready_kernel_queue`, otherwise we need to enqueue them into
// this unavailable result. This function also publish the `result` to the
// corresponding `result_register` so that the subscribers can use it.
LLVM_ATTRIBUTE_ALWAYS_INLINE void BEFExecutor::ProcessUsedBysAndSetRegister(
    llvm::ArrayRef<unsigned> users, ReadyKernelQueue& ready_kernel_queue,
    RCReference<AsyncValue> result,
    BEFFileImpl::RegisterInfo* result_register) {
  // If the result is available, we can set the register and schedule ready
  // users immediately.
  if (result->IsAvailable()) {
    // SetRegisterValue() must be done before DecrementReadyCountAndEnqueue()
    // because as soon as we decrement a kernel's ready count, it might be
    // executed in another thread.
    SetRegisterValue(result_register, std::move(result));
    ready_kernel_queue.DecrementReadyCountAndEnqueue(users);
    return;
  }

  // If the result is unavailable but has no users, we just need to set the
  // register which should be only used as the function result.
  if (users.empty()) {
    SetRegisterValue(result_register, std::move(result));
    return;
  }

  // Otherwise, the kernel is going to produce its result asynchronously -
  // we process the user whenever the value becomes available.

  // Keep this executor alive until the kernel runs.
  AddRef();

  // Process the whole batch when this result becomes available. Note that
  // we capture `users` which is an ArrayRef instead of copying the
  // content. This is fine because the underlying BEF file is supposed to be
  // alive when the BEF executor is alive.
  auto* result_ptr = result.get();
  result_ptr->AndThen([this, stream_id = ready_kernel_queue.stream_id(), users,
                       result_register, result = std::move(result)]() mutable {
    ReadyKernelQueue ready_kernel_queue(stream_id, kernel_infos());

    // SetRegisterValue() must be done before
    // DecrementReadyCountAndEnqueue() because as soon as we decrement a
    // kernel's ready count, it might be executed in another thread.
    SetRegisterValue(result_register, std::move(result));
    ready_kernel_queue.DecrementReadyCountAndEnqueue(users);
    this->ProcessReadyKernels(ready_kernel_queue);
    this->DropRef();
  });
}

// Enqueue the `users` for the pseudo kernel's `result` for later processing.
// Different from ProcessUsedBysAndSetRegister(), it does not take ownership of
// the `result` and does not publish it to the corresponding register, because
// the `result` is a function argument and the register has been set up in
// BEFExecutor::Execute().
//
// TODO(tfrt-devs): Consider also setting the argument register here so that we
// can unify ProcessUsedBysAndSetRegister() and ProcessPseudoKernelUsedBys().
void BEFExecutor::ProcessPseudoKernelUsedBys(
    llvm::ArrayRef<unsigned> users, ReadyKernelQueue& ready_kernel_queue,
    AsyncValue* result) {
  // If the result is available, we can schedule ready users immediately.
  if (result->IsAvailable()) {
    ready_kernel_queue.DecrementReadyCountAndEnqueue(users);
    return;
  }

  // If the result is unavailable but has users, we don't need to do anything.
  //
  // TODO(tfrt-devs): Consider eliminating unused function arguments in the
  // compiler, as it is pointless to process unused arguments in runtime.
  if (users.empty()) return;

  // Otherwise, the kernel is going to produce its result asynchronously -
  // we process the user whenever the value becomes available.

  // Keep this executor alive until the kernel runs.
  AddRef();

  // Process the whole batch when this result becomes available. Note that we
  // capture `users` which is an ArrayRef instead of copying the content. This
  // is fine because the underlying BEF file is supposed to be alive when the
  // BEF executor is alive.
  result->AndThen([this, users]() {
    ReadyKernelQueue ready_kernel_queue(
        kernel_infos()[kPseudoKernelId].stream_id, kernel_infos());
    ready_kernel_queue.DecrementReadyCountAndEnqueue(users);
    this->ProcessReadyKernels(ready_kernel_queue);
    this->DropRef();
  });
}

// Process the arguments pseudo kernel and enqueue the ready users of these
// arguments to `ready_kernel_queue`. For non-ready users (eg. the function
// argument is unavailable), it sets up AndThen() callback to call
// ProcessReadyKernels() when the result is ready.
void BEFExecutor::ProcessArgumentsPseudoKernel(
    ReadyKernelQueue& ready_kernel_queue) {
  assert(ready_kernel_queue.inline_kernel_ids().empty());
  assert(ready_kernel_queue.outline_kernel_ids().empty());

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

  // The first result is the pseudo result to trigger execution of the kernels
  // with no operands.
  assert(!results.empty());
  assert(results.front() == register_array.size());

  // Process the pseudo result first, which has no corresponding AsyncValue.
  auto used_bys = GetNextUsedBys(kernel, /*result_number=*/0, &used_by_offset);
  ready_kernel_queue.DecrementReadyCountAndEnqueue(used_bys);

  for (int result_number = 1; result_number < results.size(); ++result_number) {
    auto& result_register = register_array[results[result_number]];
    // TODO(chky): mlir_to_bef should not emit unused args.
    if (result_register.user_count == 0) continue;

    AsyncValue* result = GetRegisterValue(result_register);
    assert(result && "Argument AsyncValue is not set.");

    auto used_bys = GetNextUsedBys(kernel, result_number, &used_by_offset);

    // Process users of this result.
    ProcessPseudoKernelUsedBys(used_bys, ready_kernel_queue, result);
  }
}

void BEFExecutor::DebugPrintError(const BEFKernel& kernel, unsigned kernel_id,
                                  AsyncValue* result) {
#ifdef DEBUG_BEF_EXECUTOR
  // Print the error in debug mode.
  if (result->IsError()) {
    std::string error_message;
    llvm::raw_string_ostream os(error_message);
    os << result->GetError();
    DEBUG_PRINT("Kernel %d %s got error: %s\n", kernel_id,
                BefFile()->GetKernelName(kernel.kernel_code()),
                os.str().c_str());
  }
#endif
}

// Process the kernel for `kernel_id` and populate `ready_kernel_queue` with
// ready users.
void BEFExecutor::ProcessReadyKernel(unsigned kernel_id,
                                     KernelFrameBuilder* kernel_frame,
                                     ReadyKernelQueue& ready_kernel_queue) {
  MutableArrayRef<BEFFileImpl::RegisterInfo> register_array = register_infos();

  assert(kernel_infos()[kernel_id].offset % kKernelEntryAlignment == 0);
  BEFKernel kernel(kernels().data() +
                   kernel_infos()[kernel_id].offset / kKernelEntryAlignment);

  // Keep track of whether we saw any error arguments. If so, we propagate
  // the error to the results automatically. Initialize it with the cancel
  // async value if the execution has been canceled.
  AsyncValue* any_error_argument = exec_ctx_.GetCancelAsyncValue();

  // Find the kernel implementation of this kernel.
  AsyncKernelImplementation kernel_fn =
      BefFile()->GetAsyncKernel(kernel.kernel_code());

  // Check the low bit of special_metadata, which indicates if the kernel
  // is non-strict.
  bool is_nonstrict_kernel =
      static_cast<bool>(kernel.special_metadata() &
                        static_cast<uint32_t>(SpecialAttribute::kNonStrict));

#if !defined(TFRT_DISABLE_TRACING) || defined(DEBUG_BEF_EXECUTOR)
  const auto kernel_name = BefFile()->GetKernelName(kernel.kernel_code());
#endif

  DEBUG_PRINT("Run %skernel %u %s\n", is_nonstrict_kernel ? "non-strict " : "",
              kernel_id, kernel_name);

  // Set up operands.
  int entry_offset = 0;
  auto arguments =
      kernel.GetKernelEntries(entry_offset, kernel.num_arguments());
  for (auto reg_idx : arguments) {
    BEFFileImpl::RegisterInfo& reg = register_array[reg_idx];

    // The argument register may not be available if this is a non-strict
    // kernel that is starting before all operands are available. In that
    // case, we use an IndirectAsyncValue so it can be resolved later.
    RCReference<AsyncValue> value =
        TakeRef(GetOrCreateRegisterValue(&reg, GetHost()));
    // TODO(b/142757465): remove arguments_and_results_ vector in
    // AsyncKernelFrame.
    if (value->IsError()) any_error_argument = value.get();
    kernel_frame->AddArg(std::move(value));
  }

  // TODO(b/142757465): remove arguments_and_results_ vector in
  // AsyncKernelFrame.
  kernel_frame->SetNumResults(kernel.num_results());

  // Set up attributes.
  entry_offset += arguments.size();
  auto attributes =
      kernel.GetKernelEntries(entry_offset, kernel.num_attributes());
  kernel_frame->SetAttributes(attributes);

  // Set up functions.
  entry_offset += attributes.size();
  auto function_indices =
      kernel.GetKernelEntries(entry_offset, kernel.num_functions());
  kernel_frame->SetFunctionIndices(function_indices);

  // If all arguments are good or if the kernel is non-strict, run the
  // function.
  if (any_error_argument == nullptr || is_nonstrict_kernel) {
    // Get the location to pass down to the kernels so they can report an
    // error.
    kernel_frame->SetLocation(
        {BefFile()->location_handler(), kernel.kernel_location()});

#if !defined(TFRT_DISABLE_TRACING)
    // Pass down debug info to kernels.
    kernel_frame->SetDebugInfo({bef_file_.get(), &kernel});
#endif

    // kernel_fn should populate results in kernel_frame with pointers to
    // AsyncValue before it returns.
    {
      TFRT_TRACE_SCOPE(Debug, kernel_name);
      kernel_fn(kernel_frame);
    }
  } else {
    // Otherwise, automatically propagate errors to the result values.
    for (size_t i = 0, e = kernel_frame->GetNumResults(); i != e; ++i) {
      kernel_frame->SetResultAt(i, FormRef(any_error_argument));
    }
  }

  kernel_frame->ResetArguments();

  // The following loop iterates over all results of the kernel. If a result
  // has no users, it will be skipped. If the kernel immediately completed a
  // result, then we can mark all kernels using it as ready to go, otherwise
  // we need to enqueue them on their unavailable operands.

  // Move entry offset to start of results.
  entry_offset += function_indices.size();
  auto results = kernel.GetKernelEntries(entry_offset, kernel.num_results());
  // Move entry offset to start of all used_bys.
  entry_offset += results.size();

  for (int result_number = 0; result_number < results.size(); ++result_number) {
    auto& result_register = register_array[results[result_number]];

    // This kernel is not a pesudo kernel, assert the result register is
    // either unset or an IndirectAsyncValue.
    assert(GetRegisterValue(result_register) == nullptr ||
           GetRegisterValue(result_register)->IsUnresolvedIndirect());

    // Copy back the result AsyncValue to this result register.
    RCReference<AsyncValue> result =
        kernel_frame->ReleaseResultAt(result_number);
    assert(result && "Kernel did not set result AsyncValue");
    if (result_register.user_count == 0) {
      // If no one uses this result, skip storing the value in the register.
      // Note the reference to `result` will be dropped.
      continue;
    }

    DebugPrintError(kernel, kernel_id, result.get());

    auto used_bys = GetNextUsedBys(kernel, result_number, &entry_offset);

    // Process users of this result.
    ProcessUsedBysAndSetRegister(used_bys, ready_kernel_queue,
                                 std::move(result), &result_register);
  }
}

// Enqueue `kernel_ids` to the concurrent work queue so that they can be
// executed in a dfferent thread in parallel.
LLVM_ATTRIBUTE_NOINLINE void BEFExecutor::EnqueueReadyKernels(
    std::vector<unsigned> kernel_ids) {
  AddRef();
  EnqueueWork(exec_ctx_, [this, kernel_ids = std::move(kernel_ids)]() mutable {
    ReadyKernelQueue ready_kernel_queue(kernel_infos(), std::move(kernel_ids));
    ProcessReadyKernels(ready_kernel_queue);
    DropRef();
  });
}

// Iteratively process ready kernels in `ready_kernel_queue` and inserts ready
// users back for next round of processing, until there are no more ready
// kernels.
void BEFExecutor::ProcessReadyKernels(ReadyKernelQueue& ready_kernel_queue) {
  TFRT_TRACE_SCOPE(Verbose, "BEFExecutor::ProcessReadyKernels");

  // Process the kernel record to get information about what argument
  // registers, result registers, and attributes should be passed.
  KernelFrameBuilder kernel_frame(exec_ctx_);
  kernel_frame.SetAttributeSection(BefFile()->attribute_section_);
  kernel_frame.SetFunctions(BefFile()->functions_);

  if (!ready_kernel_queue.outline_kernel_ids().empty()) {
    EnqueueReadyKernels(std::move(ready_kernel_queue.outline_kernel_ids()));
  }
  assert(ready_kernel_queue.outline_kernel_ids().empty());

  // The loop below process inline kernels in a breadth-first order, so that
  // independent sequences can be launched as early as possible. Outline kernels
  // are enqueued to the concurrent work queue immediately.

  std::vector<unsigned> buffer;
  while (!ready_kernel_queue.inline_kernel_ids().empty()) {
    assert(buffer.empty());

    buffer.swap(ready_kernel_queue.inline_kernel_ids());
    for (unsigned kernel_id : buffer) {
      ProcessReadyKernel(kernel_id, &kernel_frame, ready_kernel_queue);

      if (!ready_kernel_queue.outline_kernel_ids().empty()) {
        EnqueueReadyKernels(std::move(ready_kernel_queue.outline_kernel_ids()));
      }
      assert(ready_kernel_queue.outline_kernel_ids().empty());
    }
    buffer.clear();
  }
}

//===----------------------------------------------------------------------===//
// Executor Setup
//===----------------------------------------------------------------------===//

BEFExecutor::BEFExecutor(ExecutionContext exec_ctx, BEFFileImpl* bef_file)
    : exec_ctx_(std::move(exec_ctx)), bef_file_(FormRef(bef_file)) {}

BEFExecutor::~BEFExecutor() {}

void BEFExecutor::Execute() {
  // Each KernelInfo::arguments_not_ready to the number of arguments (or one for
  // kernels with no arguments). This means that as we walk the list to drop the
  // argument count, if we hit zero then it is time for us to trigger the
  // computation. This arrangement is nice because any sync or async kernel that
  // immediately produces results will immediately unblock subsequent kernels to
  // be run. And some of the subsequent kernels can be run in the same thread,
  // which results in fewer thread hops, clean top-down execution semantics
  // (very cache friendly), and results in all the atomics staying in that
  // cores' cache, if these benefits outweigh the latency improvement from
  // launching these kernels in different threads.
  ReadyKernelQueue ready_kernel_queue(kernel_infos()[kPseudoKernelId].stream_id,
                                      kernel_infos());

  // The first kernel (kernel_id == 0) is a pseudo kernel that provides the
  // arguments, which gets special handling.
  ProcessArgumentsPseudoKernel(ready_kernel_queue);

  // After ProcessArgumentsPseudoKernel() returns, `ready_kernel_queue` is
  // populated with available kernels. Then we start processing by calling
  // ProcessReadyKernels().
  ProcessReadyKernels(ready_kernel_queue);
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
  exec->Execute();

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
