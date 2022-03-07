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

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>

#include "bef_file_impl.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "tfrt/bef/bef_encoding.h"
#include "tfrt/bef/bef_reader.h"
#include "tfrt/host_context/async_dispatch.h"
#include "tfrt/host_context/async_value.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/kernel_frame.h"
#include "tfrt/host_context/location.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/ref_count.h"
#include "tfrt/tracing/tracing.h"

#ifdef TFRT_BEF_EXECUTOR_DEBUG
#define DEBUG_PRINT(...) fprintf(stderr, __VA_ARGS__)
#else
#define DEBUG_PRINT(...)
#endif

namespace tfrt {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

namespace {

// Take one reference to `new_value` and set it in the register. The final
// AsyncValue inside this register may be different from `new_value` in case
// that there is an existing indirect async value.
LLVM_ATTRIBUTE_ALWAYS_INLINE void SetRegisterValue(
    BEFFileImpl::RegisterInfo* reg, RCReference<AsyncValue> result) {
  assert(reg->user_count > 0 &&
         "No need to set register value if it is not being used by anyone.");

  if (reg->value) {
    // If the register already has a value, it must be a return result that is
    // an indirect async value.
    auto* indirect_value = cast<IndirectAsyncValue>(reg->value);
    // Move one reference to the indirect value. Though a register might be used
    // as multiple return results, `reg->user_count` will only include one
    // reference for all return results.
    indirect_value->ForwardTo(std::move(result));
    // Drop the reference of this indirect async value as it is no longer needed
    // in this function.
    indirect_value->DropRef();
  } else {
    auto* raw = result.release();
    // Note that `result` already has +1 reference. So add (user_count - 1) more
    // refs, bringing its effective refcount to +(user_count).
    raw->AddRef(reg->user_count - 1);
    // Set the register value for other kernels to use.
    reg->value = raw;
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

  // Constructs a queue using `kernel_ids`, all kernels of which belong to the
  // same stream with `stream_id`.
  ReadyKernelQueue(int stream_id,
                   MutableArrayRef<BEFFileImpl::KernelInfo> kernel_array,
                   std::vector<unsigned> kernel_ids)
      : stream_id_(stream_id),
        kernel_array_(kernel_array),
        inline_kernel_ids_(std::move(kernel_ids)) {}

  // If the inline kernels are empty, we can move some of the outline kernels
  // into the inline kernels, and update the stream id. This allows to reduce
  // the number of enqueued tasks to process outline kernels.
  void SwitchStreamId() {
    assert(inline_kernel_ids_.empty() && "inlined kernels must be empty");

    // We can't switch stream id if we do not have outline kernels.
    if (outline_kernel_ids_.empty()) return;

    // Pick the new stream id from the ready outline kernels.
    stream_id_ = kernel_array_[outline_kernel_ids_[0]].stream_id;

    // Partition outlined kernels using the new stream id.
    auto inline_kernels_begin = std::partition(
        outline_kernel_ids_.begin(), outline_kernel_ids_.end(),
        [&](unsigned id) { return kernel_array_[id].stream_id != stream_id_; });

    // Move outline kernels belonging to the new stream into the inline kernels.
    inline_kernel_ids_.assign(inline_kernels_begin, outline_kernel_ids_.end());
    outline_kernel_ids_.erase(inline_kernels_begin, outline_kernel_ids_.end());
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
      // `arguments_not_ready` must be a postive number, so if it equals 1, then
      // this is the last producer kernel touching the consumer kernel, and we
      // don't need to perform the expensive fetch_sub for this case.
      auto& ready_count = kernel_info.arguments_not_ready;
      assert(ready_count.load() > 0);
      if (ready_count.load(std::memory_order_acquire) == 1 ||
          ready_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
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

  static void ExecuteAsync(ExecutionContext exec_ctx, const BEFFunction& fn,
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

  void Execute(ArrayRef<AsyncValue*> arguments);

 private:
  // Create BEFExecutor by setting up arguments and results in the register.
  // `results` will be populated with unavailable AsyncValues that are served as
  // futures (i.e. emplace() or SetError must not be called on these async
  // values).
  static RCReference<BEFExecutor> Create(
      ExecutionContext exec_ctx, const BEFFunction& fn,
      ArrayRef<AsyncValue*> arguments,
      MutableArrayRef<RCReference<AsyncValue>> results);

  // Iteratively process ready kernels in `ready_kernel_queue` and inserts ready
  // users back for next round of processing, until there are no more ready
  // kernels.
  void ProcessReadyKernels(ReadyKernelQueue& ready_kernel_queue);

  // Process the first pseudo kernel and populate its ready users in
  // `ready_kernel_queue`.
  void ProcessArgumentsPseudoKernel(ArrayRef<AsyncValue*> arguments,
                                    ReadyKernelQueue& ready_kernel_queue);

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

  // Enqueue `kernel_ids` to the concurrent work queue so that they can be
  // executed in a dfferent thread in parallel.
  void EnqueueReadyKernels(std::vector<unsigned>& kernel_ids);

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

// Process the arguments pseudo kernel and enqueue the ready users of these
// arguments to `ready_kernel_queue`. For non-ready users (eg. the function
// argument is unavailable), it sets up AndThen() callback to call
// ProcessReadyKernels() when the result is ready.
void BEFExecutor::ProcessArgumentsPseudoKernel(
    ArrayRef<AsyncValue*> arguments, ReadyKernelQueue& ready_kernel_queue) {
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

  assert(arguments.size() + 1 == results.size());
  for (int argument_number = 0, result_number = 1;
       result_number < results.size(); ++argument_number, ++result_number) {
    auto& result_register = register_array[results[result_number]];

    // Skip setting register if there is no use.
    if (result_register.user_count == 0) continue;

    auto used_bys = GetNextUsedBys(kernel, result_number, &used_by_offset);

    // Process users of this result.
    ProcessUsedBysAndSetRegister(used_bys, ready_kernel_queue,
                                 FormRef(arguments[argument_number]),
                                 &result_register);
  }
}

void BEFExecutor::DebugPrintError(const BEFKernel& kernel, unsigned kernel_id,
                                  AsyncValue* result) {
#ifdef TFRT_BEF_EXECUTOR_DEBUG
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

  DEBUG_PRINT("Run kernel %u %s\n", kernel_id,
              BefFile()->GetKernelName(kernel.kernel_code()));

  // Set up operands.
  int entry_offset = 0;
  auto arguments =
      kernel.GetKernelEntries(entry_offset, kernel.num_arguments());
  for (auto reg_idx : arguments) {
    BEFFileImpl::RegisterInfo& reg = register_array[reg_idx];

    RCReference<AsyncValue> value = TakeRef(reg.value);
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

  // If all arguments are good, run the function.
  if (any_error_argument == nullptr) {
    // Get the location to pass down to the kernels so they can report an
    // error.
    kernel_frame->SetLocation(
        {BefFile()->location_handler(), kernel.kernel_location()});

    // TODO(b/210018544): Move tracing and debugging code to kernel registration
    // so that we don't have extra bookkeeping in bef executor.
    TFRT_TRACE_SCOPE(Debug, BefFile()->GetKernelName(kernel.kernel_code()));

    // kernel_fn should populate results in kernel_frame with pointers to
    // AsyncValue before it returns.
    kernel_fn(kernel_frame);
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
    assert(result_register.value == nullptr ||
           result_register.value->IsUnresolvedIndirect());

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
    std::vector<unsigned>& kernel_ids) {
  auto kernel_array = kernel_infos();

  // Sort the kernels by streams to group them.
  std::sort(
      kernel_ids.begin(), kernel_ids.end(), [&](unsigned x_id, unsigned y_id) {
        assert(x_id < kernel_array.size());
        assert(y_id < kernel_array.size());
        return kernel_array[x_id].stream_id < kernel_array[y_id].stream_id;
      });

  // For each stream group, we enqueue the kernels to the work queue.
  for (auto iter = kernel_ids.begin(); iter != kernel_ids.end();) {
    int stream_id = kernel_array[*iter].stream_id;
    auto jter = iter++;
    for (;
         iter != kernel_ids.end() && kernel_array[*iter].stream_id == stream_id;
         ++iter) {
    }

    std::vector<unsigned> stream_kernel_ids(jter, iter);
    AddRef();
    EnqueueWork(
        exec_ctx_,
        [this, stream_id, kernel_ids = std::move(stream_kernel_ids)]() mutable {
          ReadyKernelQueue ready_kernel_queue(stream_id, kernel_infos(),
                                              std::move(kernel_ids));
          ProcessReadyKernels(ready_kernel_queue);
          DropRef();
        });
  }

  // Clear the kernel_ids as they are enqueued.
  kernel_ids.clear();
}

// Iteratively process ready kernels in `ready_kernel_queue` and inserts ready
// users back for next round of processing, until there are no more ready
// kernels.
void BEFExecutor::ProcessReadyKernels(ReadyKernelQueue& ready_kernel_queue) {
  // Process the kernel record to get information about what argument
  // registers, result registers, and attributes should be passed.
  KernelFrameBuilder kernel_frame(exec_ctx_);
  kernel_frame.SetAttributeSection(BefFile()->attribute_section_);
  kernel_frame.SetFunctions(BefFile()->functions_);

  // Switch stream id if there are no inline kernels to process.
  if (ready_kernel_queue.inline_kernel_ids().empty())
    ready_kernel_queue.SwitchStreamId();

  // Enqueue outline kernels into the concurrent work queue.
  if (!ready_kernel_queue.outline_kernel_ids().empty())
    EnqueueReadyKernels(ready_kernel_queue.outline_kernel_ids());
  assert(ready_kernel_queue.outline_kernel_ids().empty());

  // The loop below process inline kernels in a LIFO order for cache locality.
  // Outline kernels are enqueued to the concurrent work queue immediately.

  while (!ready_kernel_queue.inline_kernel_ids().empty()) {
    auto kernel_id = ready_kernel_queue.inline_kernel_ids().back();
    ready_kernel_queue.inline_kernel_ids().pop_back();

    ProcessReadyKernel(kernel_id, &kernel_frame, ready_kernel_queue);

    // Switch stream id if there are no inline kernels to process.
    if (ready_kernel_queue.inline_kernel_ids().empty())
      ready_kernel_queue.SwitchStreamId();

    // Enqueue outline kernels into the concurrent work queue.
    if (!ready_kernel_queue.outline_kernel_ids().empty())
      EnqueueReadyKernels(ready_kernel_queue.outline_kernel_ids());
    assert(ready_kernel_queue.outline_kernel_ids().empty());
  }
}

//===----------------------------------------------------------------------===//
// Executor Setup
//===----------------------------------------------------------------------===//

BEFExecutor::BEFExecutor(ExecutionContext exec_ctx, BEFFileImpl* bef_file)
    : exec_ctx_(std::move(exec_ctx)), bef_file_(FormRef(bef_file)) {}

BEFExecutor::~BEFExecutor() {}

void BEFExecutor::Execute(ArrayRef<AsyncValue*> arguments) {
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
  ProcessArgumentsPseudoKernel(arguments, ready_kernel_queue);

  // After ProcessArgumentsPseudoKernel() returns, `ready_kernel_queue` is
  // populated with available kernels. Then we start processing by calling
  // ProcessReadyKernels().
  ProcessReadyKernels(ready_kernel_queue);
}

RCReference<BEFExecutor> BEFExecutor::Create(
    ExecutionContext exec_ctx, const BEFFunction& fn,
    ArrayRef<AsyncValue*> arguments,
    MutableArrayRef<RCReference<AsyncValue>> results) {
  BEFFileImpl* bef_file = fn.bef_file();
  assert(arguments.size() == fn.argument_types().size() &&
         "incorrect number of arguments passed to function call");
  assert(results.size() == fn.result_types().size() &&
         "incorrect number of results passed to function call");

  HostContext* host = exec_ctx.host();
  auto* exec_ptr = host->Allocate<BEFExecutor>();
  auto* exec = new (exec_ptr) BEFExecutor(std::move(exec_ctx), bef_file);

  size_t location_offset;
  llvm::SmallVector<size_t, 4> result_regs;
  bool success = bef_file->ReadFunction(fn.function_offset(), fn.result_types(),
                                        &location_offset, &exec->function_info_,
                                        &result_regs, host->allocator());
  if (!success) return {};
  assert(result_regs.size() == fn.result_types().size());

  MutableArrayRef<BEFFileImpl::RegisterInfo> register_array =
      exec->register_infos();

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

    if (!result_reg.value) {
      // Create an indirect async value for return results.
      auto* indirect_value = MakeIndirectAsyncValue().release();
      // Add user_count to its refcount, which makes the total refcount
      // (user_count + 1). The user_count is for all users including tfrt.return
      // in the function. The additional +1 is to pin this async value for this
      // function, in case that the external users drop the reference before the
      // kernels in the function populates it.
      indirect_value->AddRef(result_reg.user_count);
      result_reg.value = indirect_value;
    }

    // Now that the user_count is set up correctly (either in the current
    // iteration or a previous iteration), we just need to take one reference
    // for the result.
    results[i] = TakeRef(result_reg.value);
  }

  return TakeRef(exec);
}

void BEFExecutor::Execute(ExecutionContext exec_ctx, const BEFFunction& fn,
                          ArrayRef<AsyncValue*> arguments,
                          MutableArrayRef<RCReference<AsyncValue>> results) {
  RCReference<BEFExecutor> exec =
      BEFExecutor::Create(std::move(exec_ctx), fn, arguments, results);
  if (!exec) return;

  DEBUG_PRINT("Execute function %s start\n",
              fn.name().empty() ? "(unknown)" : fn.name().str().c_str());

  // Kick off BEF execution starting from ready kernels.
  exec->Execute(arguments);

  DEBUG_PRINT("Execute function %s end\n",
              fn.name().empty() ? "(unknown)" : fn.name().str().c_str());
}

void BEFExecutor::ExecuteAsync(
    ExecutionContext exec_ctx, const BEFFunction& fn,
    ArrayRef<AsyncValue*> arguments,
    MutableArrayRef<RCReference<AsyncValue>> results) {
  RCReference<BEFExecutor> exec =
      BEFExecutor::Create(std::move(exec_ctx), fn, arguments, results);
  if (!exec) return;

  std::vector<AsyncValue*> arg_copies;
  arg_copies.reserve(arguments.size());
  for (auto* arg : arguments) {
    arg->AddRef();
    arg_copies.push_back(arg);
  }

  auto& work_queue = exec->exec_ctx_.work_queue();
  work_queue.AddTask(
      [&fn, exec = std::move(exec), arg_copies = std::move(arg_copies)]() {
        DEBUG_PRINT("Execute function %s start\n",
                    fn.name().empty() ? "(unknown)" : fn.name().str().c_str());

        // Kick off BEF execution starting from ready kernels.
        exec->Execute(arg_copies);

        for (auto* arg : arg_copies) {
          arg->DropRef();
        }

        DEBUG_PRINT("Execute function %s end\n",
                    fn.name().empty() ? "(unknown)" : fn.name().str().c_str());
        (void)fn;
      });
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

void BEFFunction::ExecuteAsync(
    const ExecutionContext& exec_ctx, ArrayRef<AsyncValue*> arguments,
    MutableArrayRef<RCReference<AsyncValue>> results) const {
  BEFExecutor::ExecuteAsync(exec_ctx, *this, arguments, results);
}

// To keep this function alive, we have to keep the underlying BEF file alive.
void BEFFunction::AddRef() const { bef_file_->AddRef(); }

// To keep this function alive, we have to keep the underlying BEF file alive.
void BEFFunction::DropRef() const { bef_file_->DropRef(); }

}  // namespace tfrt
