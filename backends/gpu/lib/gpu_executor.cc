// Copyright 2021 The TensorFlow Runtime Authors
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

// Implementation of gpu executor driver.
#include "tfrt/gpu/gpu_executor.h"

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>

#include "gpu_entry_point.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"
#include "tfrt/bef/bef_buffer.h"
#include "tfrt/bef_executor/bef_file.h"
#include "tfrt/gpu/gpu_types.h"
#include "tfrt/gpu/wrapper/cuda_wrapper.h"
#include "tfrt/gpu/wrapper/wrapper.h"
#include "tfrt/host_context/async_dispatch.h"
#include "tfrt/host_context/async_value.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/attribute_utils.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/function.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/kernel_registry.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/aligned_buffer.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {

static RCReference<ErrorAsyncValue> MakeErrorAsyncValueRef(llvm::Error error) {
  return MakeErrorAsyncValueRef(llvm::toString(std::move(error)));
}

namespace gpu {

// Releases the wrapped OwnedContext or OwnedStream so that it is not destroyed
// along with the AsyncValue. The `value` must hold the last reference.
template <typename T>
static void ReleaseGpuResource(AsyncValueRef<T> value) {
  assert(value.GetAsyncValue()->NumRef() == 1 && "dangling reference");
  value->release();
}

// Calls all pending callbacks in order to release internal ref-counts
// potentially held by these callbacks (e.g. from a memcpy). The context has
// already been synchronized appropriately to guarantee no user-held ref-counts.
static Error InvokeAllCallbacks(const GpuContext& context) {
  auto none_pending = context.MaybeInvokeCallbacks();
  if (!none_pending) return none_pending.takeError();
  if (*none_pending) return Error::success();
  return MakeStringError("not all pending callbacks invoked");
}

GpuContextCache::~GpuContextCache() {
  for (auto& pair : context_resources_) {
    AsyncValueRef<GpuContext> context = std::move(pair.second.first);
    LogIfError(InvokeAllCallbacks(*context));
    delete pair.second.second;
    ReleaseGpuResource(std::move(context));
  }
}

GpuContextCache::Resources GpuContextCache::GetOrCreate(
    wrapper::Context context) {
  mutex_lock lock(mutex_);
  auto pair = context_resources_.try_emplace(context);
  if (pair.second) {
    auto gpu_context =
        MakeAvailableAsyncValueRef<GpuContext>(wrapper::OwningContext(context));
    pair.first->second =
        std::make_pair(std::move(gpu_context), new ResourceContext());
  }
  return pair.first->second;
}

void BorrowedStreamDeleter::operator()(pointer ptr) {
  ReleaseGpuResource(AsyncValueRef<GpuStream>(TakeRef(ptr.value())));
}

BorrowedStream MakeBorrowedStream(AsyncValueRef<GpuContext> context,
                                  wrapper::Stream stream) {
  auto gpu_stream = MakeAvailableAsyncValueRef<GpuStream>(
      std::move(context), wrapper::OwningStream(stream));
  BorrowedStream::pointer pointer(gpu_stream.release());
  return BorrowedStream(pointer);
}

namespace {
// Owns the buffer with sufficient alignment.
class BefMemoryBuffer : public llvm::MemoryBuffer {
 public:
  explicit BefMemoryBuffer(llvm::MemoryBuffer* buffer)
      : buffer_(reinterpret_cast<const uint8_t*>(buffer->getBufferStart()),
                reinterpret_cast<const uint8_t*>(buffer->getBufferEnd())),
        identifier_(buffer->getBufferIdentifier().str()) {
    init(reinterpret_cast<const char*>(buffer_.data()),
         reinterpret_cast<const char*>(buffer_.data() + buffer_.size()),
         /*RequiresNullTerminator*/ false);
  }
  llvm::StringRef getBufferIdentifier() const override { return identifier_; }
  BufferKind getBufferKind() const override { return MemoryBuffer_Malloc; }

 private:
  BefBuffer buffer_;  // Aligned bef data.
  std::string identifier_;
};
}  // namespace

static bool HasRequiredBefAlignment(const void* ptr) {
  return reinterpret_cast<uintptr_t>(ptr) % GetRequiredBefAlignment() == 0;
}

llvm::Expected<UniqueMemoryBuffer> OpenBefBuffer(llvm::StringRef path) {
  std::string message;
  auto file = mlir::openInputFile(path, &message);
  if (!file) return MakeStringError(message);
  if (HasRequiredBefAlignment(file->getBufferStart())) return std::move(file);
  return std::make_unique<BefMemoryBuffer>(file.get());
}

static mlir::Location GetLocation(std::optional<DecodedLocation> loc,
                                  mlir::MLIRContext* context) {
  if (!loc) return mlir::UnknownLoc::get(context);
  if (loc->is<FileLineColLocation>()) {
    auto file_loc = loc->get<FileLineColLocation>();
    return mlir::FileLineColLoc::get(context, file_loc.filename, file_loc.line,
                                     file_loc.column);
  }
  if (loc->is<OpaqueLocation>()) {
    auto identifier =
        mlir::StringAttr::get(context, loc->get<OpaqueLocation>().loc);
    return mlir::NameLoc::get(identifier);
  }
  return mlir::UnknownLoc::get(context);
}

DiagHandler GetDiagHandler(mlir::MLIRContext* context) {
  return [=](const DecodedDiagnostic& diagnostic) {
    mlir::emitError(GetLocation(diagnostic.location, context))
        << "runtime error: " << diagnostic.message();
  };
}

namespace {
// Work-queue which executes tasks in the caller's thread, except for tasks that
// don't allow queuing which are executed in an unbounded thread pool.
class GpuWorkQueue : public ConcurrentWorkQueue {
  // Wraps a thread which runs 'work' and then adds itself to 'stack_' until
  // new work is pushed.
  class Thread {
   public:
    // Sentinel value for 'stack_' to block pushing or popping.
    static Thread* const kQuiesce;

    explicit Thread(std::atomic<Thread*>& stack, TaskFunction work)
        : stack_(stack),
          work_(std::move(work)),
          thread_(std::mem_fn(&Thread::Work), this) {}

    Thread(const Thread&) = delete;
    Thread& operator=(const Thread&) = delete;

    // Prerequisite: needs to be in idle state.
    ~Thread() {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = true;
      }
      cond_var_.notify_all();
      thread_.join();
    }

    // Pushes new 'work' to be executed on the thread.
    // May be called once after 'this' has been popped off the 'stack'.
    void Push(TaskFunction work) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        this->work_ = std::move(work);
      }
      cond_var_.notify_all();
    }

    // Pops top thread from the 'stack'.
    static Thread* Pop(std::atomic<Thread*>& stack) {
      Thread* thread = stack.load();
      do {
        // Return if stack is empty or blocked from popping.
        if (thread == nullptr || thread == kQuiesce) return nullptr;
        // Repeat until this thread wins the race to pop from 'stack'.
      } while (!stack.compare_exchange_weak(thread, thread->next_));
      return thread;
    }

    // Block until thread is idle and update 'next_'.
    void Wait(Thread* next) {
      std::unique_lock<std::mutex> lock(mutex_);
      cond_var_.wait(lock, [&] { return !work_; });
      next_ = next;
    }

   private:
    // The main loop of 'thread'.
    void Work() {
      std::unique_lock<std::mutex> lock(mutex_);
      do {
        work_();
        work_ = nullptr;
        // Push this instance to 'stack_', unless quiescing.
        next_ = stack_.load();
        do {
          if (next_ == kQuiesce) {
            // Notify 'Wait()', which updates next_.
            // 'Quiesce()' then pushes all threads onto the stack at once.
            cond_var_.notify_all();
            break;
          }
          // Repeat until this thread wins the race to push to 'stack_'.
        } while (!stack_.compare_exchange_weak(next_, this));
        // Enter idle state.
        cond_var_.wait(lock, [&] { return work_ || shutdown_; });
      } while (work_);
    }

    std::atomic<Thread*>& stack_;
    std::mutex mutex_;
    std::condition_variable cond_var_;
    TaskFunction work_;
    bool shutdown_ = false;
    std::thread thread_;
    Thread* next_;
  };

  void AddBlockingTaskImpl(TaskFunction work, bool allow_queuing) {
    // If queuing is allowed, execute 'work' in calling thread.
    if (allow_queuing) return work();
    // Execute 'work' in idle thread if available.
    if (Thread* thread = Thread::Pop(stack_))
      return thread->Push(std::move(work));
    // Execute 'work' in new thread.
    std::lock_guard<std::mutex> lock(mutex_);
    threads_.emplace_back(new Thread(stack_, std::move(work)));
  }

 public:
  // The instance needs to be quiesced. The HostContext takes care of this.
  ~GpuWorkQueue() override = default;

  std::string name() const override { return "GpuWorkQueue"; }

  void AddTask(TaskFunction work) override { work(); };

  std::optional<TaskFunction> AddBlockingTask(TaskFunction work,
                                              bool allow_queuing) override {
    AddBlockingTaskImpl(std::move(work), allow_queuing);
    return std::nullopt;
  }

  void Await(ArrayRef<RCReference<AsyncValue>> values) override {
    // Count values that are not yet available.
    int unavailable = llvm::count_if(
        values, [](const auto& value) { return value->IsUnavailable(); });
    if (unavailable == 0) return;

    // std::latch (C++20)
    std::atomic<int> counter(unavailable);
    bool complete = false;
    auto decrement = [&]() mutable {
      if (counter.fetch_sub(1) > 1) return;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        complete = true;
      }
      cond_var_.notify_all();
    };

    // Count down latch each time a value becomes available.
    for (const auto& value : values) {
      if (value->IsAvailable()) continue;
      value->AndThen(decrement);
    }

    // Wait for latch to reach zero.
    std::unique_lock<std::mutex> lock(mutex_);
    cond_var_.wait(lock, [&] { return complete; });
  }

  void Quiesce() override {
    // The implementation temporarily disables the stack from being pushed or
    // popped, which prevents threads from going from idle back to working
    // state. It then waits for each thread to reach the idle state. If any
    // pending work adds new blocking tasks, those will spin up new threads. The
    // process is repeated until no more threads are spun up. Threads that are
    // spun up during this process are stopped again before returning from the
    // function. Spinning up temporary threads is not efficient, but we don't
    // expect this function to be called often.
    //
    // An alternative implementation would loop to call Thread::Wait() on each
    // thread until the stack isn't pushed anymore, but detecting that case
    // would make the more common AddBlockingTask() implementation slower.
    //
    // A more optimized implementation would loop to lock all thread's mutexes
    // at once (e.g. using boost::lock()) until no more threads are spun up.
    // Correctly locking a range of loops is non-trivial to implement though.

    // Stop pushing idle threads to 'stack_' while quiescing.
    stack_.store(Thread::kQuiesce);
    std::vector<std::unique_ptr<Thread>> threads;
    size_t num_threads = std::numeric_limits<size_t>::max();
    Thread* head = nullptr;
    do {
      size_t num_idle = threads.size();
      // Temporarily lock mutex_ to move 'threads_' to local variable.
      {
        std::lock_guard<std::mutex> lock(mutex_);
        llvm::move(threads_, std::back_inserter(threads));
        threads_.resize(0);
      }
      // Wait for newly added threads to become idle.
      for (auto it = threads.begin() + num_idle; it != threads.end(); ++it) {
        (*it)->Wait(head);
        head = it->get();
      }
      // Save threads.size() during the first iteration.
      num_threads = std::min(threads.size(), num_threads);
    } while ([&] {
      std::lock_guard<std::mutex> lock(mutex_);
      // Repeat if new threads were spun up.
      if (!threads_.empty()) return true;
      // Delete threads that were spun up during the process above.
      threads.resize(num_threads);
      // Move pre-existing threads from local variable back to 'threads_'.
      threads_ = std::move(threads);
      // Restore 'stack_' by adding all threads.
      stack_.store(threads_.empty() ? nullptr : threads_.back().get());
      return false;
    }());
  }

  int GetParallelismLevel() const override { return 0; }
  bool IsInWorkerThread() const override { return true; }

 private:
  std::mutex mutex_;
  std::condition_variable cond_var_;
  // Guarded by mutex_, but without annotating so because std::lock_guard
  // has no thread annotations.
  std::vector<std::unique_ptr<Thread>> threads_;
  // Head of idle threads stack.
  std::atomic<Thread*> stack_;
};

GpuWorkQueue::Thread* const GpuWorkQueue::Thread::kQuiesce =
    reinterpret_cast<GpuWorkQueue::Thread*>(-1);
}  // namespace

std::unique_ptr<HostContext> CreateHostContext(DiagHandler diag_handler) {
  auto host_ctx =
      std::make_unique<HostContext>(diag_handler, tfrt::CreateMallocAllocator(),
                                    std::make_unique<GpuWorkQueue>());
  tfrt::RegisterStaticKernels(host_ctx->GetMutableRegistry());
  return host_ctx;
}

llvm::Expected<ExecutionContext> CreateExecutionContext(
    HostContext* host, ResourceContext* resource_ctx) {
  auto request_ctx = RequestContextBuilder(host, resource_ctx).build();
  if (!request_ctx) return request_ctx.takeError();
  return tfrt::ExecutionContext(std::move(*request_ctx));
}

llvm::Expected<EntryPoint> GetEntryPoint(const BEFFile& file,
                                         const ExecutionContext& exec_ctx) {
  const Function* function = file.GetFunction(GetEntryPointFuncName());
  if (!function) return MakeStringError(GetEntryPointFuncName(), " not found");

  RCReference<AsyncValue> result;
  function->Execute(exec_ctx, {}, result);
  tfrt::Await(result);

  if (result->IsError())
    return tfrt::MakeStringError(result->GetError().message());

  return *AsyncValueRef<EntryPoint>(std::move(result));
}

llvm::Error PreloadGpuResources(const BEFFile& file,
                                const ExecutionContext& exec_ctx,
                                AsyncValueRef<GpuContext> context) {
  const Function* function = file.GetFunction(PreloadResourcesFuncName());
  if (!function)
    return MakeStringError(PreloadResourcesFuncName(), " not found");

  RCReference<AsyncValue> result;
  function->Execute(exec_ctx, {context.GetAsyncValue()}, result);
  tfrt::Await(result);

  if (result->IsError())
    return tfrt::MakeStringError(result->GetError().message());

  return llvm::Error::success();
}

static Expected<GpuStream> CreateGpuStreamImpl(wrapper::Platform platform,
                                               int ordinal) {
  if (auto error = wrapper::Init(platform)) return std::move(error);
  auto device = wrapper::DeviceGet(platform, ordinal);
  if (!device) return device.takeError();
  auto context = wrapper::DevicePrimaryCtxRetain(*device);
  if (!context) return context.takeError();
  auto current = wrapper::CtxSetCurrent(context->get());
  if (!current) return current.takeError();
  auto stream = wrapper::StreamCreateNonBlocking(*current);
  if (!stream) return stream.takeError();
  auto gpu_ctx = MakeAvailableAsyncValueRef<GpuContext>(std::move(*context));
  return GpuStream(std::move(gpu_ctx), std::move(*stream));
}

AsyncValueRef<GpuStream> CreateGpuStream(wrapper::Platform platform,
                                         int ordinal) {
  auto stream = CreateGpuStreamImpl(platform, ordinal);
  if (!stream) return MakeErrorAsyncValueRef(stream.takeError());
  return MakeAvailableAsyncValueRef<GpuStream>(std::move(*stream));
}

// Creates a buffer containing gpu device memory of the given size.
AsyncValueRef<GpuBuffer> AllocateGpuBuffer(const GpuStream& stream,
                                           size_t size_bytes) {
  auto allocator =
      MakeAvailableAsyncValueRef<GpuDefaultAllocator>(stream.context());
  auto buffer =
      GpuBuffer::Allocate(std::move(allocator), size_bytes, stream.get());
  if (!buffer) return MakeErrorAsyncValueRef(buffer.takeError());
  return MakeAvailableAsyncValueRef<GpuBuffer>(std::move(*buffer));
}

static llvm::Error CheckFunctionSignature(const KernelRegistry& registry,
                                          const Function& function) {
  if (function.num_results() != 1)
    return MakeStringError("Expected single result");

  if (function.result_types().front() != registry.GetType("!tfrt.chain"))
    return MakeStringError("Expected result to be !tfrt.chain");

  if (function.num_arguments() < 2)
    return MakeStringError("Expected at least two arguments");

  auto arg_types = function.argument_types();
  if (arg_types[0] != registry.GetType("!tfrt.chain"))
    return MakeStringError("Expected first argument to be !tfrt.chain");
  if (arg_types[1] != registry.GetType("!tfrt_gpu.stream"))
    return MakeStringError("Expected second argument to be !tfrt_gpu.stream");
  for (size_t i = 2; i < function.num_arguments(); ++i) {
    if (arg_types[i] != registry.GetType("!tfrt_gpu.buffer"))
      return MakeStringError("Expected tail arguments to be !tfrt_gpu.buffer");
  }

  return llvm::Error::success();
}

AsyncValueRef<Chain> Execute(const ExecutionContext& exec_ctx,
                             const Function& function,
                             AsyncValueRef<Chain> chain,
                             AsyncValueRef<GpuStream> stream,
                             ArrayRef<AsyncValueRef<GpuBuffer>> buffers) {
  if (auto error = CheckFunctionSignature(exec_ctx.host()->GetKernelRegistry(),
                                          function)) {
    return MakeErrorAsyncValueRef(std::move(error));
  }

  llvm::SmallVector<AsyncValue*, 4> arguments = {chain.GetAsyncValue(),
                                                 stream.GetAsyncValue()};
  llvm::transform(buffers, std::back_inserter(arguments),
                  [](const AsyncValueRef<GpuBuffer>& buffer) {
                    return buffer.GetAsyncValue();
                  });

  if (function.num_arguments() != arguments.size())
    return MakeErrorAsyncValueRef("Incorrect number of arguments");

  RCReference<AsyncValue> result;
  function.Execute(exec_ctx, arguments, result);
  return AsyncValueRef<Chain>(std::move(result));
}

static llvm::Expected<EntryPoint> GetEntryPointKernel(
    ArrayAttribute<int64_t> buffer_sizes, StringAttribute function_name,
    Attribute<int32_t> platform, Attribute<int64_t> version) {
  if (*version != GetEntryPointVersion()) {
    return MakeStringError("Expected version ", GetEntryPointVersion(),
                           ", got ", *version);
  }
  return EntryPoint{static_cast<wrapper::Platform>(*platform),
                    function_name.str(),
                    {buffer_sizes.data().begin(), buffer_sizes.data().end()}};
}

TFRT_STATIC_KERNEL_REGISTRATION([](KernelRegistry* kernel_reg) {
  kernel_reg->AddKernel(GetEntryPointOpName(),
                        TFRT_KERNEL(GetEntryPointKernel));
});

}  // namespace gpu
}  // namespace tfrt
