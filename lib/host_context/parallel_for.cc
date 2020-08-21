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

//===- parallel_for.cc ------------------------------------------*- C++ -*-===//
//
// Parallel algorithms implementation for the HostContext.
//
//===----------------------------------------------------------------------===//

#include "tfrt/host_context/parallel_for.h"

#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/host_context.h"

namespace tfrt {

using BlockSizes = ParallelFor::BlockSizes;

//===----------------------------------------------------------------------===//
// BlockSizes configures how a range is split into blocks executed in parallel.
//===----------------------------------------------------------------------===//

BlockSizes ParallelFor::BlockSizes::Fixed(size_t n) {
  return BlockSizes([n](size_t) { return n; });
}

BlockSizes ParallelFor::BlockSizes::Min(size_t min) {
  return BlockSizes(
      [min](size_t block_size) { return std::max(min, block_size); });
}

size_t ParallelFor::BlockSizes::GetBlockSize(size_t num_worker_threads,
                                             size_t total_size) const {
  // Do not create too many small blocks.
  static constexpr size_t kMaxOversharding = 4;

  // Split input range to assign `kMaxOversharding` tasks to each worker thread.
  assert(total_size > 0 && "Illegal total size");
  size_t block_size = total_size / (kMaxOversharding * num_worker_threads);

  // Compute final block sizes using implementation function if it is specified.
  if (impl_) block_size = impl_(block_size);
  assert(block_size >= 0 && "Illegal block size");
  block_size = std::min(block_size, total_size);

  return block_size;
}

//===----------------------------------------------------------------------===//
// Parallel for algorithms.
//===----------------------------------------------------------------------===//
namespace {

// If ParallelFor will choose to execute the `compute` function asynchronously,
// it will move all the arguments into this context, and will keep it on the
// heap, until all submitted asynchronous work is completed.
class ParallelForExecutionContext {
 public:
  static ParallelForExecutionContext* Allocate(
      HostContext* host, size_t n, size_t block_size,
      llvm::unique_function<void(size_t, size_t)> compute,
      llvm::unique_function<void()> on_done) {
    return new ParallelForExecutionContext(
        host, n, block_size, std::move(compute), std::move(on_done));
  }

  // EvalBlocks() recursively splits the assigned block range and enqueues work
  // to the HostContext. This improves latency, by removing a sequential step
  // from the caller thread. After enqueueing work to the host context, it
  // evaluates a single block in the caller thread. Blocks to evaluate are
  // specified by the half-open interval [start_block, end_block).
  void EvalBlocks(size_t start_block, size_t end_block) {
    while (end_block - start_block > 1) {
      const size_t mid_block = start_block + (end_block - start_block) / 2;

      // Evaluate [mid_block, end_block) blocks.
      host_->EnqueueWork(
          [this, mid_block, end_block]() { EvalBlocks(mid_block, end_block); });

      // Current range becomes [start_block, mid_block).
      end_block = mid_block;
    }

    assert(end_block - start_block == 1);

    // Call `compute` for a single block.
    compute_(start_block * block_size_, std::min(n_, end_block * block_size_));

    // Delete this context if it was the last block.
    if (pending_blocks_.fetch_sub(1) == 1) delete this;
  }

  int PendingBlocks() { return pending_blocks_; }

 private:
  ParallelForExecutionContext(
      HostContext* host, size_t n, size_t block_size,
      llvm::unique_function<void(size_t, size_t)> compute,
      llvm::unique_function<void()> on_done)
      : host_(host),
        n_(n),
        block_size_(block_size),
        pending_blocks_(DivUp(n, block_size)),
        compute_(std::move(compute)),
        on_done_(std::move(on_done)) {}

  ~ParallelForExecutionContext() { on_done_(); }

  // Faster equivalent of `std::ceil((float) x / (float) y)`.
  static size_t DivUp(const size_t x, const size_t y) {
    assert(y > 0);
    return (x + y - 1) / y;
  }

  HostContext* host_;  // must stay alive before the `on_done` is called

  size_t n_;
  size_t block_size_;
  std::atomic<size_t> pending_blocks_;

  llvm::unique_function<void(size_t, size_t)> compute_;
  llvm::unique_function<void()> on_done_;
};

}  // namespace

void ParallelFor::Execute(size_t total_size, const BlockSizes& block_sizes,
                          llvm::unique_function<void(size_t, size_t)> compute,
                          llvm::unique_function<void()> on_done) const {
  // Immediately call `on_done` if nothing to execute.
  if (total_size == 0) return on_done();

  // Compute a parallel block size for the non-empty range [0, total_size).
  const size_t block_size =
      block_sizes.GetBlockSize(host_->GetNumWorkerThreads(), total_size);
  assert(block_size > 0 && "Illegal block size");
  assert(block_size <= total_size && "Illegal block size");

  // Execute single block in the caller thread.
  if (total_size <= block_size) {
    compute(0, total_size);
    on_done();
    return;
  }

  // Allocate parallel for execution context on the heap.
  ParallelForExecutionContext* ctx = ParallelForExecutionContext::Allocate(
      host_, total_size, block_size, std::move(compute), std::move(on_done));
  ctx->EvalBlocks(0, ctx->PendingBlocks());
}

AsyncValueRef<Chain> ParallelFor::Execute(
    size_t total_size, const BlockSizes& block_sizes,
    llvm::unique_function<void(size_t, size_t)> compute) const {
  auto chain = MakeConstructedAsyncValueRef<Chain>(host_);
  Execute(total_size, block_sizes, std::move(compute),
          [chain = chain.CopyRef()]() { chain.SetStateConcrete(); });
  return chain;
}

}  // namespace tfrt
