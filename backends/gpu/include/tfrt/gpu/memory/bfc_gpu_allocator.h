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

// BFC GPU allocator
//
// This file defines the interface for a BFC GPU memory allocator.
#ifndef TFRT_GPU_MEMORY_BFC_GPU_ALLOCATOR_H_
#define TFRT_GPU_MEMORY_BFC_GPU_ALLOCATOR_H_

#include <map>
#include <set>
#include <unordered_map>

#include "llvm/Support/Error.h"
#include "tfrt/gpu/gpu_types.h"
#include "tfrt/gpu/wrapper/driver_wrapper.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/support/mutex.h"
#include "tfrt/support/ref_count.h"
#include "tfrt/support/thread_annotations.h"

namespace tfrt {
namespace gpu {
// A GPU memory allocator that implements a 'best-fit with coalescing'
// algorithm.  This is essentially a very simple version of Doug Lea's
// malloc (dlmalloc).
//
// The goal of this allocator is to support defragmentation via
// coalescing.  One assumption we make is that the process using this
// allocator owns pretty much all of the GPU memory, and that nearly
// all requests to allocate GPU memory go through this interface.
class BfcGpuAllocator : public gpu::GpuAllocator {
 public:
  explicit BfcGpuAllocator(const wrapper::CurrentContext& current);

  // BfcGpuAllocator does not support streams. If it is asked to allocate
  // something on a stream different from the stream of any previous
  // allocations, the allocation request will fail.
  llvm::Expected<gpu::GpuPointer> Allocate(size_t num_bytes,
                                           wrapper::Stream stream) override;

  llvm::Error Deallocate(gpu::GpuPointer pointer,
                         wrapper::Stream stream) override;

 private:
  struct Bin;

  // Chunks point to GPU memory.  Their prev/next pointers form a
  // doubly-linked list of addresses sorted by GPU base address that
  // must be contiguous.  Chunks contain information about whether
  // they are in use or whether they are free, and contain a pointer
  // to the bin they are in.
  struct Chunk {
    size_t size = 0;  // Full size of GPU buffer.

    bool in_use = false;
    void* ptr = 0;  // pointer to granted GPU subbuffer.

    // If not null, the memory referred to by 'prev' is directly
    // preceding the memory used by this chunk.  E.g., it should start
    // at 'ptr - prev->size'
    Chunk* prev = nullptr;

    // If not null, the memory referred to by 'next' is directly
    // following the memory used by this chunk.  E.g., it should be at
    // 'ptr + size'
    Chunk* next = nullptr;

    // What bin are we in?
    Bin* bin = nullptr;

    std::string DebugString(bool with_neighbors) const;
  };

  Chunk* AllocateNewChunk(size_t num_bytes);
  void SplitChunk(Chunk* c, size_t num_bytes) TFRT_REQUIRES(mu_);
  void Merge(Chunk* c1, Chunk* c2) TFRT_REQUIRES(mu_);
  void MaybeCoalesce(Chunk* c) TFRT_REQUIRES(mu_);

  void ReassignChunkToBin(Chunk* c);
  void RemoveChunkFromBin(Chunk* c) TFRT_REQUIRES(mu_);

  // DumpMemoryLog prints extra statistics for the bin that would
  // serve an allocation of size `num_bytes`.
  void DumpMemoryLog(size_t num_bytes);

  // A Bin is a collection of similar-sized Chunks.
  struct Bin {
    // All chunks in this bin have >= bin_size memory.
    size_t bin_size = 0;

    struct ChunkComparator {
      bool operator()(Chunk* a, Chunk* b) const { return a->size < b->size; }
    };

    // List of chunks within the bin, sorted by chunk size.
    std::multiset<Chunk*, ChunkComparator> chunks;

    explicit Bin(size_t bs) : bin_size(bs) {}

    ~Bin() {
      for (Chunk* chunk : chunks) {
        delete chunk;
      }
    }
  };

  static bool CompareFirst(const std::pair<size_t, std::unique_ptr<Bin>>& a,
                           int64_t b) {
    return a.first < b;
  }

  // Structures immutable after construction
  wrapper::Context context_;
  // The base pointer where all the GPU memory begins.
  wrapper::DeviceMemory<void> base_ptr_;
  uint64_t gpu_memory_size_ = 0;

  // Map from bin size to Bin
  // After construction, the bin map is never resized.
  std::vector<std::pair<size_t, std::unique_ptr<Bin>>> bins_;

  // Structures mutable after construction
  mutable mutex mu_;
  std::unordered_map<void*, Chunk*> ptr_to_chunk_map_ TFRT_GUARDED_BY(mu_);
  // Because we don't support multiple streams, stream_ is the stream
  // for all allocations. All allocation requests on a different stream will be
  // denied.
  // We can't easily support "stream transitioning" now because:
  //  - we need to synchronize the former stream when we transition to the new
  //  stream.
  //  - the allocator is not notified when the stream is destroyed. So, the
  //  synchronization can happen after the stream is destroyed causing
  //  segfault.
  wrapper::Stream stream_ TFRT_GUARDED_BY(mu_);
};

}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_MEMORY_BFC_GPU_ALLOCATOR_H_
