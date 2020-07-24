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

//===- caching_gpu_allocator.cc - CUDA CachingGpuAllocator  ---------------===//
//
// This file implements the C++ interface to CUDA caching allocator.
//
//===----------------------------------------------------------------------===//

#include "tfrt/gpu/memory/bfc_gpu_allocator.h"

#include <algorithm>
#include <cstdint>

#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "tfrt/gpu/stream/cuda_wrapper.h"
#include "tfrt/gpu/stream/stream_wrapper.h"
#include "tfrt/support/logging.h"
#include "tfrt/support/ref_count.h"
#include "tfrt/support/string_util.h"
#include "tfrt/tracing/tracing.h"

namespace tfrt {
namespace gpu {
BfcGpuAllocator::BfcGpuAllocator(const stream::CurrentContext& current)
    : context_(current.context()) {
  llvm::ExitOnError die_if_error;
  stream::MemoryInfo mem_info = die_if_error(stream::MemGetInfo(current));
  gpu_memory_size_ =
      static_cast<uint64_t>(static_cast<float>(mem_info.free_bytes) * 0.50);
  base_ptr_ = die_if_error(stream::MemAlloc(current, gpu_memory_size_));

  // Create a bunch of bins of various good sizes.

  // Covers allocations of exactly 256 bytes (the minimum size).
  bins_.push_back(std::make_pair(256, std::make_unique<Bin>(256)));

  // We create bins to fit all possible ranges that cover the
  // gpu_memory_size_ starting from allocations up to 1024 bytes to
  // allocations up to (and including) the memory limit.
  for (size_t bin_size = 1024; bin_size < gpu_memory_size_ * 2; bin_size *= 2) {
    bins_.push_back(std::make_pair(bin_size, std::make_unique<Bin>(bin_size)));
  }

  assert(std::is_sorted(bins_.begin(), bins_.end()));

  // Create one large chunk for the whole memory space that will
  // be chunked later.
  BfcGpuAllocator::Chunk* c = new BfcGpuAllocator::Chunk();
  stream::Pointer<void> p = base_ptr_.get();
  c->ptr = p.raw(p.platform());
  c->size = gpu_memory_size_;
  c->in_use = false;
  c->prev = nullptr;
  c->next = nullptr;

  ptr_to_chunk_map_.insert(std::make_pair(c->ptr, c));

  // Insert the chunk into the right bin.
  ReassignChunkToBin(c);
}

llvm::Expected<RCReference<gpu::GpuBuffer>> BfcGpuAllocator::Allocate(
    size_t num_bytes, gpu::stream::Stream stream) {
  TFRT_TRACE_SCOPE("BfcGpuAllocator::Allocate");
  // First, always allocate memory of at least 256 bytes, and always
  // allocate multiples of 256 bytes so all memory addresses are
  // nicely byte aligned.
  static_assert(
      GpuAllocator::kAlignment <= 256,
      "BfcGpuAllocator does not support alignment to more than 256 bytes");
  size_t rounded_bytes = (256 * ((num_bytes + 255) / 256));
  if (rounded_bytes == 0) {
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "Tried to allocate a size 0 buffer.");
  }

  // The BFC allocator tries to find the best fit first.
  //
  // First identify the first bin that could satisfy rounded_bytes.
  auto it =
      std::lower_bound(bins_.begin(), bins_.end(), rounded_bytes, CompareFirst);
  if (it == bins_.end()) {
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        tfrt::StrCat(" Asked for ", rounded_bytes, " but largest bin was ",
                     bins_.rbegin()->first));
  }

  mutex_lock l(mu_);
  if (stream_ == nullptr) {
    stream_ = stream;
  } else if (stream != stream_) {
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "BfcGpuAllocator does not support multiple streams");
  }
  for (; it != bins_.end(); ++it) {
    // Start searching from the first bin for the smallest chunk that fits
    // rounded_bytes.
    const std::unique_ptr<Bin>& b = it->second;
    for (BfcGpuAllocator::Chunk* chunk : b->chunks) {
      if (!chunk->in_use && chunk->size >= rounded_bytes) {
        // We found an existing chunk that fits us that wasn't in use.
        chunk->in_use = true;

        // If we can break the size of the chunk into two reasonably
        // large pieces, do so.
        //
        // TODO(vrv): What should be the criteria when deciding when
        // to split?
        if (chunk->size >= rounded_bytes * 2) {
          SplitChunk(chunk, rounded_bytes);
        }

        return TakeRef(new gpu::GpuBuffer(
            stream::Pointer<void>(chunk->ptr, stream.platform()), num_bytes,
            this));
      }
    }
  }

  // We searched all bins for an existing free chunk to use and
  // couldn't find one.  This means we must have run out of memory,
  return llvm::createStringError(
      llvm::errc::not_enough_memory,
      tfrt::StrCat("Ran out of memory trying to allocate ",
                   HumanReadableNumBytes(num_bytes)));
}

void BfcGpuAllocator::SplitChunk(BfcGpuAllocator::Chunk* c, size_t num_bytes) {
  // Create a new chunk starting num_bytes after c
  BfcGpuAllocator::Chunk* new_chunk = new BfcGpuAllocator::Chunk();
  new_chunk->ptr = static_cast<void*>(static_cast<char*>(c->ptr) + num_bytes);
  ptr_to_chunk_map_.insert(std::make_pair(new_chunk->ptr, new_chunk));

  // Set the new sizes of the chunks.
  new_chunk->size = c->size - num_bytes;
  c->size = num_bytes;

  // The new chunk is not in use.
  new_chunk->in_use = false;

  // Maintain the pointers.
  // c <-> c_neighbor becomes
  // c <-> new_chunk <-> c_neighbor
  BfcGpuAllocator::Chunk* c_neighbor = c->next;
  new_chunk->prev = c;
  new_chunk->next = c_neighbor;
  c->next = new_chunk;
  if (c_neighbor) {
    c_neighbor->prev = new_chunk;
  }

  // Maintain the bins
  ReassignChunkToBin(new_chunk);
  ReassignChunkToBin(c);
}

void BfcGpuAllocator::Deallocate(const gpu::GpuBuffer& buffer) {
  mutex_lock l(mu_);

  // Find the chunk from the ptr.
  auto ptr = GetRawPointer<void>(buffer);
  auto it = ptr_to_chunk_map_.find(ptr);
  assert(it != ptr_to_chunk_map_.end() &&
         "Asked to deallocate a pointer we never allocated");

  BfcGpuAllocator::Chunk* c = it->second;
  // Mark the chunk as no longer in use
  c->in_use = false;

  // Consider coalescing it.
  MaybeCoalesce(c);
}

llvm::Error BfcGpuAllocator::RecordUsage(const gpu::GpuBuffer& buffer,
                                         gpu::stream::Stream stream) {
  llvm_unreachable("RecordUsage is not implemented.");
}

// Merges c1 and c2 when c1->next is c2 and c2->prev is c1.
// We merge c2 into c1.
void BfcGpuAllocator::Merge(BfcGpuAllocator::Chunk* c1,
                            BfcGpuAllocator::Chunk* c2) {
  // We can only merge chunks that are not in use.
  assert(!c1->in_use && !c2->in_use);

  // c1's prev doesn't change, still points to the same ptr, and is
  // still not in use.

  // Fix up neighbor pointers
  //
  // c1 <-> c2 <-> c3 should become
  // c1 <-> c3
  BfcGpuAllocator::Chunk* c3 = c2->next;
  c1->next = c3;
  assert(c2->prev == c1);
  if (c3 != nullptr) {
    c3->prev = c1;
  }

  // Set the new size
  c1->size += c2->size;

  // Delete c2 and cleanup all state
  RemoveChunkFromBin(c2);
}

void BfcGpuAllocator::ReassignChunkToBin(BfcGpuAllocator::Chunk* c) {
  auto it = std::lower_bound(bins_.begin(), bins_.end(), c->size, CompareFirst);
  assert(it != bins_.end() && " Tried to reassign to non-existant bin");

  Bin* new_bin = it->second.get();

  // If the bin has not changed, do nothing.
  Bin* old_bin = c->bin;
  if (old_bin != nullptr && new_bin == old_bin) {
    return;
  }

  // The bin has changed.  Add the chunk to the new bin and remove
  // the chunk from the old bin.
  new_bin->chunks.insert(c);
  c->bin = new_bin;

  if (old_bin == nullptr) {
    return;
  }

  // Remove chunk from old bin
  for (auto it = old_bin->chunks.begin(); it != old_bin->chunks.end(); ++it) {
    if (*it == c) {
      old_bin->chunks.erase(it);
      return;
    }
  }
  TFRT_LOG(FATAL) << "Could not find chunk in old bin";
}

void BfcGpuAllocator::RemoveChunkFromBin(BfcGpuAllocator::Chunk* c) {
  Bin* b = c->bin;
  for (auto it = b->chunks.begin(); it != b->chunks.end(); ++it) {
    Chunk* other_c = *it;
    if (other_c->ptr == c->ptr) {
      b->chunks.erase(it);
      ptr_to_chunk_map_.erase(c->ptr);
      delete c;
      return;
    }
  }

  TFRT_LOG(FATAL) << "Could not find chunk in bin";
}

void BfcGpuAllocator::MaybeCoalesce(BfcGpuAllocator::Chunk* c) {
  // This chunk is no longer in-use, consider coalescing the chunk
  // with adjacent chunks.
  Chunk* chunk_to_reassign = nullptr;

  // If the next chunk is free, coalesce the two, if the result would
  // fit in an existing bin.
  if (c->next && !c->next->in_use) {
    chunk_to_reassign = c;

    // Deletes c->next
    Merge(c, c->next);
  }

  // If the previous chunk is free, coalesce the two
  if (c->prev && !c->prev->in_use) {
    chunk_to_reassign = c->prev;

    // Deletes c
    Merge(c->prev, c);
  }

  // Reassign the final merged chunk into the right bin.
  if (chunk_to_reassign) {
    ReassignChunkToBin(chunk_to_reassign);
  }
}

void BfcGpuAllocator::DumpMemoryLog(size_t num_bytes) {
  // For each bin: tally up the total number of chunks and bytes.
  for (const auto& bin : bins_) {
    Bin* b = bin.second.get();

    size_t total_bytes_in_use = 0;
    size_t total_bytes_in_bin = 0;
    size_t total_chunks_in_use = 0;
    size_t total_chunks_in_bin = 0;
    for (Chunk* c : b->chunks) {
      total_bytes_in_bin += c->size;
      ++total_chunks_in_bin;
      if (c->in_use) {
        total_bytes_in_use += c->size;
        ++total_chunks_in_use;
      }
    }

    TFRT_LOG(INFO) << "Bin (" << b->bin_size
                   << "): \tTotal Chunks: " << total_chunks_in_bin
                   << ", Chunks in use: " << total_chunks_in_use << " "
                   << HumanReadableNumBytes(total_bytes_in_bin)
                   << " allocated for chunks. "
                   << HumanReadableNumBytes(total_bytes_in_use)
                   << " in use in bin. ";
  }

  // Find the bin that we would have liked to allocate in, so we
  // can get some further analysis about fragmentation.
  auto it =
      std::lower_bound(bins_.begin(), bins_.end(), num_bytes, CompareFirst);
  if (it != bins_.end()) {
    Bin* b = it->second.get();

    TFRT_LOG(INFO) << "Bin for " << HumanReadableNumBytes(num_bytes) << " was "
                   << HumanReadableNumBytes(b->bin_size) << ", Chunk State: ";

    for (Chunk* c : b->chunks) {
      TFRT_LOG(INFO) << c->DebugString(/*with_neighbors=*/true);
    }
  }
}

std::string BfcGpuAllocator::Chunk::DebugString(bool with_neighbors) const {
  std::string dbg;
  StrAppend(&dbg, "  Size: ", HumanReadableNumBytes(size),
            " | in_use: ", in_use);
  if (with_neighbors && prev) {
    StrAppend(&dbg, ", prev: ", prev->DebugString(false));
  }
  if (with_neighbors && next) {
    StrAppend(&dbg, ", next: ", next->DebugString(false));
  }
  return dbg;
}

}  // namespace gpu
}  // namespace tfrt
