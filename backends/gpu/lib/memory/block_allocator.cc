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

//===- block_allocator.cc - BlockAllocator ---------------------*- C++ -*--===//
//
// This file implements the C++ interface to CUDA caching allocator.

#include "tfrt/gpu/memory/block_allocator.h"

#include <llvm/Support/Errc.h>

#include <cstdint>

namespace tfrt {
namespace gpu {

llvm::Expected<RCReference<gpu::GpuBuffer>> BlockAllocator::Allocate(
    size_t size, gpu::stream::Stream stream) {
  auto block = FindFreeBlock(size, stream);
  if (!block) {
    // Allocated a new and assume that allocation has gone through well.
    // TODO(apryakhin): Add a real allocation logic and record the usage.
    auto pointer = sub_allocator_->Allocate(size, stream);
    if (!pointer) return pointer.takeError();
    block = Block(stream, pointer.get(), size);
  }

  assert(block && block->allocated == false);
  block->allocated = true;
  allocated_blocks_.emplace(reinterpret_cast<uintptr_t>(
                                block->pointer.raw(sub_allocator_->platform())),
                            *block);

  return TakeRef(new GpuBuffer(block->pointer, block->size, this));
}

void BlockAllocator::Deallocate(const gpu::GpuBuffer& buffer) {
  assert(buffer.IsValid());
  FreeBlock(reinterpret_cast<uintptr_t>(
      buffer.pointer().raw(sub_allocator_->platform())));
}

llvm::Error BlockAllocator::RecordUsage(const gpu::GpuBuffer&,
                                        gpu::stream::Stream) {
  // Nothing to do here, this allocator is currently just a stub.
  return llvm::Error::success();
}

llvm::Expected<BlockAllocator::Block> BlockAllocator::FindFreeBlock(
    size_t size, gpu::stream::Stream stream) {
  // TODO(apryakhin): Add logic to control all possible block sizes
  // as well as the block spliting mechanism.
  // Lookup for an existing block that is the first
  // with the size that is not less than the given key.
  auto free_blocks_it = free_blocks_.lower_bound(size);
  if (free_blocks_it != free_blocks_.end()) {
    auto& blocks = free_blocks_it->second;
    // Lookup an existing free block for a given stream.
    auto it = std::find_if(
        blocks.begin(), blocks.end(),
        [&](Block const& block) { return block.stream == stream; });
    if (it != blocks.end()) {
      auto free_block = *it;
      blocks.erase(it);
      if (blocks.empty()) {
        // No more blocks per 'size', so we remove it.
        free_blocks_.erase(free_blocks_it);
      }
      return free_block;
    }
  }
  return llvm::createStringError(llvm::errc::invalid_argument,
                                 "Could not find a free block.");
}

void BlockAllocator::FreeBlock(uintptr_t address) {
  auto allocated_blocks_it = allocated_blocks_.find(address);
  assert(allocated_blocks_it != allocated_blocks_.end() &&
         "Could not find an allocated block.");
  auto block = allocated_blocks_it->second;
  assert(block.allocated == true);
  // Remove the allocated block from the allocated table.
  allocated_blocks_.erase(allocated_blocks_it);
  auto free_blocks_it = free_blocks_.lower_bound(block.size);
  if (free_blocks_it == free_blocks_.end()) {
    auto emplace_result = free_blocks_.emplace(block.size, BlockList());
    assert(emplace_result.second == true);
    free_blocks_it = emplace_result.first;
  }
  assert(free_blocks_it != free_blocks_.end());
  // Mark this block as free.
  block.allocated = false;
  free_blocks_it->second.push_back(block);
}

}  // namespace gpu
}  // namespace tfrt
