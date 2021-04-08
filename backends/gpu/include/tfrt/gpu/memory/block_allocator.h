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

// BlockAllocator
//
// This file defines the interface for a block caching GPU memory allocator.
#ifndef TFRT_GPU_MEMORY_BLOCK_ALLOCATOR_H_
#define TFRT_GPU_MEMORY_BLOCK_ALLOCATOR_H_

#include <map>
#include <unordered_map>
#include <vector>

#include "llvm/Support/Error.h"
#include "tfrt/gpu/memory/gpu_allocator.h"
#include "tfrt/gpu/memory/sub_allocator.h"
#include "tfrt/gpu/stream/stream_wrapper.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/support/ref_count.h"

namespace tfrt {
namespace gpu {

// BlockAllocator is the default TFRT CUDA allocator.
// BlockAllocator is thread-safe.
class BlockAllocator : public gpu::GpuAllocator {
 public:
  explicit BlockAllocator(SubAllocator* sub_allocator)
      : sub_allocator_(sub_allocator) {}

  llvm::Expected<RCReference<gpu::GpuBuffer>> Allocate(
      size_t size, gpu::stream::Stream stream) override;

  void Deallocate(const gpu::GpuBuffer& buffer) override;

  llvm::Error RecordUsage(const gpu::GpuBuffer& buffer,
                          gpu::stream::Stream stream) override;

 private:
  // Represents a block of device memory.
  struct Block {
    Block(gpu::stream::Stream stream, gpu::stream::Pointer<void> pointer,
          size_t size)
        : stream(stream), pointer(pointer), size(size), allocated(false) {}
    // Defines the stream associated with this block.
    gpu::stream::Stream stream;
    // Defines the device memory pointer;
    gpu::stream::Pointer<void> pointer;
    // Defines the block size.
    size_t size;
    // Defines whether this block is in use.
    bool allocated;
  };

  using BlockList = std::vector<Block>;

  // Returns an existing free block and removes it from the block pool.
  llvm::Expected<Block> FindFreeBlock(size_t size, gpu::stream::Stream stream);

  // Bring an allocated block from the allocated blocks map back
  // to the block pool.
  void FreeBlock(uintptr_t address);

  // Stores all existing unused blocks.
  // Mapping: Blocksize -> (Block0, Block1...BlockN-1)
  // TODO(apryakhin): Consider adding unique_ptr
  // and alternative container for efficiency.
  std::map<size_t, BlockList> free_blocks_;

  // Stores the used blocks.
  // Mapping: BlockAddress -> Block.
  std::unordered_map<uintptr_t, Block> allocated_blocks_;

  SubAllocator* sub_allocator_ = nullptr;
};

}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_MEMORY_BLOCK_ALLOCATOR_H_
