/*
 * Copyright 2022 The TensorFlow Runtime Authors
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

#ifndef TFRT_BACKENDS_JITRT_MEMORY_MAPPER_H_
#define TFRT_BACKENDS_JITRT_MEMORY_MAPPER_H_

#include <features.h>
#include <sys/types.h>

#include <memory>
#include <string>
#include <system_error>

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"

namespace tfrt {
namespace jitrt {

class JitRtMemoryMapper final
    : public llvm::SectionMemoryManager::MemoryMapper {
 public:
  static std::unique_ptr<JitRtMemoryMapper> Create(llvm::StringRef name);

  llvm::sys::MemoryBlock allocateMappedMemory(
      llvm::SectionMemoryManager::AllocationPurpose purpose, size_t len,
      const llvm::sys::MemoryBlock* const near_block, unsigned prot_flags,
      std::error_code& error_code) override;

  std::error_code protectMappedMemory(const llvm::sys::MemoryBlock& block,
                                      unsigned prot_flags) override;

  std::error_code releaseMappedMemory(llvm::sys::MemoryBlock& block) override;

 private:
  explicit JitRtMemoryMapper(llvm::StringRef name) : name_(name.str()) {}

  std::string name_;
};

}  // namespace jitrt
}  // namespace tfrt

#endif  // TFRT_BACKENDS_JITRT_MEMORY_MAPPER_H_
