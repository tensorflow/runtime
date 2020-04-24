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

//===- bef_file_impl.h ------------------------------------------*- C++ -*-===//
//
// This file declares BEFFileImpl - the implementation details behind a BEFFile.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_LIB_BEF_EXECUTOR_BEF_FILE_IMPL_H_
#define TFRT_LIB_BEF_EXECUTOR_BEF_FILE_IMPL_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "tfrt/bef_executor/bef_file.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/host_context/kernel_registry.h"
#include "tfrt/host_context/native_function.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {

class BEFFileImpl;
class DecodedLocation;

// This class implements Function for BEF files.
class BEFFunction final : public Function {
 public:
  BEFFunction(string_view name, ArrayRef<TypeName> arguments,
              ArrayRef<TypeName> results, size_t function_offset,
              BEFFileImpl* bef_file)
      : Function(name, arguments, results),
        function_offset_(function_offset),
        bef_file_(bef_file) {}

  BEFFunction(BEFFunction&& other)
      : Function(std::move(other)),
        function_offset_(other.function_offset_),
        bef_file_(other.bef_file_) {}

  size_t function_offset() const { return function_offset_; }
  BEFFileImpl* bef_file() const { return bef_file_; }

  void Execute(ArrayRef<AsyncValue*> arguments,
               MutableArrayRef<RCReference<AsyncValue>> results,
               HostContext* host) const override;
  void AddRef() const override;
  void DropRef() const override;

 private:
  size_t function_offset_;
  BEFFileImpl* bef_file_;
};

// This class is the implementation details behind the BEFFile::Open method,
// which maintains all the state necessary for the BEFExecutor.  It is fully
// public because it is a private implementation detail within this library.
class BEFFileImpl : public BEFFile {
 public:
  ~BEFFileImpl() override;

  explicit BEFFileImpl(ErrorHandler error_handler);

  // Emit an error message about a malformed BEF file.
  void EmitFormatError(const char* message);

  // When decoding a function info descriptor, this describes each register.
  struct RegisterInfo {
    // This is the number of uses of the register in the program.  The value
    // may be deallocated when this number of uses are complete.
    const unsigned user_count;
    // 'value' is not used by BEFFileImpl. BEFExecutor takes ownership of
    // RegisterInfo, and uses 'value' to track the register's contents as it
    // executes a function.
    std::atomic<AsyncValue*> value{nullptr};

    explicit RegisterInfo(unsigned user_count) : user_count(user_count) {}
  };

  // When decoding the kernel table for a function, we get the offset of
  // each kernel as well as the number of operands it has.
  //
  // The executor keeps an array of these, indexed by kernel number to know
  // where to find each kernel in the kernels section, and to know how many
  // arguments are still waiting to come in before the kernel can start.
  //
  // This struct is defined here, because ReadFunction() below will populate it.
  struct KernelInfo {
    unsigned offset;
    std::atomic<int> arguments_not_ready;

    // We initialize the ready list to "num_operands + 1" so we can drop the
    // last count in the executor constructor.
    KernelInfo(unsigned offset, unsigned num_operands)
        : offset(offset), arguments_not_ready(num_operands + 1) {}
  };

  // Decode the specified BEFFunction, returning an ArrayRef of kernel entries
  // for all kernels, decoded information about the registers used by the
  // function, and a table of offsets to each kernel within the function.
  // `host_allocator` is used for the heap-allocated buffer that backs
  // `kernel_infos`.
  //
  // On error, an error is emitted and a null pointer is returned.
  ArrayRef<uint32_t> ReadFunction(size_t function_offset,
                                  ArrayRef<TypeName> results,
                                  size_t* location_offset,
                                  HostArray<RegisterInfo>* register_infos,
                                  HostArray<KernelInfo>* kernel_infos,
                                  SmallVectorImpl<size_t>* result_regs,
                                  HostAllocator* host_allocator);

  // Given an offset into the LocationPositions section, decode it and return
  // a DecodedDiagnostic.
  DecodedLocation DecodeLocation(size_t location_position_offset);

  // Only used for debugging. Populates kernel_names_ on first call, which is
  // slow.
  const char* GetKernelName(size_t kernel_id);

  ErrorHandler error_handler_;

  ArrayRef<uint8_t> location_filenames_section_;
  ArrayRef<uint8_t> location_positions_section_;
  ArrayRef<uint8_t> string_section_;
  ArrayRef<uint8_t> attribute_section_;
  ArrayRef<uint8_t> kernels_section_;
  ArrayRef<uint8_t> types_section_;
  ArrayRef<uint8_t> function_section_;
  ArrayRef<uint8_t> function_index_section_;
  SmallVector<KernelImplementation, 8> kernels_;
  SmallVector<TypeName, 8> type_names_;
  llvm::StringMap<size_t> function_symbol_table_;
  SmallVector<std::unique_ptr<Function>, 8> functions_;

  // Maps from kernel_id to the name of the kernel. Only nonempty when
  // debugging.
  std::vector<const char*> kernel_names_;
};

}  // namespace tfrt

#endif  // TFRT_LIB_BEF_EXECUTOR_BEF_FILE_IMPL_H_
