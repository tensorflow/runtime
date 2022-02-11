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

// This file declares BEFFileImpl - the implementation details behind a BEFFile.

#ifndef TFRT_LIB_BEF_EXECUTOR_BEF_FILE_IMPL_H_
#define TFRT_LIB_BEF_EXECUTOR_BEF_FILE_IMPL_H_

#include <type_traits>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "tfrt/bef_executor/bef_file.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/host_context/kernel_registry.h"
#include "tfrt/host_context/location.h"
#include "tfrt/host_context/native_function.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {

class BEFFileImpl;
class Value;

// Inlined array to keep registers and kernels info together with a BEF executor
// if their size is small. Default constructed as empty, and must be resized
// before use. If the number of records is larger than `n` it allocates
// HostArray for storage.
template <typename InfoT, size_t n>
class BEFInfoArray {
 public:
  BEFInfoArray() : inlined_size_(0) {}

  ~BEFInfoArray() {
    for (size_t i = 0; i < inlined_size_; ++i) {
      (inlined_data() + i)->~InfoT();
    }
  }

  void resize(size_t num_objects, HostAllocator* allocator) {
    assert(inlined_size_ == 0 && host_array_.size() == 0 && num_objects >= 0);
    if (num_objects > n) {
      host_array_ = HostArray<InfoT>(num_objects, allocator);
    } else {
      inlined_size_ = num_objects;
    }
  }

  MutableArrayRef<InfoT> mutable_array() {
    if (host_array_.size() > 0) {
      return host_array_.mutable_array();
    }
    return {inlined_data(), inlined_size_};
  }

  size_t size() const {
    if (host_array_.size() > 0) {
      return host_array_.size();
    }
    return inlined_size_;
  }

  InfoT& operator[](size_t index) {
    assert(index < size());
    if (host_array_.size() > 0) {
      return host_array_[index];
    }
    return *(inlined_data() + index);
  }

 private:
  BEFInfoArray(const BEFInfoArray&) = delete;
  BEFInfoArray(BEFInfoArray&&) = delete;
  BEFInfoArray& operator=(const BEFInfoArray&) = delete;
  BEFInfoArray& operator=(BEFInfoArray&&) = delete;

  InfoT* inlined_data() { return reinterpret_cast<InfoT*>(&inlined_array_[0]); }

  size_t inlined_size_;
  typename std::aligned_storage<sizeof(InfoT), alignof(InfoT)>::type
      inlined_array_[n];
  HostArray<InfoT> host_array_;
};

// This class implements Function for BEF files.
class BEFFunction : public Function {
 public:
  BEFFunction(string_view name, ArrayRef<TypeName> arguments,
              ArrayRef<TypeName> results, size_t function_offset,
              BEFFileImpl* bef_file)
      : BEFFunction(name, FunctionKind::kBEFFunction, arguments, results,
                    function_offset, bef_file) {}

  BEFFunction(BEFFunction&& other)
      : Function(std::move(other)),
        function_offset_(other.function_offset_),
        bef_file_(other.bef_file_) {}

  size_t function_offset() const { return function_offset_; }
  BEFFileImpl* bef_file() const { return bef_file_; }

  void Execute(const ExecutionContext& exec_ctx,
               ArrayRef<AsyncValue*> arguments,
               MutableArrayRef<RCReference<AsyncValue>> results) const override;
  void ExecuteAsync(
      const ExecutionContext& exec_ctx, ArrayRef<AsyncValue*> arguments,
      MutableArrayRef<RCReference<AsyncValue>> results) const override;
  void AddRef() const override;
  void DropRef() const override;

 protected:
  BEFFunction(string_view name, FunctionKind function_kind,
              ArrayRef<TypeName> arguments, ArrayRef<TypeName> results,
              size_t function_offset, BEFFileImpl* bef_file)
      : Function(name, function_kind, arguments, results),
        function_offset_(function_offset),
        bef_file_(bef_file) {}

  size_t function_offset_;
  BEFFileImpl* bef_file_;
};

// This class implements SyncFunction for BEF files.
class SyncBEFFunction final : public BEFFunction {
 public:
  struct RegisterInfo {
    uint32_t user_count : 31;
    bool is_arg_or_result : 1;
  };

  // Create a SyncBEFFunction. Return nullptr if the BEF file has format error.
  static Expected<std::unique_ptr<SyncBEFFunction>> Create(
      string_view name, ArrayRef<TypeName> arguments,
      ArrayRef<TypeName> results, size_t function_offset,
      BEFFileImpl* bef_file);

  void Execute(
      const ExecutionContext& exec_ctx, ArrayRef<AsyncValue*> arguments,
      MutableArrayRef<RCReference<AsyncValue>> results) const override {
    // TODO(b/160501723): Implement the async Execute() function for
    // SyncBEFFunction when we need to interoperate between sync and async
    // functions. This requires implementing conversion between Value and
    // AsyncValue.
    assert(false && "Not implemented");
  }

  // Execute SyncBEFFunction synchronously. Return excution error in the Error
  // return value.
  Error SyncExecute(const ExecutionContext& exec_ctx,
                    ArrayRef<Value*> arguments, ArrayRef<Value*> results) const;

  // Return an array of descriptors for all of our registers, indexed by
  // their register number.
  ArrayRef<RegisterInfo> register_infos() const { return register_infos_; }

  // Return the kernel entries of all kernels of this function.
  ArrayRef<uint32_t> kernels() const { return kernels_; }

  // Return an array of offsets for all of the kernels in this function,
  // indexed by the kernel number.
  ArrayRef<uint32_t> kernel_offsets() const { return kernel_offsets_; }

  // Return an array of register index for the result registers.
  ArrayRef<uint32_t> result_regs() const { return result_regs_; }

 private:
  SyncBEFFunction(string_view name, ArrayRef<TypeName> arguments,
                  ArrayRef<TypeName> results, size_t function_offset,
                  BEFFileImpl* bef_file)
      : BEFFunction(name, FunctionKind::kSyncBEFFunction, arguments, results,
                    function_offset, bef_file) {}

  // Read the register and kernel information for the function. We cache
  // this information in SyncBEFFunction to avoid repeatedly reading this
  // information for every function execution.
  Error Init();

  // This is an array of descriptors for all of our registers, indexed by
  // their register number.
  llvm::SmallVector<RegisterInfo, 16> register_infos_;

  // This ArrayRef contains kernel entries of all kernels of this function.
  ArrayRef<uint32_t> kernels_;

  // This is an array of offsets for all of the kernels in this function,
  // indexed by the kernel number. This does not include the pseudo kernel as it
  // is not used in the interpreter.
  llvm::SmallVector<uint32_t, 8> kernel_offsets_;

  // This is an array of register index for the result registers.
  llvm::SmallVector<uint32_t, 4> result_regs_;
};

class BEFFileImpl;

class BEFLocationHandler final : public LocationHandler {
 public:
  explicit BEFLocationHandler(BEFFileImpl* bef_file) : bef_file_(bef_file) {}

  DecodedLocation DecodeLocation(Location loc) const override;
  Optional<DebugInfo> GetDebugInfo(Location loc) const override;

 private:
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
  void EmitFormatError(string_view message);

  // When decoding a function info descriptor, this describes each register.
  struct RegisterInfo {
    // This is the number of uses of the register in the program.  The value
    // may be deallocated when this number of uses are complete.
    unsigned user_count = 0;
    // 'value' is not used by BEFFileImpl. BEFExecutor takes ownership of
    // RegisterInfo, and uses 'value' to track the register's contents as it
    // executes a function.
    AsyncValue* value = nullptr;

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
    unsigned stream_id;
    std::atomic<int> arguments_not_ready;

    // We initialize the ready list to at least 1 so that kernels with no
    // operands can be triggered by the pseudo kernel.
    //
    // TODO(b/173800007): Add perf benchmark to illustrate the improvement from
    // the reduced number of kernel enqueues.
    KernelInfo(unsigned offset, unsigned stream_id, unsigned num_operands)
        : offset(offset),
          stream_id(stream_id),
          arguments_not_ready(std::max(1u, num_operands)) {}
  };

  using RegisterInfoArray = BEFInfoArray<BEFFileImpl::RegisterInfo, 24>;
  using KernelInfoArray = BEFInfoArray<BEFFileImpl::KernelInfo, 8>;

  // Decoded BEFFunction information.
  struct FunctionInfo {
    // This ArrayRef contains kernel entries of all kernels of this function.
    ArrayRef<uint32_t> kernels;
    // This is an array of descriptors for all of the kernels in this function,
    // indexed by the kernel number.
    RegisterInfoArray register_infos;
    // This is an array of descriptors for all of our registers, indexed by
    // their register number.
    KernelInfoArray kernel_infos;
  };

  // Decode the specified BEFFunction into the FunctionInfo. `host_allocator` is
  // used for the heap-allocated buffer that backs info arrays in FunctionInfo.
  //
  // On error, an error is emitted and false is returned.
  //
  // ReadFunction is invoked for every BEFFunction execution. We can consider
  // caching the kernel and register information in the BEFFunction object to
  // avoid repeadly reading the same information from the BEF file. However, in
  // the current implementation, we couple BEFExecutor states, e.g. AsyncValue
  // for RegisterInfo, with the function reading. Caching kernel and register
  // information would require us to avoid such coupling which can adversely
  // affect the performance.
  bool ReadFunction(size_t function_offset, ArrayRef<TypeName> results,
                    size_t* location_offset, FunctionInfo* function_info,
                    llvm::SmallVectorImpl<size_t>* result_regs,
                    HostAllocator* host_allocator);

  // Given an offset into the LocationPositions section, decode it and return
  // a DecodedDiagnostic.
  DecodedLocation DecodeLocation(size_t location_position_offset);
  Optional<DebugInfo> GetDebugInfo(size_t location_position_offset);

  // Only used for debugging. If TFRT_BEF_EXECUTOR_DEBUG is not defined, it
  // returns "unknown".
  const char* GetKernelName(size_t kernel_id) const;

  AsyncKernelImplementation GetAsyncKernel(uint32_t kernel_code) const {
    assert(kernel_code < kernels_.size());
    const KernelImplementation& kernel_impl = kernels_[kernel_code];
    assert(kernel_impl.is<AsyncKernelImplementation>());
    return kernel_impl.get<AsyncKernelImplementation>();
  }

  SyncKernelImplementation GetSyncKernel(uint32_t kernel_code) const {
    assert(kernel_code < kernels_.size());
    const KernelImplementation& kernel_impl = kernels_[kernel_code];
    assert(kernel_impl.is<SyncKernelImplementation>());
    return kernel_impl.get<SyncKernelImplementation>();
  }

  ArrayRef<uint8_t> function_section() const { return function_section_; }

  ErrorHandler error_handler_;

  ArrayRef<uint8_t> string_section_;
  ArrayRef<uint8_t> attribute_section_;
  ArrayRef<uint8_t> kernels_section_;
  ArrayRef<uint8_t> types_section_;
  ArrayRef<uint8_t> function_section_;
  ArrayRef<uint8_t> function_index_section_;
  llvm::SmallVector<KernelImplementation, 8> kernels_;
  llvm::SmallVector<TypeName, 8> type_names_;
  llvm::StringMap<size_t> function_symbol_table_;
  llvm::SmallVector<std::unique_ptr<Function>, 8> functions_;
  ArrayRef<uint8_t> location_strings_section_;
  ArrayRef<uint8_t> locations_section_;

#if defined(TFRT_BEF_EXECUTOR_DEBUG)
  // Maps from kernel_id to the name of the kernel.
  std::vector<const char*> kernel_names_;
#endif
};

}  // namespace tfrt

#endif  // TFRT_LIB_BEF_EXECUTOR_BEF_FILE_IMPL_H_
