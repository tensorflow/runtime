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

//===- bef_file.cpp - Abstraction for working with a BEF File -------------===//
//
// This file implements the BEFFile class, which works with the 'Binary Executor
// Format' in memory for use primarily by the BEFExecutor.
//
//===----------------------------------------------------------------------===//

#include "tfrt/bef_executor/bef_file.h"

#include "bef_file_impl.h"
#include "tfrt/host_context/async_value.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/location.h"
#include "tfrt/host_context/native_function.h"
#include "tfrt/support/bef_encoding.h"
#include "tfrt/support/bef_reader.h"
#include "tfrt/support/error_util.h"

namespace tfrt {

// BEFFileReader Implementation

namespace {

// This struct is a simple representation of an entry in FunctionIndex section.
struct FunctionIndex {
  FunctionKind kind;
  size_t function_offset;
  size_t name_offset;
  SmallVector<TypeName, 4> arguments;
  SmallVector<TypeName, 4> results;
};

// This class is a direct reflection of some of the BEF file contents in memory,
// expressed with ranges and other helpers to decode them. The BEFFile
// constructor uses these (combined with the kernel registry) to resolve and
// build the tables in BEFFile that the executor walks.
//
// The convention here is that these functions return false on success and emit
// an error message and return true on failure.
class BEFFileReader : public BEFReader {
 public:
  BEFFileReader(ArrayRef<uint8_t> file, KernelRegistry* registry,
                BEFFileImpl* bef_file)
      : BEFReader(file), registry_(registry), bef_file_(bef_file) {}

  bool ReadNextSection();
  bool ReadKernelsSection(HostAllocator* host_allocator);
  bool ReadTypesSection();
  bool ReadFunctionIndexSection();

 private:
  bool ReadFunctionIndexSectionInternal(
      SmallVectorImpl<FunctionIndex>* function_indices);
  bool ReadFormatVersionSection();
  bool DiagnoseUnknownKernel(size_t kernel_idx, const char* kernel_name,
                             HostAllocator* host_allocator);

  // These are things set up at construction time.
  KernelRegistry* registry_;

  // This is the file structure we're reading.
  BEFFileImpl* bef_file_;
};
}  // namespace

bool BEFFileReader::ReadNextSection() {
  uint8_t section_id;
  ArrayRef<uint8_t> section_data;

  if (ReadSection(&section_id, &section_data)) {
    bef_file_->EmitFormatError("BEF file section header corrupted");
    return true;
  }

  // Process the sections we know about, skip the ones we don't.
  switch (static_cast<BEFSectionID>(section_id)) {
    default:
      SkipPast(section_data);
      break;

    case BEFSectionID::kFormatVersion:
      if (ReadFormatVersionSection()) return true;
      break;

    case tfrt::BEFSectionID::kLocationFilenames:
      bef_file_->location_filenames_section_ = section_data;
      SkipPast(section_data);
      break;

    // We just remember the locations of the LocationPositions section - it is
    // directly indexed with a byte offset.
    case tfrt::BEFSectionID::kLocationPositions:
      bef_file_->location_positions_section_ = section_data;
      SkipPast(section_data);
      break;

    // We just remember the locations of the Strings section - it is directly
    // indexed with a byte offset.
    case BEFSectionID::kStrings:
      bef_file_->string_section_ = section_data;
      SkipPast(section_data);
      break;

    // We just remember the locations of the Constants section - it is directly
    // indexed with a byte offset.
    case BEFSectionID::kAttributes:
      bef_file_->attribute_section_ = section_data;
      SkipPast(section_data);
      break;

    case BEFSectionID::kKernels:
      bef_file_->kernels_section_ = section_data;
      SkipPast(section_data);
      break;

    case BEFSectionID::kTypes:
      bef_file_->types_section_ = section_data;
      SkipPast(section_data);
      break;

    // We just remember the locations of the Functions section - it is directly
    // indexed with a byte offset from the FunctionIndex section.
    case BEFSectionID::kFunctions:
      bef_file_->function_section_ = section_data;
      SkipPast(section_data);
      break;

    case BEFSectionID::kFunctionIndex:
      bef_file_->function_index_section_ = section_data;
      SkipPast(section_data);
      break;
  }

  // Make sure the section reader consumed the right number of bytes.  Not
  // doing so will lead to consistency errors downstream.
  if (file().begin() != section_data.end())
    return bef_file_->EmitFormatError("unexpected data in BEF section"), true;
  return false;
}

bool BEFFileReader::ReadFormatVersionSection() {
  // In the case of the FormatVersion section, we verify that we have a
  // supported version.
  uint8_t version;
  if (ReadByte(&version) || version != kBEFVersion0)
    return bef_file_->EmitFormatError("BEF file has unknown version number"),
           true;

  // TODO: Should have per-file location info in the header.
  return false;
}

// BEF files can refer to arbitrary kernels, which get resolved at load time
// through a KernelRegistry.  When an unknown kernel name is found, this method
// is invoked to try to resolve a nice location to emit the error.
//
// This does not need to be particularly fast, given that it only happens in
// error cases.
//
// If we can't find a nice location, we can fallback to a poor location.
bool BEFFileReader::DiagnoseUnknownKernel(size_t kernel_idx,
                                          const char* kernel_name,
                                          HostAllocator* host_allocator) {
  std::string error_message =
      "unknown kernel name '" + std::string(kernel_name) + "'";

  // This gets run before the functionindex is processed, but we want to use it
  // below.
  SmallVector<FunctionIndex, 8> function_indices;
  if (ReadFunctionIndexSectionInternal(&function_indices)) {
    bef_file_->EmitFormatError(error_message.c_str());
    return true;
  }

  // The unknown kernel must be referenced by some function in the program,
  // and each kernel record has location info.  Scan through to see if we can
  // figure out where the reference is coming from.
  SmallVector<size_t, 4> result_regs;

  for (const auto& function_index : function_indices) {
    if (function_index.kind == FunctionKind::kNativeFunction) continue;

    BEFFileImpl::FunctionInfo function_info;

    result_regs.clear();
    size_t location_offset;
    bool success = bef_file_->ReadFunction(
        function_index.function_offset, function_index.results,
        &location_offset, &function_info, &result_regs, host_allocator);
    if (!success) continue;

    // Decode all of the kernels to see if any refers to our unknown kernel.
    MutableArrayRef<BEFFileImpl::KernelInfo> kernel_infos_array =
        function_info.kernel_infos.mutable_array();
    for (const auto& kernel_info : kernel_infos_array) {
      assert(kernel_info.offset % kKernelEntryAlignment == 0);
      BEFKernel kernel(function_info.kernels.data() +
                       kernel_info.offset / kKernelEntryAlignment);

      // Okay, we decoded the kernel.  See if this is referring to the
      // current kernel_idx.  If so, we can use its location.  We know that the
      // relevant KernelEntry for the opcode is always first, so we just
      // need to check it.
      if (kernel.kernel_code() == kernel_idx) {
        auto decoded_loc = bef_file_->DecodeLocation(kernel.kernel_location());
        bef_file_->error_handler_(
            DecodedDiagnostic(decoded_loc, error_message));
        return true;
      }
    }
  }

  bef_file_->EmitFormatError(error_message.c_str());
  return true;
}

// Read the Kernels section from a BEF file, resolving the kernels and
// returning false on success.  Emit an error and return true on failure.
bool BEFFileReader::ReadKernelsSection(HostAllocator* host_allocator) {
  auto format_error = [&]() -> bool {
    bef_file_->EmitFormatError("invalid Kernels section in BEF file");
    return true;
  };

  BEFReader reader(bef_file_->kernels_section_);

  size_t num_kernels;
  if (reader.ReadInt(&num_kernels)) return format_error();

  bef_file_->kernels_.reserve(num_kernels);
  while (num_kernels--) {
    // Each kernel is encoded as an offset into the string table of the
    // kernel name.
    size_t kernel_name_offset;

    // Make sure the kernel name is valid.
    if (reader.ReadInt(&kernel_name_offset) ||
        kernel_name_offset >= bef_file_->string_section_.size())
      return format_error();

    // If this is an unknown kernel, bail out.
    const char* kernel_name = reinterpret_cast<const char*>(
        &bef_file_->string_section_[kernel_name_offset]);

    auto kernel = registry_->GetKernel(kernel_name);
    if (kernel.isNull()) {
      return DiagnoseUnknownKernel(bef_file_->kernels_.size(), kernel_name,
                                   host_allocator);
    }

    // Otherwise remember it.
    bef_file_->kernels_.push_back(kernel);
  }

  return false;
}

// Read the Types section from a BEF file, resolving the types and returning
// false on success.  Emit an error and return true on failure.
bool BEFFileReader::ReadTypesSection() {
  auto format_error = [&]() -> bool {
    return bef_file_->EmitFormatError("invalid Types section in BEF file"),
           true;
  };

  BEFReader reader(bef_file_->types_section_);

  size_t num_types;
  if (reader.ReadInt(&num_types)) return format_error();

  bef_file_->type_names_.reserve(num_types);
  while (num_types--) {
    // Each type is encoded as an offset into the string table of the type name.
    size_t type_name_offset;

    // Make sure the kernel name is valid.
    if (reader.ReadInt(&type_name_offset) ||
        type_name_offset >= bef_file_->string_section_.size())
      return format_error();

    // If this is an unknown type, bail out.
    const char* type_name_str = reinterpret_cast<const char*>(
        &bef_file_->string_section_[type_name_offset]);
    auto type_name = registry_->GetType(type_name_str);

    // Otherwise remember it.
    bef_file_->type_names_.push_back(type_name);
  }

  return false;
}

// Read the FunctionIndex section from a BEF file, and put the information into
// `function_indices`.
bool BEFFileReader::ReadFunctionIndexSectionInternal(
    SmallVectorImpl<FunctionIndex>* function_indices) {
  BEFReader reader(bef_file_->function_index_section_);

  size_t num_functions;
  if (reader.ReadInt(&num_functions)) return true;

  // bef_file_->functions_.reserve(num_functions);
  function_indices->clear();
  function_indices->reserve(num_functions);

  SmallVector<TypeName, 4> arguments;
  SmallVector<TypeName, 4> results;
  while (num_functions--) {
    function_indices->emplace_back();
    auto& function_index = function_indices->back();

    uint8_t function_kind;
    if (reader.ReadByte(&function_kind) ||
        reader.ReadInt(&function_index.function_offset) ||
        reader.ReadInt(&function_index.name_offset) ||
        function_index.name_offset >= bef_file_->string_section_.size()) {
      return true;
    }

    function_index.kind = static_cast<FunctionKind>(function_kind);

    // Read the argument types.
    size_t num_args;
    if (reader.ReadInt(&num_args)) return true;

    while (num_args--) {
      size_t arg_type;
      if (reader.ReadInt(&arg_type)) return true;

      if (arg_type >= bef_file_->type_names_.size()) return true;
      function_index.arguments.push_back(bef_file_->type_names_[arg_type]);
    }

    // Read the result types.
    size_t num_results;
    if (reader.ReadInt(&num_results)) return true;

    while (num_results--) {
      size_t result_type;
      if (reader.ReadInt(&result_type)) return true;

      if (result_type >= bef_file_->type_names_.size()) return true;
      function_index.results.push_back(bef_file_->type_names_[result_type]);
    }
  }

  return false;
}

// Read the FunctionIndex section from a BEF file, building the functions_ table
// and the function_symbol_table_, and returning false on success. Emit an error
// and return true on failure.
bool BEFFileReader::ReadFunctionIndexSection() {
  auto format_error = [&]() -> bool {
    bef_file_->EmitFormatError("invalid FunctionIndex section in BEF file");
    return true;
  };

  SmallVector<FunctionIndex, 8> function_indices;
  if (ReadFunctionIndexSectionInternal(&function_indices))
    return format_error();

  bef_file_->functions_.reserve(function_indices.size());

  for (const auto& function_index : function_indices) {
    // Put named functions in the function_symbol_table_.
    const char* name = reinterpret_cast<const char*>(
        &bef_file_->string_section_[function_index.name_offset]);
    if (*name)
      bef_file_->function_symbol_table_[name] = bef_file_->functions_.size();

    // TODO(tfrt-devs): Consider adding a factory for functions.
    switch (function_index.kind) {
      case FunctionKind::kBEFFunction: {
        if (function_index.function_offset >=
            bef_file_->function_section_.size())
          return format_error();
        auto bef_function = std::make_unique<BEFFunction>(
            name, function_index.arguments, function_index.results,
            function_index.function_offset, bef_file_);
        bef_file_->functions_.push_back(std::move(bef_function));
        break;
      }
      case FunctionKind::kSyncBEFFunction: {
        if (function_index.function_offset >=
            bef_file_->function_section_.size())
          return format_error();
        auto bef_function = SyncBEFFunction::Create(
            name, function_index.arguments, function_index.results,
            function_index.function_offset, bef_file_);
        if (!bef_function) return format_error();
        bef_file_->functions_.push_back(std::move(bef_function.get()));
        break;
      }
      case FunctionKind::kNativeFunction: {
        auto callable = NativeFunctionRegistry::GetGlobalRegistry().Get(name);
        if (callable == nullptr) {
          bef_file_->EmitFormatError(
              "unable to find native function in global registry");
          return true;
        }
        bef_file_->functions_.push_back(std::make_unique<NativeFunction>(
            name, function_index.arguments, function_index.results, callable));
        break;
      }
    }
  }

  return false;
}

// BEFFile / BEFFileImpl Implementation

BEFFile::~BEFFile() {}

RCReference<BEFFile> BEFFile::Open(ArrayRef<uint8_t> file,
                                   KernelRegistry* registry,
                                   ErrorHandler error_handler,
                                   tfrt::HostAllocator* host_allocator) {
  auto* bef_impl = new BEFFileImpl(error_handler);
  auto bef_rc = TakeRef(bef_impl);

  BEFFileReader reader(file, registry, bef_impl);

  uint8_t header[2];

  // Make sure the file has a header.
  if (reader.ReadByte(&header[0]) || reader.ReadByte(&header[1]) ||
      header[0] != kBEFMagic1 || header[1] != kBEFMagic2) {
    bef_impl->EmitFormatError("invalid BEF file header detected");
    return {};
  }

  while (!reader.Empty()) {
    if (reader.ReadNextSection()) return {};
  }

  // Now that we've figured out the contents of the sections, resolve some
  // things.
  if (reader.ReadKernelsSection(host_allocator) || reader.ReadTypesSection() ||
      reader.ReadFunctionIndexSection())
    return {};

  // Now that we decoded the whole thing, return the BEFFile to the caller.
  return bef_rc;
}

BEFFileImpl::BEFFileImpl(std::function<void(DecodedDiagnostic)> error_handler)
    : error_handler_(error_handler) {}

BEFFileImpl::~BEFFileImpl() {}

void BEFFileImpl::EmitFormatError(const char* message) {
  error_handler_(DecodedDiagnostic(message));
}

// TODO(b/160504938): Refactor this function to return Error instead of
// reporting error via EmitFormatError to make the API more natural.
bool BEFFileImpl::ReadFunction(size_t function_offset,
                               ArrayRef<TypeName> results,
                               size_t* location_offset,
                               FunctionInfo* function_info,
                               SmallVectorImpl<size_t>* result_regs,
                               HostAllocator* host_allocator) {
  auto format_error = [&]() -> bool {
    EmitFormatError("invalid Function section in BEF file");
    return false;
  };

  if (function_offset >= function_section_.size()) return format_error();

  BEFReader reader(function_section_.drop_front(function_offset));

  // First we have the location info and register info table.
  size_t num_registers;
  if (reader.ReadInt(location_offset) || reader.ReadInt(&num_registers))
    return format_error();

  function_info->register_infos.resize(num_registers, host_allocator);
  auto* register_info_ptr =
      function_info->register_infos.mutable_array().data();
  unsigned register_idx = 0;
  while (num_registers--) {
    size_t user_count;
    if (reader.ReadInt(&user_count)) return format_error();
    new (register_info_ptr + register_idx) RegisterInfo(user_count);
    ++register_idx;
  }

  // Next we have the kernel index table.
  size_t num_kernels;
  if (reader.ReadInt(&num_kernels)) return format_error();

  function_info->kernel_infos.resize(num_kernels, host_allocator);
  auto* kernel_info_ptr = function_info->kernel_infos.mutable_array().data();
  unsigned kernel_idx = 0;
  while (num_kernels--) {
    size_t offset, num_operands;
    if (reader.ReadInt(&offset) || reader.ReadInt(&num_operands))
      return format_error();
    new (kernel_info_ptr + kernel_idx)
        KernelInfo(unsigned(offset), unsigned(num_operands));
    ++kernel_idx;
  }

  // Read the result registers.
  result_regs->reserve(results.size());
  for (unsigned i = 0, e = results.size(); i != e; ++i) {
    size_t result_reg;
    if (reader.ReadInt(&result_reg) || result_reg >= num_registers)
      return format_error();
    result_regs->push_back(result_reg);
  }

  // Kernels are aligned to kKernelEntryAlignment.
  if (reader.ReadAlignment(kKernelEntryAlignment)) return format_error();

  // We found the start of our kernel section.
  function_info->kernels = llvm::makeArrayRef(
      reinterpret_cast<const uint32_t*>(reader.file().begin()),
      reader.file().size() / kKernelEntryAlignment);

  return true;
}

// Given an offset into location_positions_section_, decode it and return
// a DecodedDiagnostic.
DecodedLocation BEFFileImpl::DecodeLocation(size_t location_position_offset) {
  DecodedLocation result;

  // Read from location_positions_section_, from the specified offset.
  BEFReader reader(location_positions_section_);
  if (location_position_offset >= location_positions_section_.size())
    return result;

  reader.SkipOffset(location_position_offset);

  size_t file_idx, line, column;
  if (reader.ReadInt(&file_idx) || reader.ReadInt(&line) ||
      reader.ReadInt(&column))
    return result;

  result.line = line;
  result.column = column;

  // The file is an index into location_filenames_section_.  We expect
  // lookups in this section to be rare (only on errors) and the number of
  // entries to be small, so we just scan through the section.
  string_view filenames(
      reinterpret_cast<const char*>(location_filenames_section_.data()),
      location_filenames_section_.size());
  // Skip over file_idx number of entries.
  while (file_idx && !filenames.empty()) {
    auto next_end = filenames.find('\0');
    if (next_end == string_view::npos)
      filenames = "";
    else
      filenames = filenames.drop_front(next_end + 1);
  }

  // The filename is everything up to the next \0.
  auto end_pos = filenames.find('\0');
  if (end_pos != string_view::npos)
    result.filename = filenames.substr(0, end_pos).str();
  return result;
}

const char* BEFFileImpl::GetKernelName(size_t kernel_id) {
  // If this is the first time we've been called, decode kernels_section_ and
  // initialize kernel_names_.
  if (kernel_names_.empty()) {
    BEFReader reader(kernels_section_);

    size_t num_kernels;
    if (reader.ReadInt(&num_kernels)) return "(failed to decode num_kernels)";

    kernel_names_.reserve(num_kernels);
    while (num_kernels--) {
      size_t kernel_name_offset;

      const char* kernel_name = "(failed to read kernel_name)";
      if (!reader.ReadInt(&kernel_name_offset) &&
          kernel_name_offset < string_section_.size())
        kernel_name =
            reinterpret_cast<const char*>(&string_section_[kernel_name_offset]);

      kernel_names_.push_back(kernel_name);
    }
  }

  if (kernel_id >= kernel_names_.size()) return "(invalid kernel_id)";

  return kernel_names_[kernel_id];
}

// Read a list of function names out of the BEF file function index.
void BEFFile::GetFunctionList(SmallVectorImpl<const Function*>* results) const {
  auto* impl = static_cast<const BEFFileImpl*>(this);

  results->reserve(impl->functions_.size());
  for (auto& fn : impl->functions_) results->push_back(fn.get());
}

// Return the Function record with the specified name, or null if it isn't
// found in this BEF file.
const Function* BEFFile::GetFunction(string_view function_name) const {
  auto* impl = static_cast<const BEFFileImpl*>(this);

  auto it = impl->function_symbol_table_.find(function_name);
  if (it == impl->function_symbol_table_.end()) return nullptr;
  return impl->functions_[it->second].get();
}

Expected<std::unique_ptr<SyncBEFFunction>> SyncBEFFunction::Create(
    string_view name, ArrayRef<TypeName> arguments, ArrayRef<TypeName> results,
    size_t function_offset, BEFFileImpl* bef_file) {
  // std::make_unique cannot be used, as the constructor of SyncBEFFunction is
  // private.
  // NOLINTNEXTLINE
  auto bef_function = std::unique_ptr<SyncBEFFunction>(
      new SyncBEFFunction(name, arguments, results, function_offset, bef_file));

  if (auto error = bef_function->Init())
    return std::move(error);
  else
    return std::move(bef_function);
}

Error SyncBEFFunction::Init() {
  assert(register_infos_.empty());
  assert(kernels_.empty());
  assert(kernel_offsets_.empty());
  assert(result_regs_.empty());

  auto format_error = [&](const char* msg) -> Error {
    return MakeStringError("Invalid Function section in BEF file: ", msg);
  };

  auto function_section = bef_file_->function_section();
  if (function_offset_ >= function_section.size())
    return format_error("Invalid function offset");

  BEFReader reader(function_section.drop_front(function_offset_));

  // First we have the location info and register info table.
  size_t num_registers;
  size_t location_offset;
  if (reader.ReadInt(&location_offset) || reader.ReadInt(&num_registers))
    return format_error("Failed to read location_offset or num_registers");

  register_infos_.reserve(num_registers);
  for (size_t reg_index = 0; reg_index < num_registers; ++reg_index) {
    size_t user_count;
    if (reader.ReadInt(&user_count))
      return format_error("Failed to read register user_count");

    bool is_arg = (reg_index < num_arguments());
    if (is_arg) {
      // +1 on the user count so that we do not reset the argument Value in the
      // function evaluation.
      ++user_count;
    }
    register_infos_.push_back(
        RegisterInfo{static_cast<uint32_t>(user_count), is_arg});
  }

  // Next we have the kernel index table.
  size_t num_kernels;
  if (reader.ReadInt(&num_kernels))
    return format_error("Failed to read num_kernels");

  kernel_offsets_.reserve(num_kernels);
  for (size_t kernel_index = 0; kernel_index < num_kernels; ++kernel_index) {
    size_t offset, num_operands;
    if (reader.ReadInt(&offset) || reader.ReadInt(&num_operands))
      return format_error("Failed to read kernel offset or num_operands");

    if (num_arguments() == 0 || kernel_index != 0) {
      // Only insert if it is not the first kernel which is an argument pseudo
      // kernel when there are arguments.
      kernel_offsets_.push_back(offset);
    }
  }

  // Read the result registers.
  size_t num_results = result_types().size();
  result_regs_.reserve(num_results);
  for (unsigned i = 0, e = num_results; i != e; ++i) {
    size_t result_reg;
    if (reader.ReadInt(&result_reg) || result_reg >= num_registers)
      return format_error("Failed to read result_reg");
    result_regs_.push_back(result_reg);

    // +1 on the user count so that we do not reset the result Value in the
    // function evaluation.
    auto& reg_info = register_infos_[result_reg];
    reg_info.is_arg_or_result = true;
    ++reg_info.user_count;
  }

  // Kernels are aligned to kKernelEntryAlignment.
  if (reader.ReadAlignment(kKernelEntryAlignment))
    return format_error("Failed to align BEF to kKernelEntryAlignment");

  // We found the start of our kernel section.
  kernels_ = llvm::makeArrayRef(
      reinterpret_cast<const uint32_t*>(reader.file().begin()),
      reader.file().size() / kKernelEntryAlignment);

  return Error::success();
}

Error ExecuteSyncBEFFunction(const Function& func,
                             const ExecutionContext& exec_ctx,
                             ArrayRef<Value*> arguments,
                             ArrayRef<Value*> results) {
  assert(func.function_kind() == FunctionKind::kSyncBEFFunction);
  const SyncBEFFunction& sync_func = static_cast<const SyncBEFFunction&>(func);
  return sync_func.SyncExecute(exec_ctx, arguments, results);
}

}  // namespace tfrt
