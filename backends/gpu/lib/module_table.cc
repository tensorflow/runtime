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

// Implementation of the ModuleTable used to load and track CUDA modules.
#include "tfrt/gpu/module_table.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "tfrt/host_context/attribute_utils.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/logging.h"

namespace tfrt {
namespace gpu {
namespace {

class ModuleTableImpl : public ModuleTable {
 public:
  ModuleTableImpl(std::vector<stream::OwningModule> modules,
                  std::vector<stream::Function> functions)
      : modules_(std::move(modules)), functions_(std::move(functions)) {}

  stream::Function GetFunction(ModuleFuncHandle handle) const override {
    return functions_.at(handle.raw());
  }

 private:
  // RAII handles to the modules. These are not accessed after initalization and
  // are stored purely for lifetime management.
  const std::vector<stream::OwningModule> modules_;

  // Table of loaded function pointers.
  const std::vector<stream::Function> functions_;
};

// Maintains a sorted vector of mappings from device_id to module table.
class MultiDeviceModuleTableImpl : public MultiDeviceModuleTable {
 public:
  llvm::Error AddTable(const stream::Device& device,
                       std::unique_ptr<ModuleTable> table) override {
    const int device_id = device.id(device.platform());
    Entry to_insert{device_id, std::move(table)};
    const auto insertion_it = std::lower_bound(tables_.begin(), tables_.end(),
                                               to_insert, CompareEntries{});
    if (insertion_it != tables_.end() &&
        insertion_it->device_id == to_insert.device_id) {
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          StrCat("Unable to load CUDA module table. Table has "
                 "already been created for device ",
                 device_id));
    }
    tables_.insert(insertion_it, std::move(to_insert));
    return llvm::Error::success();
  }

  llvm::Optional<const ModuleTable*> GetTable(
      const stream::Device& device) const override {
    const int device_id = device.id(device.platform());
    auto it = std::lower_bound(tables_.begin(), tables_.end(), Entry{device_id},
                               CompareEntries{});
    if (it == tables_.end()) {
      return llvm::None;
    }
    return it->module_table.get();
  }

 private:
  struct Entry {
    int device_id;
    std::unique_ptr<ModuleTable> module_table;
  };
  struct CompareEntries {
    bool operator()(const Entry& lhs, const Entry& rhs) const {
      return lhs.device_id < rhs.device_id;
    }
  };

  // Sorted vector. 16 is chosen as a cautious upper estimate for the number of
  // GPUs in a system.
  llvm::SmallVector<Entry, 16> tables_;
};

}  // namespace

// Wrapper for module loading that prints logs when in debug mode.
static llvm::Expected<stream::OwningModule> LoadModule(
    stream::CurrentContext current, const char* module_data) {
#ifdef NDEBUG
  return stream::ModuleLoadData(current, module_data);
#else
  std::string info_log;
  std::string error_log;

  stream::ModuleLoadOptions options{&info_log, &error_log, 1};
  auto maybe_module = stream::ModuleLoadDataEx(current, module_data, options);
  if (!info_log.empty()) {
    TFRT_LOG_INFO << "CUDA JIT info Log: " << info_log;
  }
  if (!maybe_module) {
    TFRT_LOG_ERROR << "CUDA JIT error Log: " << error_log;
  }
  return maybe_module;
#endif
}

static bool IsCString(string_view s) { return s.back() == 0; }

/*static*/ std::unique_ptr<MultiDeviceModuleTable>
MultiDeviceModuleTable::Create() {
  return std::make_unique<MultiDeviceModuleTableImpl>();
}

// Safely converts string_views to c_str for functions that require char * null
// terminated string arguments.
#define AS_CSTR(s) (IsCString(s) ? s.data() : s.str().c_str())

/*static*/ llvm::Expected<std::unique_ptr<ModuleTable>> ModuleTable::Create(
    stream::CurrentContext current, const ModuleTable::Spec& spec) {
  const int module_count = spec.modules.size();
  int function_count = 0;
  for (const auto& module_spec : spec.modules) {
    function_count += module_spec.function_symbols.size();
  }

  std::vector<stream::OwningModule> modules;
  modules.reserve(module_count);
  std::vector<stream::Function> functions;
  functions.reserve(function_count);

  for (const auto& module_spec : spec.modules) {
    TFRT_ASSIGN_OR_RETURN(
        std::back_inserter(modules),
        LoadModule(current, AS_CSTR(module_spec.module_data)));
    for (string_view function_symbol : module_spec.function_symbols) {
      TFRT_ASSIGN_OR_RETURN(
          std::back_inserter(functions),
          stream::ModuleGetFunction(modules.back().get(),
                                    AS_CSTR(function_symbol)));
    }
  }

  return std::make_unique<ModuleTableImpl>(std::move(modules),
                                           std::move(functions));
}

#undef AS_CSTR

template <typename... T>
static llvm::Error SpecParseError(T&&... messages) {
  return llvm::createStringError(llvm::errc::invalid_argument,
                                 StrCat("CUDA module table spec is malformed; ",
                                        std::forward<T>(messages)...));
}

llvm::Expected<ModuleTable::Spec> ParseModuleTableSpec(
    const AggregateAttr& module_table,
    const ArrayAttribute<int32_t>& funcs_per_module,
    const AggregateAttr& function_table) {
  // TODO(imintz): Develop a versioning scheme for the spec. Ideally this would
  // be standard for TFRT and not an adhoc versioning within the attribute.

  const size_t num_modules = module_table.GetNumElements();
  if (num_modules == 0) {
    return SpecParseError("No module data specified");
  }

  if (num_modules != funcs_per_module.size()) {
    return SpecParseError(
        "Number of entries in function count list doesn't match number of "
        "modules; ",
        funcs_per_module.size(), " vs ", num_modules);
  }

  const size_t total_function_count = function_table.GetNumElements();
  size_t accum_func_list_size = 0;
  for (const auto& count : llvm::enumerate(funcs_per_module.data())) {
    if (count.value() <= 0) {
      return SpecParseError(
          llvm::formatv("Invalid function count ({0}) specified for module {1}",
                        count.value(), count.index()));
    }
    accum_func_list_size += count.value();
  }
  if (accum_func_list_size != total_function_count) {
    return SpecParseError(
        "Mismatch between total size of functions per module and size of "
        "function table; ",
        accum_func_list_size, " vs ", total_function_count);
  }

  ModuleTable::Spec spec;
  spec.modules.reserve(num_modules);

  size_t func_table_offset = 0;
  for (int module_index = 0; module_index < num_modules; ++module_index) {
    TypedAttrBase module_element = module_table.GetAttribute(module_index);
    if (!module_element.isa<StringAttr>()) {
      return SpecParseError(llvm::formatv(
          "Expected StringAttr but got {0}; module {1}",
          // TODO(imintz): Human readable BEFAttributeType.
          static_cast<uint16_t>(module_element.type()), module_index));
    }

    const size_t num_functions = funcs_per_module[module_index];
    std::vector<string_view> function_symbols;
    function_symbols.reserve(num_functions);
    for (size_t func_index = 0; func_index < num_functions; ++func_index) {
      const TypedAttrBase func_symbol =
          function_table.GetAttribute(func_table_offset + func_index);
      if (!func_symbol.isa<StringAttr>()) {
        return SpecParseError(llvm::formatv(
            "Expected StringAttr but got {0}; module {1}, function {2}",
            static_cast<uint16_t>(func_symbol.type()), module_index,
            func_index));
      }
      function_symbols.push_back(func_symbol.cast<StringAttr>().GetValue());
    }

    spec.modules.push_back({module_element.cast<StringAttr>().GetValue(),
                            std::move(function_symbols)});
  }

  return spec;
}

}  // namespace gpu
}  // namespace tfrt
