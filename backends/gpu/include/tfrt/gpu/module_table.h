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

// ModuleTable is a resource used to load and keep track of CUDA modules and
// their functions.
//
// All functions from all modules are stored in a flat array that is indexed by
// their absolute order within the Spec provided at creation time.

#ifndef TFRT_GPU_MODULE_TABLE_H_
#define TFRT_GPU_MODULE_TABLE_H_

#include <vector>

#include "tfrt/gpu/stream/stream_wrapper.h"
#include "tfrt/support/forward_decls.h"

namespace tfrt {

class AggregateAttr;

template <typename T>
class ArrayAttribute;

namespace gpu {

class ModuleFuncHandle;

class ModuleTable {
 public:
  virtual ~ModuleTable(){};

  // ModuleTable::Spec specifies the modules to be loaded by the ModuleTable as
  // well as the functions from those modules that should be loaded.
  struct Spec {
    struct ModuleDesc {
      string_view module_data;
      std::vector<string_view> function_symbols;
    };
    std::vector<ModuleDesc> modules;
  };

  // Given a Spec and a Context, loads all requested modules and gets handles to
  // all specified functions. Each function is assigned an index which is the
  // sum of their module's index in the spec modules vector and the function
  // index in the ModuleDesc function_symbols vector.
  //
  // TODO(imintz): This interface is primarily used by the compiler where this
  // sort of global indexing is easier. We may want to add a separate lookup
  // function based on module index and relative function index.
  static llvm::Expected<std::unique_ptr<ModuleTable>> Create(
      stream::CurrentContext current, const Spec& spec);

  // Retrieves function pointer for the given handle.
  virtual stream::Function GetFunction(ModuleFuncHandle handle) const = 0;
};

// Parses a raw attributes as ModuleTable::Spec. The source AggregateAttrs
// must outlive the resultant spec.
llvm::Expected<ModuleTable::Spec> ParseModuleTableSpec(
    const AggregateAttr& module_table,
    const ArrayAttribute<int32_t>& funcs_per_module,
    const AggregateAttr& function_table);

// Opaque handle to functions. These are supplied as attributes in MLIR.
class ModuleFuncHandle {
 public:
  ModuleFuncHandle() = default;
  explicit ModuleFuncHandle(uint32_t init_value) : raw_(init_value) {}
  uint32_t raw() const { return raw_; }

 private:
  uint32_t raw_ = 0;
};

// MultiDeviceModuleTable is a map from Device to ModuleTable indexed by device
// id.
class MultiDeviceModuleTable {
 public:
  static std::unique_ptr<MultiDeviceModuleTable> Create();

  virtual ~MultiDeviceModuleTable() = default;

  // Takes ownership of a ModuleTable and associates it with the device id of
  // the provided device.
  // invalid_argument error if the device already has an associated table.
  virtual llvm::Error AddTable(const stream::Device& device,
                               std::unique_ptr<ModuleTable> table) = 0;

  // Returns the ModuleTable associated with the device, or None if the device
  // has not yet been associated with a ModuleTable.
  virtual llvm::Optional<const ModuleTable*> GetTable(
      const stream::Device& device) const = 0;
};

}  // namespace gpu
}  // namespace tfrt

#endif  // TFRT_GPU_MODULE_TABLE_H_
