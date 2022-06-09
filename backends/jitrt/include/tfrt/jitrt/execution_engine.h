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

#ifndef TFRT_BACKENDS_JITRT_EXECUTION_ENGINE_H_
#define TFRT_BACKENDS_JITRT_EXECUTION_ENGINE_H_

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Support/MemoryBuffer.h"
#include "mlir/IR/BuiltinOps.h"

namespace tfrt {
namespace jitrt {

// Forward declare types enabling compiled kernel <-> runtime integration.
namespace runtime {
struct KernelContext;
}  // namespace runtime

// A pre-fabricated wrapper around ORC JIT stack for running JitRt programs.
//
// It allows to run jit-compiled JitRt programs, AOT compile them or load
// previously AOT compiled JitRt programs.
//
// JitRt Executable is responsible for the function signature verification,
// arguments packing according to the ABI, results decoding and linking the
// executable with runtime intrinsics. Execution engine only helps with setting
// up ORC JIT stack to support the execution, but itself doesn't know what it is
// executing.
class ExecutionEngine {
 public:
  // Pointer to a compiled JitRt entrypoint function.
  //
  // JitRt entrypoint function expects all arguments to be passed as an array of
  // opaque pointers to the actual values. In C++ it would look like this:
  //
  //   void entrypoint(int32_t arg0, float arg1, ...);
  //
  //   void __jitrt_entrypoint(void** args) {
  //      int32_t arg0 = *reinterpret_cast<int32_t*>(args[0]);
  //      float arg1 = *reinterpret_cast<float*>(args[1]);
  //      ...
  //      entrypoint(arg0, arg1, ...);
  //   }
  //
  // This is required to avoid dealing with ABI of the compiled function. See
  // `SetUpEntrypointFunction` for implementation details.
  using EntrypointFunctionPtr = void (*)(void **);

  // Callback to register symbols with the execution engine (e.g. to register
  // custom runtime intrinsics for Gpu integration).
  using SymbolsBinding =
      std::function<llvm::orc::SymbolMap(llvm::orc::MangleAndInterner)>;

  // Compose multiple symbol bindings into a single symbol binding function.
  static SymbolsBinding BindAll(std::vector<SymbolsBinding> bindings);

  //------------------------------------------------------------------------- //
  // Options for creating execution engine from an MLIR source.
  //------------------------------------------------------------------------- //

  struct JitOptions {
    // User-provided codegen optimization level.
    llvm::CodeGenOpt::Level opt_level = llvm::CodeGenOpt::Level::Default;

    // User-provided target machine specification.
    llvm::TargetMachine *target_machine = nullptr;

    // User-provided memory mapper for allocating memory for executables.
    llvm::SectionMemoryManager::MemoryMapper *section_memory_mapper = nullptr;

    // User-provided bindings for symbols.
    SymbolsBinding symbols_binding = nullptr;

    // Notify the llvm's global GDB notifications listener.
    bool enable_gdb_listener = true;

    // Notify the llvm's global Perf notifications listener.
    bool enable_perf_listener = true;

    // Save compiled object file.
    bool save_compiled_obj_file = true;
  };

  // Creates a new execution engine by compiling the provided MLIR module to
  // a native function using LLVM ORC stack.
  static llvm::Expected<std::unique_ptr<ExecutionEngine>> CreateFromSource(
      mlir::ModuleOp module, llvm::StringRef entrypoint, JitOptions options);

  //------------------------------------------------------------------------- //
  // Options for creating execution engine from an AOT compiled object file.
  //------------------------------------------------------------------------- //

  struct AotOptions {
    // User-provided memory mapper for allocating memory for executables.
    llvm::SectionMemoryManager::MemoryMapper *section_memory_mapper = nullptr;

    // User-provided bindings for symbols.
    SymbolsBinding symbols_binding = nullptr;

    // Notify the llvm's global GDB notifications listener.
    bool enable_gdb_listener = true;

    // Notify the llvm's global Perf notifications listener.
    bool enable_perf_listener = true;
  };

  // Creates a new execution engine by loading AOT compiled JitRt executable
  // object file.
  static llvm::Expected<std::unique_ptr<ExecutionEngine>> CreateFromObjFile(
      std::unique_ptr<llvm::MemoryBuffer>, llvm::StringRef entrypoint,
      AotOptions options);

  //------------------------------------------------------------------------- //

  // Returns a pointer to the JitRt entrypoint function.
  EntrypointFunctionPtr entrypoint() const { return entrypoint_ptr_; }

  // Return a memory buffer with a object file behind this execution engine. Can
  // be null if execution engine didn't save the compiled object file.
  std::unique_ptr<llvm::MemoryBuffer> obj_file() const;

 private:
  ExecutionEngine(bool enable_gdb_listener, bool enable_perf_listener);

  // We build execution engine on top of the ORC LLJIT API, which owns all
  // compiled/loaded object files and does the linking at run time.
  std::unique_ptr<llvm::orc::LLJIT> jit_;

  // Pointer to a resolved entrypoint function.
  EntrypointFunctionPtr entrypoint_ptr_ = nullptr;

  // Object file that has the compiled entrypoint function. Can be null.
  std::unique_ptr<llvm::MemoryBuffer> obj_file_;

  llvm::JITEventListener *gdb_listener_ = nullptr;
  llvm::JITEventListener *perf_listener_ = nullptr;
};

}  // namespace jitrt
}  // namespace tfrt

#endif  // TFRT_BACKENDS_JITRT_EXECUTION_ENGINE_H_
