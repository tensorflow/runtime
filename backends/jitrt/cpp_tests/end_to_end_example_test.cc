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

#include <memory>
#include <utility>
#include <vector>

#include "benchmark/benchmark.h"
#include "gtest/gtest.h"
#include "llvm/ADT/SmallVector.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/jitrt/jitrt.h"
#include "tfrt/jitrt/jitrt_compiler.h"
#include "tfrt/support/logging.h"

namespace tfrt {
namespace jitrt {
namespace {

using llvm::SmallVector;
using mlir::LogicalResult;
using mlir::success;

// Features supported in JitRt but missing in this example:
//   1. Returning results from the compiled function.
//   2. Launching async tasks.
//   3. Returning async results from the compiled function.

// TODO(ezhulenev): Show all the features supported by JitRt?

// JitRt input program can be defined in arbitrary dialects, the only
// requirement is that the user must pass a pipeline that can lower the input
// program to the LLVM dialect (see `create_compilation_pipeline` option below).
static const char* mlir_module = R"(
  module {
    // Declare your own "runtime" intrinsics library in the compiled module.
    func.func private @my.runtime.intrinsic()
      attributes { rt.custom_call = "my.runtime.intrinsic" }

    func.func @compute(%arg0: memref<?xf32>, %arg1: memref<?xf32>) {

      // Pass attributes to the runtime intrinsics.
      func.call @my.runtime.intrinsic() { api_version = 1 : i32 } : () -> ()

      // Host computation on buffers written as linalg op.
      linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      }
      ins(%arg0: memref<?xf32>) outs(%arg1 : memref<?xf32>) {
        ^bb0(%in: f32, %out: f32):
          %2 = arith.addf %in, %in : f32
          linalg.yield %2 : f32
      }

      func.return
    }
  })";

static const char* entrypoint = "compute";

// Context structure that encapsulats all the state that has to be available
// to your runtime intrinsics.
struct MyRuntimeContext {};

// Implement your runtime intrinsic as a regular C++ function.
static LogicalResult MyRuntimeIntrinsic(MyRuntimeContext* ctx,
                                        int32_t api_version) {
  return success();
}

// Register your runtime support library with JitRt as custom calls.
void RegisterMyRuntimeIntrinsics(CustomCallRegistry* registry) {
  registry->Register(CustomCall::Bind("my.runtime.intrinsic")
                         .UserData<MyRuntimeContext*>()
                         .Attr<int32_t>("api_version")
                         .To(MyRuntimeIntrinsic));
}

// Static registration with the JitRt global registry.
JITRT_STATIC_CUSTOM_CALL_REGISTRATION(RegisterMyRuntimeIntrinsics);

TEST(EndToEndExampleTest, CompiledAndExecute) {
  // Step by step guide for compiling and executing your programs on top of the
  // JitRt library.

  // ------------------------------------------------------------------------ //
  // 1. Set up options for the JitRt executable compilation/recompilation.
  // ------------------------------------------------------------------------ //
  CompilationOptions opts;

  // Do not recompile executable for different symbolic shapes, because we do
  // not set up specialization pipeline that can propagate symbolic shapes from
  // operands to operations in the function body.
  //
  // For example for Tensorflow dialect shape propagation can canonicalize away
  // some of the broadcasts or reshapes.
  opts.specialization = CompilationOptions::Specialization::kDisabled;

  // Define what dialects are supported in the input IR module. If you have your
  // own custom dialects in the input IR you must pass a callback that registers
  // all the dialects that are considered legal for your inpuyt program.
  opts.register_dialects = RegisterDefaultJitRtDialects;

  // ------------------------------------------------------------------------ //
  // 2. Set up compilation pipeline that lowers input module to LLVM.
  // ------------------------------------------------------------------------ //

  // We rely on a default JitRt compilation pipeline, because we do not have
  // any non "standard" dialects in this test.
  opts.create_compilation_pipeline = [&](mlir::PassManager& pm) {
    CompilationPipelineOptions copts;
    CreateDefaultJitRtCompilationPipeline(pm, copts);
  };

  // If your input IR requires specialization, you'll also need to define the
  // `opts.create_compilation_pipeline` callback

  // ------------------------------------------------------------------------ //
  // 3. Instantiate JitExecutable from the input MLIR source.
  // ------------------------------------------------------------------------ //

  // JitExecutable does compilation/recompilation from the input source to the
  // Executable artifact. In this particular example we always rely on default
  // executable, because we didn't provide a specialization pipeline.
  llvm::Expected<JitExecutable> jit_executable =
      JitExecutable::Instantiate(mlir_module, entrypoint, opts);
  if (auto err = jit_executable.takeError()) ASSERT_FALSE(err) << StrCat(err);

  AsyncValuePtr<Executable> executable = jit_executable->DefaultExecutable();
  Await(executable.value());  // make sure executable is available

  // ------------------------------------------------------------------------ //
  // 4. Prepare input data for the compiled program.
  // ------------------------------------------------------------------------ //

  // JitRt Executable knows how to pass MemrefDesc to the compiled program
  // according to the MLIR C ABI (memrefs passed as `StridedMemRefType` struct).
  //
  // For "real" programs instead of vectors we should have tensors flying
  // around.

  // Allocate storage for arguments.
  std::vector<float> arg0 = {1.0, 2.0, 3.0, 4.0};
  std::vector<float> arg1(4, 0.0);

  // Prepare memref descriptors for the executable.
  llvm::SmallVector<MemrefDesc> args;
  args.emplace_back(DType::F32, arg0.data(), 0, 4, 1);
  args.emplace_back(DType::F32, arg1.data(), 0, 4, 1);

  // ------------------------------------------------------------------------ //
  // 5. Prepare options for executing the JitRt executable.
  // ------------------------------------------------------------------------ //

  Executable::ExecuteOpts execute_opts;

  // We don't expect to launch any async tasks in this example.
  execute_opts.async_task_runner =
      reinterpret_cast<jitrt::AsyncTaskRunner*>(0XDEADBEEF);

  // Pass runtime context to all runtime intrinsics handlers.
  MyRuntimeContext runtime_context;

  CustomCall::UserData user_data;
  user_data.insert(&runtime_context);
  execute_opts.custom_call_data = &user_data;

  // ------------------------------------------------------------------------ //
  // 6. Define how to convert returned values back to C++ objects.
  // ------------------------------------------------------------------------ //

  // TODO(ezhulenev): This is a very boring example that doesn't return
  // anything. Make it more interesting if someone is curious to see how
  // returning values will look like.
  NoOpReturnValueConverter converter;

  // ------------------------------------------------------------------------ //
  // 7. Call JitRt executable with the prepared operands.
  // ------------------------------------------------------------------------ //

  // Execute Jit compiled executable.
  ASSERT_FALSE(executable->Execute(args, converter, execute_opts));

  // Check that compute function produced the correct results.
  auto expected = llvm::map_range(arg0, [](float a) { return a + a; });
  EXPECT_EQ(arg1, std::vector<float>(expected.begin(), expected.end()));

  // ------------------------------------------------------------------------ //
  // 8. Saving/Restoring JitRt executable to/from object file.
  // ------------------------------------------------------------------------ //

  // See `aot_compilation_test` for an example of serializing JitRt executable
  // as an object file.
}

}  // namespace
}  // namespace jitrt
}  // namespace tfrt
