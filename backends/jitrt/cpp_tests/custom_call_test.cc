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

#include "third_party/tensorflow/compiler/xla/runtime/custom_call.h"

#include <utility>

#include "benchmark/benchmark.h"
#include "gtest/gtest.h"
#include "mlir/Support/LogicalResult.h"
#include "tfrt/dtype/dtype.h"
#include "tfrt/jitrt/custom_calls/custom_call_testlib.h"
#include "tfrt/jitrt/jitrt_compiler.h"
#include "third_party/tensorflow/compiler/xla/mlir/runtime/transforms/compiler.h"
#include "third_party/tensorflow/compiler/xla/runtime/arguments.h"
#include "third_party/tensorflow/compiler/xla/runtime/diagnostics.h"
#include "third_party/tensorflow/compiler/xla/runtime/executable.h"
#include "third_party/tensorflow/compiler/xla/runtime/execution_engine.h"
#include "third_party/tensorflow/compiler/xla/runtime/jit_executable.h"

// Forward declare types enabling compiled kernel <-> runtime integration.
namespace xla {
namespace runtime {
struct ExecutionContext;
}  // namespace runtime
}  // namespace xla

namespace tfrt {
namespace jitrt {

using namespace xla::runtime;  // NOLINT

using llvm::SmallVector;
using llvm::StringRef;
using llvm::orc::SymbolMap;

using mlir::success;

using RuntimeChecks = CustomCall::RuntimeChecks;

// Give short aliases to enums for benchmarks pretty printing.
static constexpr RuntimeChecks all = RuntimeChecks::kDefault;
static constexpr RuntimeChecks less = RuntimeChecks::kLess;
static constexpr RuntimeChecks none = RuntimeChecks::kNone;

// -------------------------------------------------------------------------- //
// A set of benchmarks to measure custom call overheads.
// -------------------------------------------------------------------------- //

static void BenchmarkCustomCall(benchmark::State& state, StringRef module,
                                ExecutionEngine::SymbolsBinding symbols_binding,
                                SmallVector<MemrefDesc> operands = {}) {
  // Use the default JitRt compilation pipeline to compile the executable.
  JitExecutable::Options opts;

  opts.specialization = JitExecutable::Specialization::kDisabled;
  opts.compiler.symbols_binding = symbols_binding;

  opts.compiler.register_dialects =
      [&](xla::runtime::DialectRegistry& dialects) {
        dialects->insert<TestlibDialect>();
        RegisterDefaultJitRtDialects(dialects);
      };

  opts.compiler.create_compilation_pipeline =
      [&](xla::runtime::PassManager& passes) {
        CompilationPipelineOptions copts;
        copts.populate_type_id_names = PopulateCustomCallTypeIdNames;
        copts.populate_attr_encodings = PopulateCustomCallAttrEncoding;
        CreateDefaultJitRtCompilationPipeline(passes, copts);
      };

  // Get the default executable (it must be always available).
  absl::StatusOr<JitExecutable> jit_executable =
      JitExecutable::Instantiate(module, "compute", opts);
  if (!jit_executable.ok())
    TFRT_LOG(FATAL) << jit_executable.status().message();

  AsyncValuePtr<Executable> executable = jit_executable->DefaultExecutable();
  if (executable.IsError()) TFRT_LOG(FATAL) << executable.GetError().message();

  // Prepare the call frame outside of a benchmark loop.
  Executable::CallFrame call_frame;
  if (auto initialized = executable->InitializeCallFrame(operands, &call_frame);
      !initialized.ok())
    TFRT_LOG(FATAL) << initialized.message();

  Executable::ExecuteOpts execute_opts;
  // We don't expect to launch any async tasks in this test.
  execute_opts.async_task_runner =
      reinterpret_cast<jitrt::AsyncTaskRunner*>(0XDEADBEEF);

  // Dump all emitted diagnostics to the llvm::errs() by default.
  DiagnosticEngine diagnostic_engine;
  execute_opts.diagnostic_engine = &diagnostic_engine;

  for (auto _ : state) {
    call_frame.args[0] = nullptr;  // reset kernel context
    executable->Execute(call_frame, execute_opts);
    if (call_frame.is_error) TFRT_LOG(FATAL) << "Failed to execute the kernel";
  }
}

static SmallVector<MemrefDesc> GetFakeMemrefs(
    SmallVector<ArrayRef<int64_t>> shapes) {
  SmallVector<MemrefDesc> memrefs;
  memrefs.reserve(shapes.size());

  for (auto& shape : shapes) {
    // Data type of the fake memrefs doesn't matter.
    MemrefDesc desc(xla::PrimitiveType::F32, nullptr, 0, shape,
                    shape /* fake strides */);
    memrefs.push_back(std::move(desc));
  }

  return memrefs;
}

template <typename SymPtr>
static ExecutionEngine::SymbolsBinding Bind(StringRef name, SymPtr symbol_ptr) {
  return xla::runtime::ToSymbolsBinding(
      [=](DirectCustomCallRegistry& custom_calls) {
        custom_calls.Register(name, symbol_ptr);
      },
      PopulateCustomCallTypeIdNames);
}

// -------------------------------------------------------------------------- //
// Custom call with a single i32 argument.
// -------------------------------------------------------------------------- //

static const char* custom_call_i32x1 = R"(
    func.func private @custom_call(%arg0: i32)
      attributes { rt.custom_call = "testlib.custom_call" }

    func.func @compute() {
      %0 = arith.constant 0 : i32
      func.call @custom_call(%0) : (i32) -> ()
      func.return
    }
  )";

template <RuntimeChecks checks>
static bool I32X1(xla::runtime::ExecutionContext* ctx, void** args,
                  void** attrs, void** rets) {
  static auto* handler = CustomCall::Bind("testlib.custom_call")
                             .Arg<int32_t>()
                             .To<checks>([](int32_t arg0) { return success(); })
                             .release();
  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

template <RuntimeChecks checks>
static void BM_I32X1(benchmark::State& state) {
  BenchmarkCustomCall(state, custom_call_i32x1,
                      Bind("testlib.custom_call", &I32X1<checks>));
}

BENCHMARK(BM_I32X1<all>);
BENCHMARK(BM_I32X1<none>);

// -------------------------------------------------------------------------- //
// Custom call with 12 i32 arguments.
// -------------------------------------------------------------------------- //

static const char* custom_call_i32x12 = R"(
    func.func private @custom_call(%arg0: i32, %arg1: i32, %arg2: i32,
                                   %arg3: i32, %arg4: i32, %arg5: i32,
                                   %arg6: i32, %arg7: i32, %arg8: i32,
                                   %arg9: i32, %arg10: i32, %arg11: i32)
      attributes { rt.custom_call = "testlib.custom_call" }

    func.func @compute() {
      %0 = arith.constant 0 : i32
      %1 = arith.constant 1 : i32
      %2 = arith.constant 2 : i32
      %3 = arith.constant 3 : i32
      %4 = arith.constant 4 : i32
      %5 = arith.constant 5 : i32
      %6 = arith.constant 6 : i32
      %7 = arith.constant 7 : i32
      %8 = arith.constant 8 : i32
      %9 = arith.constant 9 : i32
      %10 = arith.constant 10 : i32
      %11 = arith.constant 11 : i32
      func.call @custom_call(%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11)
        : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      func.return
    }
  )";

template <RuntimeChecks checks>
static bool I32X12(xla::runtime::ExecutionContext* ctx, void** args,
                   void** attrs, void** rets) {
  static auto* handler =
      CustomCall::Bind("testlib.custom_call")
          .Arg<int32_t>()
          .Arg<int32_t>()
          .Arg<int32_t>()
          .Arg<int32_t>()
          .Arg<int32_t>()
          .Arg<int32_t>()
          .Arg<int32_t>()
          .Arg<int32_t>()
          .Arg<int32_t>()
          .Arg<int32_t>()
          .Arg<int32_t>()
          .Arg<int32_t>()
          .To<checks>([](int32_t arg0, int32_t arg1, int32_t arg2, int32_t arg3,
                         int32_t arg4, int32_t arg5, int32_t arg6, int32_t arg7,
                         int32_t arg8, int32_t arg9, int32_t arg10,
                         int32_t arg11) { return success(); })
          .release();
  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

template <RuntimeChecks checks>
static void BM_I32X12(benchmark::State& state) {
  BenchmarkCustomCall(state, custom_call_i32x12,
                      Bind("testlib.custom_call", &I32X12<checks>));
}

BENCHMARK(BM_I32X12<all>);
BENCHMARK(BM_I32X12<none>);

// -------------------------------------------------------------------------- //
// Custom call with a single memref argument.
// -------------------------------------------------------------------------- //

static const char* custom_call_memrefx1 = R"(
    func.func private @custom_call(%arg0: memref<?x?xf32>)
      attributes { rt.custom_call = "testlib.custom_call" }

    func.func @compute(%arg0 : memref<?x?xf32>) {
      func.call @custom_call(%arg0) : (memref<?x?xf32>) -> ()
      func.return
    }
  )";

template <RuntimeChecks checks, typename MemrefType>
static bool MemrefX1(xla::runtime::ExecutionContext* ctx, void** args,
                     void** attrs, void** rets) {
  static auto* handler = CustomCall::Bind("testlib.custom_call")
                             .Arg<MemrefType>()
                             .template To<checks>([](MemrefType arg0) {
                               benchmark::DoNotOptimize(arg0);
                               return success();
                             })
                             .release();
  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

template <RuntimeChecks checks>
static void BM_FlatMemrefX1(benchmark::State& state) {
  BenchmarkCustomCall(
      state, custom_call_memrefx1,
      Bind("testlib.custom_call", &MemrefX1<checks, FlatMemrefView>),
      GetFakeMemrefs({{10, 10}}));
}

template <RuntimeChecks checks>
static void BM_MemrefX1(benchmark::State& state) {
  BenchmarkCustomCall(
      state, custom_call_memrefx1,
      Bind("testlib.custom_call", &MemrefX1<checks, MemrefView>),
      GetFakeMemrefs({{10, 10}}));
}

template <RuntimeChecks checks>
static void BM_StridedMemrefX1(benchmark::State& state) {
  BenchmarkCustomCall(
      state, custom_call_memrefx1,
      Bind("testlib.custom_call", &MemrefX1<checks, StridedMemrefView>),
      GetFakeMemrefs({{10, 10}}));
}

BENCHMARK(BM_FlatMemrefX1<all>);
BENCHMARK(BM_FlatMemrefX1<none>);

BENCHMARK(BM_MemrefX1<all>);
BENCHMARK(BM_MemrefX1<none>);

BENCHMARK(BM_StridedMemrefX1<all>);
BENCHMARK(BM_StridedMemrefX1<none>);

// -------------------------------------------------------------------------- //
// Custom call with 12 memref arguments.
// -------------------------------------------------------------------------- //

static const char* custom_call_memrefx12 = R"(
    func.func private @custom_call(
      %arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>,
      %arg3: memref<?x?xf32>, %arg4: memref<?x?xf32>, %arg5: memref<?x?xf32>,
      %arg6: memref<?x?xf32>, %arg7: memref<?x?xf32>, %arg8: memref<?x?xf32>,
      %arg9: memref<?x?xf32>, %arg10: memref<?x?xf32>, %arg11: memref<?x?xf32>
    ) attributes { rt.custom_call = "testlib.custom_call" }

    func.func @compute(
      %arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>,
      %arg3: memref<?x?xf32>, %arg4: memref<?x?xf32>, %arg5: memref<?x?xf32>,
      %arg6: memref<?x?xf32>, %arg7: memref<?x?xf32>, %arg8: memref<?x?xf32>,
      %arg9: memref<?x?xf32>, %arg10: memref<?x?xf32>, %arg11: memref<?x?xf32>
    ) {
      func.call @custom_call(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6,
                             %arg7, %arg8, %arg9, %arg10, %arg11)
        : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>,
           memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>,
           memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>
          ) -> ()
      func.return
    }
  )";

template <RuntimeChecks checks, typename MemrefType>
static bool MemrefX12(xla::runtime::ExecutionContext* ctx, void** args,
                      void** attrs, void** rets) {
  static auto* handler =
      CustomCall::Bind("testlib.custom_call")
          .template Arg<MemrefType>()
          .template Arg<MemrefType>()
          .template Arg<MemrefType>()
          .template Arg<MemrefType>()
          .template Arg<MemrefType>()
          .template Arg<MemrefType>()
          .template Arg<MemrefType>()
          .template Arg<MemrefType>()
          .template Arg<MemrefType>()
          .template Arg<MemrefType>()
          .template Arg<MemrefType>()
          .template Arg<MemrefType>()
          .template To<checks>(
              [](MemrefType arg0, MemrefType arg1, MemrefType arg2,
                 MemrefType arg3, MemrefType arg4, MemrefType arg5,
                 MemrefType arg6, MemrefType arg7, MemrefType arg8,
                 MemrefType arg9, MemrefType arg10, MemrefType arg11) {
                benchmark::DoNotOptimize(arg0);
                benchmark::DoNotOptimize(arg1);
                benchmark::DoNotOptimize(arg2);
                benchmark::DoNotOptimize(arg3);
                benchmark::DoNotOptimize(arg4);
                benchmark::DoNotOptimize(arg5);
                benchmark::DoNotOptimize(arg6);
                benchmark::DoNotOptimize(arg7);
                benchmark::DoNotOptimize(arg8);
                benchmark::DoNotOptimize(arg9);
                benchmark::DoNotOptimize(arg10);
                benchmark::DoNotOptimize(arg11);
                return success();
              })
          .release();
  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

static SmallVector<MemrefDesc> FakeMemrefsX12() {
  return GetFakeMemrefs({{10, 10},
                         {10, 10},
                         {10, 10},
                         {10, 10},
                         {10, 10},
                         {10, 10},
                         {10, 10},
                         {10, 10},
                         {10, 10},
                         {10, 10},
                         {10, 10},
                         {10, 10}});
}

template <RuntimeChecks checks>
static void BM_FlatMemrefX12(benchmark::State& state) {
  BenchmarkCustomCall(
      state, custom_call_memrefx12,
      Bind("testlib.custom_call", &MemrefX12<checks, FlatMemrefView>),
      FakeMemrefsX12());
}

template <RuntimeChecks checks>
static void BM_MemrefX12(benchmark::State& state) {
  BenchmarkCustomCall(
      state, custom_call_memrefx12,
      Bind("testlib.custom_call", &MemrefX12<checks, MemrefView>),
      FakeMemrefsX12());
}

template <RuntimeChecks checks>
static void BM_StridedMemrefX12(benchmark::State& state) {
  BenchmarkCustomCall(
      state, custom_call_memrefx12,
      Bind("testlib.custom_call", &MemrefX12<checks, StridedMemrefView>),
      FakeMemrefsX12());
}

BENCHMARK(BM_FlatMemrefX12<all>);
BENCHMARK(BM_FlatMemrefX12<none>);

BENCHMARK(BM_MemrefX12<all>);
BENCHMARK(BM_MemrefX12<none>);

BENCHMARK(BM_StridedMemrefX12<all>);
BENCHMARK(BM_StridedMemrefX12<none>);

// -------------------------------------------------------------------------- //
// Custom call with 12 i32 attributes.
// -------------------------------------------------------------------------- //

static const char* custom_call_i32_attrx12 = R"(
    func.func private @custom_call()
      attributes { rt.custom_call = "testlib.custom_call" }

    func.func @compute() {
      func.call @custom_call()
       { "attribute0" = 0 : i32,
         "attribute1" = 1 : i32,
         "attribute2" = 2 : i32,
         "attribute3" = 3 : i32,
         "attribute4" = 4 : i32,
         "attribute5" = 5 : i32,
         "attribute6" = 6 : i32,
         "attribute7" = 7 : i32,
         "attribute8" = 8 : i32,
         "attribute9" = 9 : i32,
         "attribute10" = 10 : i32,
         "attribute11" = 11 : i32,
         "attribute12" = 12 : i32
       } : () -> ()
      func.return
    }
  )";

template <RuntimeChecks checks>
static bool I32AttrX12(xla::runtime::ExecutionContext* ctx, void** args,
                       void** attrs, void** rets) {
  static auto* handler =
      CustomCall::Bind("testlib.custom_call")
          .Attr<int32_t>("attribute0")
          .Attr<int32_t>("attribute1")
          .Attr<int32_t>("attribute2")
          .Attr<int32_t>("attribute3")
          .Attr<int32_t>("attribute4")
          .Attr<int32_t>("attribute5")
          .Attr<int32_t>("attribute6")
          .Attr<int32_t>("attribute7")
          .Attr<int32_t>("attribute8")
          .Attr<int32_t>("attribute9")
          .Attr<int32_t>("attribute10")
          .Attr<int32_t>("attribute11")
          .To<checks>([](int32_t arg0, int32_t arg1, int32_t arg2, int32_t arg3,
                         int32_t arg4, int32_t arg5, int32_t arg6, int32_t arg7,
                         int32_t arg8, int32_t arg9, int32_t arg10,
                         int32_t arg11) { return success(); })
          .release();
  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

template <RuntimeChecks checks>
static void BM_I32AttrX12(benchmark::State& state) {
  BenchmarkCustomCall(state, custom_call_i32_attrx12,
                      Bind("testlib.custom_call", &I32AttrX12<checks>));
}

BENCHMARK(BM_I32AttrX12<all>);
BENCHMARK(BM_I32AttrX12<less>);
BENCHMARK(BM_I32AttrX12<none>);

// -------------------------------------------------------------------------- //
// Custom call with a single user-defined attribute argument.
// -------------------------------------------------------------------------- //

static const char* custom_call_pair_of_dimsx1 = R"(
    func.func private @custom_call()
      attributes { rt.custom_call = "testlib.custom_call" }

    func.func @compute() {
      func.call @custom_call() {
        dims = #testlib.pair_of_dims<2, [1, 1], [2, 2]>
      }: () -> ()
      func.return
    }
  )";

template <RuntimeChecks checks>
static bool PairOfDimsX1(xla::runtime::ExecutionContext* ctx, void** args,
                         void** attrs, void** rets) {
  static auto* handler = CustomCall::Bind("testlib.custom_call")
                             .Attr<RuntimePairOfDims>("dims")
                             .To<checks>([](RuntimePairOfDims dims) {
                               benchmark::DoNotOptimize(dims);
                               return success();
                             })
                             .release();
  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

template <RuntimeChecks checks>
static void BM_PairOfDimsX1(benchmark::State& state) {
  BenchmarkCustomCall(state, custom_call_pair_of_dimsx1,
                      Bind("testlib.custom_call", &PairOfDimsX1<checks>));
}

BENCHMARK(BM_PairOfDimsX1<all>);
BENCHMARK(BM_PairOfDimsX1<less>);
BENCHMARK(BM_PairOfDimsX1<none>);

}  // namespace jitrt
}  // namespace tfrt
