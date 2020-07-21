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

//===- bef_executor_driver.cc - Library for bef_executor test driver ------===//
//
// This file implements the test driver library for the bef executor. It opens
// up a given mlir file and then runs it with a host executor.
//
//===----------------------------------------------------------------------===//
#include "tfrt/bef_executor_driver/bef_executor_driver.h"

#include <limits>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm_derived/Support/raw_ostream.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"
#include "tfrt/bef_executor/bef_file.h"
#include "tfrt/core_runtime/core_runtime.h"
#include "tfrt/core_runtime/tensor_handle.h"
#include "tfrt/host_context/async_value.h"
#include "tfrt/host_context/concurrent_work_queue.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/host_context/function.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/kernel_registry.h"
#include "tfrt/host_context/location.h"
#include "tfrt/host_context/profiled_allocator.h"
#include "tfrt/host_context/resource_context.h"
#include "tfrt/host_context/value.h"
#include "tfrt/metrics/metrics_api.h"
#include "tfrt/support/mutex.h"
#include "tfrt/support/string_util.h"
#include "tfrt/tracing/tracing.h"

namespace tfrt {
// TODO(jingdong): Change const Function* to const Functino& in funciton
// argument to conform to the style guide.
static void RunBefFunction(HostContext* host, const Function* function);

int RunBefExecutor(const RunBefConfig& run_config) {
  TFRT_TRACE_SCOPE("Bef Executor");
  static auto* version_metric =
      metrics::NewGauge<std::string>("/tensorflow/runtime/version");
  static std::once_flag initialized;
  std::call_once(initialized, [] { version_metric->SetValue("TFRT_V0"); });

  // Set up the input file.
  std::string error_message;
  auto file = mlir::openInputFile(run_config.input_filename, &error_message);
  if (!file) {
    llvm::errs() << error_message << "\n";
    return 1;
  }

  // Tell source_mgr about this buffer, which is what the parser will pick up.
  llvm::SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());

  // Parse the input file.
  mlir::MLIRContext context;
  mlir::SourceMgrDiagnosticVerifierHandler source_mgr_handler(source_mgr,
                                                              &context);

  auto decoded_diagnostic_handler = [&](const DecodedDiagnostic& diag) {
    std::string message = "runtime error: " + diag.message;

    auto decoded_loc = diag.location;
    if (decoded_loc) {
      auto loc =
          mlir::FileLineColLoc::get(decoded_loc->filename, decoded_loc->line,
                                    decoded_loc->column, &context);
      emitError(loc) << message;
    } else {
      auto loc = mlir::FileLineColLoc::get("", 0, 0, &context);
      emitError(loc) << message;
    }
  };

  assert(GetNumReferenceCountedObjects() == 0 &&
         "We have reference-counted objects before we started to do anything");

  std::unique_ptr<HostAllocator> host_allocator;
  switch (run_config.host_allocator_type) {
    case HostAllocatorType::kMalloc:
      host_allocator = CreateMallocAllocator();
      tfrt::outs() << "Choosing malloc.\n";
      break;
    case HostAllocatorType::kTestFixedSizeMalloc:
      host_allocator = tfrt::CreateFixedSizeAllocator();
      tfrt::outs() << "Choosing fixed size malloc.\n";
      break;
    case HostAllocatorType::kProfiledMalloc:
      host_allocator = CreateMallocAllocator();
      host_allocator = CreateProfiledAllocator(std::move(host_allocator));
      tfrt::outs() << "Choosing profiled allocator based on malloc.\n";
      break;
    case HostAllocatorType::kLeakCheckMalloc:
      host_allocator = CreateMallocAllocator();
      host_allocator = CreateLeakCheckAllocator(std::move(host_allocator));
      tfrt::outs() << "Choosing memory leak check allocator.\n";
  }
  tfrt::outs().flush();

  // Dig the bytes out of the SourceMgr.
  auto buffer =
      source_mgr.getMemoryBuffer(source_mgr.getMainFileID())->getBuffer();
  auto buffer_arr = llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t*>(buffer.data()), buffer.size());

  std::unique_ptr<ConcurrentWorkQueue> work_queue =
      CreateWorkQueue(run_config.work_queue_type);
  if (work_queue == nullptr) {
    llvm::errs() << run_config.program_name
                 << ": couldn't create work queue type "
                 << run_config.work_queue_type << "\n";
    return 1;
  }
  tfrt::outs() << "Choosing " << work_queue->name() << " work queue.\n";
  tfrt::outs().flush();

  assert(AsyncValue::GetNumAsyncValueInstances() == 0 &&
         "We have async values allocated before we started to do anything");
  auto async_value_guard = llvm::make_scope_exit([]() {
    assert(AsyncValue::GetNumAsyncValueInstances() == 0 &&
           "All async values should be cleaned up at the end");
    assert(GetNumReferenceCountedObjects() == 0 &&
           "We have live reference-counted objects before exit.");
  });

  auto core_rt =
      CoreRuntime::Create(decoded_diagnostic_handler, std::move(host_allocator),
                          std::move(work_queue), run_config.devices);
  if (!core_rt) {
    llvm::errs() << core_rt.takeError();
    return 1;
  }

  auto* host = core_rt.get()->GetHostContext();

  // If there are any libraries specified, load them and see if they have a
  // kernel registration function.
  for (const auto& lib_name : run_config.shared_libs) {
    std::string err;
    auto dyn_lib =
        llvm::sys::DynamicLibrary::getPermanentLibrary(lib_name.c_str(), &err);
    if (!dyn_lib.isValid()) {
      llvm::errs() << run_config.program_name << ": couldn't load library "
                   << err << "\n";
      return 1;
    }

    // The library should specify a kernel registration entrypoint.
    if (auto kernel_reg = dyn_lib.SearchForAddressOfSymbol("RegisterKernels")) {
      reinterpret_cast<void (*)(KernelRegistry*)>(kernel_reg)(
          host->GetMutableRegistry());
    }
  }

  auto bef(BEFFile::Open(buffer_arr, host->GetMutableRegistry(),
                         decoded_diagnostic_handler, host->allocator()));

  if (!bef) {
    return mlir::failed(source_mgr_handler.verify());
  }

  SmallVector<const Function*, 8> function_list;

  if (run_config.functions.empty()) {
    // No functions specified in the command line. Try to run all functions in
    // the input BEF file.
    bef->GetFunctionList(&function_list);
  } else {
    function_list.reserve(run_config.functions.size());

    for (auto& fn_name : run_config.functions) {
      auto* fn = bef->GetFunction(fn_name);

      if (!fn) {
        llvm::errs() << run_config.program_name << ": couldn't find function "
                     << fn_name << "\n";
        return 1;
      }
      function_list.push_back(fn);
    }
  }

  // Run the init function first if exists.
  auto init_function = bef->GetFunction(run_config.init_function);

  if (init_function) {
    RunBefFunction(host, init_function);
  }

  // Loop over each of the functions, running each as a standalone testcase.
  for (auto* fn : function_list) {
    if (fn != init_function) {
      RunBefFunction(host, fn);
    }
  }

  bef.reset();
  // Verify the diagnostic handler to make sure that each of the diagnostics
  // matched.
  return mlir::failed(source_mgr_handler.verify());
}

// Utility function to print the result. ValueType is either Value* or
// RCReference<AsyncValue>.
template <typename ValueType>
static void PrintResult(const TypeName& type_name, const ValueType& result) {
  if (type_name.GetName() == "i1") {
    tfrt::outs() << result->template get<bool>();
  } else if (type_name.GetName() == "i32") {
    tfrt::outs() << result->template get<int32_t>();
  } else if (type_name.GetName() == "i64") {
    tfrt::outs() << result->template get<int64_t>();
  } else if (type_name.GetName() == "f32") {
    tfrt::outs() << result->template get<float>();
  } else if (type_name.GetName() == "f64") {
    tfrt::outs() << result->template get<double>();
  } else {
    tfrt::outs() << type_name.GetName() << " value";
  }
}

static void RunSyncBefFunctionHelper(HostContext* host,
                                     const Function* function) {
  TFRT_TRACE_KERNEL_SCOPE(StrCat("Function: ", function->name()));

  llvm::SmallVector<Value, 4> results;
  results.resize(function->result_types().size());

  llvm::SmallVector<Value*, 4> result_ptrs;
  for (auto& value : results) {
    result_ptrs.emplace_back(&value);
  }

  // Add a ResourceContext ops/kernels to access resources. Shared across
  // kernels in this function, but not across functions.
  tfrt::ResourceContext resource_context;
  // If any kernel calls RequestContext::Cancel, it will create an extra async
  // value that's stored inside RequestContext which is destroyed only when
  // RequestContext is destroyed.
  RCReference<RequestContext> req_ctx =
      tfrt::RequestContext::Create(host, &resource_context);
  ExecutionContext exec_ctx{std::move(req_ctx)};

  auto error = ExecuteSyncBEFFunction(*function, exec_ctx, /*arguments=*/{},
                                      result_ptrs);

  // Go ahead and print out the function results that we know about.
  if (error) {
    tfrt::outs() << "'" << function->name() << "' returned ";
    tfrt::outs() << "<<error: " << error << ">>\n";
    tfrt::outs().flush();
  } else if (!results.empty()) {
    tfrt::outs() << "'" << function->name() << "' returned ";
    auto result_types = function->result_types();

    for (int i = 0, e = results.size(); i != e; ++i) {
      auto type_name = result_types[i];

      PrintResult(type_name, &results[i]);

      // Print comma except for the last one.
      if (i != results.size() - 1) {
        tfrt::outs() << ',';
      }
    }

    tfrt::outs() << '\n';
    tfrt::outs().flush();
  }
}

static void RunAsyncBefFunctionHelper(HostContext* host,
                                      const Function* function) {
  TFRT_TRACE_KERNEL_SCOPE(StrCat("Function: ", function->name()));

  // Kick off an execution of the function body.
  llvm::SmallVector<RCReference<AsyncValue>, 4> results;
  results.resize(function->result_types().size());

  // Add a ResourceContext ops/kernels to access resources. Shared across
  // kernels in this function, but not across functions.
  tfrt::ResourceContext resource_context;
  // If any kernel calls RequestContext::Cancel, it will create an extra async
  // value that's stored inside RequestContext which is destroyed only when
  // RequestContext is destroyed.
  RCReference<RequestContext> req_ctx =
      tfrt::RequestContext::Create(host, &resource_context);
  ExecutionContext exec_ctx{std::move(req_ctx)};

  function->Execute(exec_ctx, /*arguments=*/{}, results);

  // Block until the function results are fully resolved.
  host->Await(results);

  // Go ahead and print out the function results that we know about.
  if (!results.empty()) {
    tfrt::outs() << "'" << function->name() << "' returned ";
    auto result_types = function->result_types();

    for (int i = 0, e = results.size(); i != e; ++i) {
      auto type_name = result_types[i];

      if (auto* error = results[i]->GetErrorIfPresent()) {
        tfrt::outs() << "<<error: " << error->message << ">>";
      } else {
        PrintResult(type_name, results[i]);
      }

      // Print comma except for the last one.
      if (i != results.size() - 1) {
        tfrt::outs() << ',';
      }
    }

    tfrt::outs() << '\n';
    tfrt::outs().flush();
  }

  // In this test driver, we want to make sure that every function completes
  // all execution before moving on to the next one.  This makes the leak
  // checker work better in the face of side effecting kernels that aren't
  // properly chained together (which is useful for testing).
  host->Quiesce();
}

static void RunBefFunction(HostContext* host, const Function* function) {
  // If the function takes arguments, then we can't run it from this driver.
  if (!function->argument_types().empty()) {
    tfrt::outs() << "--- Not running '" << function->name()
                 << "' because it has arguments.\n";
    tfrt::outs().flush();
    return;
  }

  // Skip anonymous functions.
  if (function->name().empty()) {
    return;
  }

  // Async value leak check before and after running the function.
  size_t before_num_values;
  if (AsyncValue::AsyncValueAllocationTrackingEnabled())
    before_num_values = AsyncValue::GetNumAsyncValueInstances();

  // Actually run the function.
  tfrt::outs() << "--- Running '" << function->name() << "':\n";
  tfrt::outs().flush();
  if (function->function_kind() == FunctionKind::kSyncBEFFunction) {
    RunSyncBefFunctionHelper(host, function);
  } else {
    RunAsyncBefFunctionHelper(host, function);
  }

  if (AsyncValue::AsyncValueAllocationTrackingEnabled()) {
    auto after_num_values = AsyncValue::GetNumAsyncValueInstances();
    if (before_num_values != after_num_values) {
      llvm::errs() << "Evaluation of function '" << function->name()
                   << "' leaked " << (after_num_values - before_num_values)
                   << " async values (before: " << before_num_values
                   << ", after: " << after_num_values << ")!\n";
      abort();
    }
  }
}

}  // namespace tfrt
