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

// This file implements the LoggingOpHandler class and the hooks to create it.

#if defined(_MSC_VER)
#include <io.h>
#ifndef STDIN_FILENO
#define STDIN_FILENO 0
#endif
#ifndef STDOUT_FILENO
#define STDOUT_FILENO 1
#endif
#ifndef STDERR_FILENO
#define STDERR_FILENO 2
#endif
#endif

#ifndef _WIN32
#include <unistd.h>
#endif

#include <system_error>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "tfrt/core_runtime/core_runtime.h"
#include "tfrt/core_runtime/kernels.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/core_runtime/op_handler.h"
#include "tfrt/core_runtime/op_invocation.h"
#include "tfrt/core_runtime/tensor_handle.h"
#include "tfrt/dtype/dtype_formatter.h"
#include "tfrt/host_context/execution_context.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/mutex.h"
#include "tfrt/support/thread_annotations.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/host_tensor.h"
#include "tfrt/tensor/string_host_tensor.h"
#include "tfrt/tensor/tensor.h"

namespace tfrt {

namespace {

void FlattenTensorAndDumpToOStream(const DenseHostTensor &dht,
                                   llvm::raw_ostream &os) {
  auto element_size = GetHostSize(dht.dtype());
  auto *data_ptr = static_cast<const char *>(dht.data());

  // TODO(tfrt-devs): Dump to BTF format once we have BTF reader/writer
  // implemented in C++.
  // This tensor dump can be loaded into numpy and reshaped.
  // t = np.genfromtxt(tensor_filename, delimiter=",")
  // t = t.reshape(original_shape)
  for (Index i = 0, e = dht.NumElements(); i != e; ++i) {
    if (i != 0) os << ", ";
    os << FormatDType(dht.dtype(), data_ptr + i * element_size);
  }
}

void FlattenTensorAndDumpToOStream(const StringHostTensor &sht,
                                   llvm::raw_ostream &os) {
  auto strings = sht.strings();
  for (Index i = 0, e = sht.NumElements(); i != e; ++i) {
    if (i != 0) os << ", ";
    os << strings[i];
  }
}

class LoggingOpHandler : public OpHandler {
 public:
  static llvm::Expected<std::unique_ptr<LoggingOpHandler>> Create(
      CoreRuntime *runtime, OpHandler *fallback, bool sync_log_results) {
    std::unique_ptr<llvm::raw_fd_ostream> metadata_ostream;
    if (auto metadata_dump_prefix =
            std::getenv("LOGGING_DEV_METADATA_DUMP_PREFIX")) {
      std::string metadata_dump_filename =
          StrCat(metadata_dump_prefix, "metadata.log");
      std::error_code error_code;
      metadata_ostream = std::make_unique<llvm::raw_fd_ostream>(
          metadata_dump_filename, error_code, llvm::sys::fs::OF_Text);
      if (error_code)
        return MakeStringError(
            StrCat("error opening file ", metadata_dump_filename));
      metadata_ostream->SetUnbuffered();
    } else {
      metadata_ostream =
          std::make_unique<llvm::raw_fd_ostream>(/*fd=*/STDERR_FILENO,
                                                 /*shouldClose=*/false,
                                                 /*unbuffered=*/true);
    }

    std::string tensor_dump_prefix_str;
    if (auto tensor_dump_prefix =
            std::getenv("LOGGING_DEV_TENSOR_DUMP_PREFIX")) {
      tensor_dump_prefix_str = tensor_dump_prefix;
    }

    auto op_handler = std::make_unique<LoggingOpHandler>(
        runtime, fallback, sync_log_results, std::move(tensor_dump_prefix_str),
        std::move(metadata_ostream));
    return std::move(op_handler);
  }

  explicit LoggingOpHandler(
      CoreRuntime *runtime, OpHandler *fallback, bool sync_log_results,
      std::string tensor_dump_prefix,
      std::unique_ptr<llvm::raw_fd_ostream> metadata_ostream)
      : OpHandler((sync_log_results ? "sync_logging" : "logging"), runtime,
                  fallback),
        sync_log_results_(sync_log_results),
        tensor_dump_prefix_(std::move(tensor_dump_prefix)),
        metadata_ostream_(std::move(metadata_ostream)) {}

  Expected<CoreRuntimeOp> MakeOp(string_view op_name) override;

 private:
  bool ShouldDumpTensorToFile() const { return !tensor_dump_prefix_.empty(); }

  // TODO(tfrt-devs): Handle error TensorHandle.
  llvm::SmallVector<RCReference<AsyncValue>, 4> CollectAsyncHostTensors(
      const ExecutionContext &exec_ctx, ArrayRef<TensorHandle> tensor_handles) {
    auto *host = exec_ctx.host();

    // Wait for all metadatas.
    std::vector<RCReference<AsyncValue>> tensor_metadatas;
    for (auto &th : tensor_handles) {
      if (!th.IsMetadataAvailable()) {
        tensor_metadatas.push_back(th.GetAsyncMetadata().CopyRCRef());
      }
    }

    host->Await(tensor_metadatas);

    // Convert all tensors to HostTensor.
    llvm::SmallVector<RCReference<AsyncValue>, 4> async_hts;
    for (auto &th : tensor_handles) {
      auto host_tensor_handle =
          th.TransferTo(exec_ctx, host->GetHostDeviceRef(),
                        th.GetAvailableMetadata().dtype == DType::String
                            ? StringHostTensor::kTensorType
                            : DenseHostTensor::kTensorType);
      async_hts.emplace_back(host_tensor_handle.ReleaseTensorRef());
    }

    // Wait for the conversion to complete.
    host->Await(async_hts);

    return async_hts;
  }

  void PrintAsyncHostTensors(
      ArrayRef<RCReference<AsyncValue>> async_host_tensors, bool is_input,
      int id_number, string_view op_name) {
    // Print out the input HostTensor.
    std::string message;
    llvm::raw_string_ostream os(message);

    os << (is_input ? "Inputs" : "Outputs") << " for [" << id_number << "]: '"
       << op_name << "':\n";

    int index = 0;
    for (auto &async_host_tensor : async_host_tensors) {
      os << (is_input ? "  Input" : "  Output") << " for [" << id_number
         << "] tensor " << index << ": ";
      auto &tensor = async_host_tensor->get<HostTensor>();
      tensor.Print(os);
      if (ShouldDumpTensorToFile()) {
        PrintTensorToFile(tensor, id_number, op_name,
                          is_input ? "input" : "output", index);
      }
      os << "\n";
      ++index;
    }
    os << "\n";

    Print(os.str().c_str());
  }

  void Print(string_view contents) {
    mutex_lock lock(mu_);
    *metadata_ostream_ << contents;
  }

  void PrintTensorToFile(const HostTensor &tensor, int log_counter,
                         string_view op_name, string_view input_or_output,
                         int input_or_output_index) {
    std::string tensor_dump_filename =
        StrCat(tensor_dump_prefix_, "op_", log_counter, "_", op_name, "_",
               input_or_output, "_", input_or_output_index);
    std::error_code error_code;
    llvm::raw_fd_ostream fostream(tensor_dump_filename, error_code,
                                  llvm::sys::fs::OF_Text);
    if (error_code) {
      fprintf(stderr, "Cannot open file %s for writing (%d): %s\n",
              tensor_dump_filename.c_str(), error_code.value(),
              error_code.message().c_str());
      abort();
    }

    if (auto *dht = llvm::dyn_cast<DenseHostTensor>(&tensor)) {
      FlattenTensorAndDumpToOStream(*dht, fostream);
    } else if (auto *sht = llvm::dyn_cast<StringHostTensor>(&tensor)) {
      FlattenTensorAndDumpToOStream(*sht, fostream);
    } else {
      fprintf(stderr,
              "Only support printing DenseHostTensor and StringHostTensor");
      abort();
    }
  }

  // Synchronously log the op results
  const bool sync_log_results_;
  std::atomic<uint32_t> log_counter_{0};
  std::string tensor_dump_prefix_;

  mutable mutex mu_;
  std::unique_ptr<llvm::raw_fd_ostream> metadata_ostream_ TFRT_GUARDED_BY(mu_);
};
}  // namespace

Expected<CoreRuntimeOp> LoggingOpHandler::MakeOp(string_view op_name) {
  auto fallback_handle = GetFallback()->MakeOp(op_name);
  if (!fallback_handle) return fallback_handle.takeError();
  return CoreRuntimeOp(
      [this, op_name = op_name.str(),
       fallback_handle =
           std::move(fallback_handle.get())](const OpInvocation &invocation) {
        // TODO(tfrt-devs): Make this class thread safe.
        auto id_number = log_counter_.fetch_add(1);

        // Used to make logging messages more grammatical.
        auto plural = [](size_t n) -> const char * {
          return n == 1 ? "" : "s";
        };

        // Print everything into a std::string, and then print it with printf.
        // This ensures that the messages are emitted atomically (even if
        // multiple threads are concurrently logging), because printf has an
        // internal mutex.
        {
          std::string message;
          llvm::raw_string_ostream os(message);

          auto num_args = invocation.arguments.size();
          auto num_results = invocation.results.size();
          os << '[' << id_number << "] dispatch '" << op_name << "' "
             << num_args << " argument" << plural(num_args) << ", "
             << num_results << " result" << plural(num_results);
          if (invocation.attrs.GetNumEntries() == 0) {
            os << ", no attributes\n";
          } else {
            os << ", ";
            invocation.attrs.Print(os);
          }

          Print(os.str());
        }

        {
          // Collect all input tensors, convert them to HostTensor's and await.
          llvm::SmallVector<RCReference<AsyncValue>, 4> async_host_tensors =
              CollectAsyncHostTensors(invocation.exec_ctx,
                                      invocation.arguments);

          PrintAsyncHostTensors(async_host_tensors, /*is_input=*/true,
                                id_number, op_name);
        }

        // Delegate to the op_handler we wrap.
        fallback_handle(invocation);
        if (sync_log_results_ && !invocation.results.empty()) {
          // Collect all output tensors, convert them to DHT and await.
          llvm::SmallVector<RCReference<AsyncValue>, 4> async_host_tensors =
              CollectAsyncHostTensors(invocation.exec_ctx, invocation.results);

          PrintAsyncHostTensors(async_host_tensors, /*is_input=*/false,
                                id_number, op_name);
        }
      },
      /*is_fallback=*/false);
}

llvm::Expected<tfrt::OpHandler *> CreateLoggingOpHandler(
    tfrt::CoreRuntime *runtime, OpHandler *fallback, bool sync_log_results) {
  auto op_handler =
      LoggingOpHandler::Create(runtime, fallback, sync_log_results);
  if (auto error = op_handler.takeError()) {
    return std::move(error);
  }
  auto op_handler_ptr = op_handler->get();
  runtime->TakeOpHandler(std::move(op_handler.get()));
  return op_handler_ptr;
}

}  // namespace tfrt
