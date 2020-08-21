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

//===- simple_kernels.cc --------------------------------------------------===//
//
// This file implements a few simple classes of synchronous kernels for testing.
//
//===----------------------------------------------------------------------===//

#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <thread>

#include "llvm_derived/Support/raw_ostream.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/error_util.h"
#include "tfrt/test_kernels.h"

namespace tfrt {
namespace {

void SleepForRandomDuration() {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> dist(1, 10);
  std::this_thread::sleep_for(std::chrono::milliseconds(dist(rng)));
}

//===----------------------------------------------------------------------===//
// String kernels
//===----------------------------------------------------------------------===//
// String attributes are passed with a pointer to the data, with a length
// before it.
static std::string TestGetString(StringAttribute str) { return str.str(); }

static Chain TestPrintString(Argument<std::string> arg) {
  tfrt::outs() << "string = " << *arg << '\n';
  tfrt::outs().flush();
  return Chain();
}

static std::string TestAppendString(Argument<std::string> arg0,
                                    Argument<std::string> arg1) {
  return arg0.get() + arg1.get();
}

static Expected<std::string> ReadStringFromFile(const std::string& path) {
  std::ifstream input(path, std::ios::in | std::ios::binary);
  std::string data((std::istreambuf_iterator<char>(input)),
                   (std::istreambuf_iterator<char>()));
  if (input.fail()) {
    return MakeStringError("failed to read from file ", path);
  }
  return data;
}

static bool TestStringEqual(const std::string& s1, const std::string& s2) {
  return s1 == s2;
}

// path_format should take an integer as argument.
static std::string FormatString(const std::string& path_format,
                                const int32_t index) {
  std::string path;
  llvm::raw_string_ostream ss(path);
  ss << llvm::format(path_format.c_str(), index);
  return ss.str();
}

static void SetupStringRegistry(KernelRegistry* registry) {
  registry->AddKernel("tfrt_test.get_string", TFRT_KERNEL(TestGetString));
  registry->AddKernel("tfrt_test.print_string", TFRT_KERNEL(TestPrintString));
  registry->AddKernel("tfrt_test.append_string", TFRT_KERNEL(TestAppendString));
  registry->AddKernel("tfrt_test.string_equal", TFRT_KERNEL(TestStringEqual));
  registry->AddKernel("tfrt_test.read_string_from_file",
                      TFRT_KERNEL(ReadStringFromFile));
  registry->AddKernel("tfrt_test.format_string", TFRT_KERNEL(FormatString));
}

//===----------------------------------------------------------------------===//
// Value Tracking kernels
//===----------------------------------------------------------------------===//
//
// These kernels fundamentally model integer kernels, but they print on
// construction and destruction so we can write tests showing that the values
// get created and destroyed at the right times.

class VTValue {
 public:
  explicit VTValue(int32_t v) : v(v) {
    tfrt::outs() << "constructed vt.value(" << v << ")\n";
    tfrt::outs().flush();
  }

  VTValue(const VTValue& rhs) : v(rhs.v) {
    tfrt::outs() << "copy constructed vt.value(" << v << ")\n";
    tfrt::outs().flush();
  }

  VTValue(VTValue&& rhs) : v(rhs.v) {
    rhs.v = -1;
    tfrt::outs() << "move constructed vt.value(" << v << ")\n";
    tfrt::outs().flush();
  }

  ~VTValue() {
    if (v == -1) return;
    tfrt::outs() << "destroyed vt.value(" << v << ")\n";
    tfrt::outs().flush();
  }

  int32_t v;
};

static VTValue VtConstant(Attribute<int32_t> v) { return VTValue(*v); }

static VTValue VtAdd(Argument<VTValue> lhs, Argument<VTValue> rhs) {
  return VTValue(lhs->v + rhs->v);
}

static Chain VtPrint(Argument<VTValue> v) {
  tfrt::outs() << "print vt_value(" << v->v << ")\n";
  tfrt::outs().flush();
  return Chain();
}

static void SetupValueTrackingRegistry(KernelRegistry* registry) {
  registry->AddKernel("vt.constant", TFRT_KERNEL(VtConstant));
  registry->AddKernel("vt.add", TFRT_KERNEL(VtAdd));
  registry->AddKernel("vt.print", TFRT_KERNEL(VtPrint));
}

//===----------------------------------------------------------------------===//
// Misc kernels
//===----------------------------------------------------------------------===//

static Chain TestPrintHello() {
  tfrt::outs() << "hello host executor!\n";
  tfrt::outs().flush();
  return Chain();
}

// Compute the sum of the input args
static void TestSum(RepeatedArguments<int32_t> args, Result<int32_t> result) {
  assert(args.size() > 0);

  int32_t sum = 0;
  for (int32_t i : args) {
    sum += i;
  }

  result.Emplace(sum);
}

// Share input args with results without copying
static void TestShareToTwo(Argument<VTValue> in, Result<VTValue> out_0,
                           Result<VTValue> out_1) {
  out_0.Set(in);
  out_1.Set(in);
}

static void TestMemoryLeakOneInt32(const ExecutionContext& exec_ctx) {
  // We allocate an integer to intentionally cause a memory leak.
  exec_ctx.host()->Allocate<int32_t>();
}

// An example of bad behaving kernel, that does expensive work (sleep to emulate
// that) in the caller thread. Intended for testing system behavior in the
// presence of bad actors (kernels).
template <typename T>
static AsyncValueRef<T> TestCopyWithDelay(Argument<T> in,
                                          const ExecutionContext& exec_ctx) {
  SleepForRandomDuration();
  return MakeAvailableAsyncValueRef<T>(exec_ctx.host(), in.get());
}

//===----------------------------------------------------------------------===//
// tfrt_test count3 kernels. For input x, returns x+1, x+2, x+3
// For demonstrating using std::tuple to return multiple outputs
//===----------------------------------------------------------------------===//
template <typename T>
std::tuple<T, T, T> TestCount3(T x) {
  return {x + 1, x + 2, x + 3};
}
}  // namespace

void RegisterSimpleKernels(KernelRegistry* registry) {
  registry->AddKernel("tfrt_test.print_hello", TFRT_KERNEL(TestPrintHello));
  registry->AddKernel("tfrt_test.sum", TFRT_KERNEL(TestSum));
  registry->AddKernel("tfrt_test.share_to_two", TFRT_KERNEL(TestShareToTwo));
  registry->AddKernel("tfrt_test.memory_leak_one_int32",
                      TFRT_KERNEL(TestMemoryLeakOneInt32));
  registry->AddKernel("tfrt_test.count3.i32", TFRT_KERNEL(TestCount3<int32_t>));
  registry->AddKernel("tfrt_test.count3.i64", TFRT_KERNEL(TestCount3<int64_t>));
  registry->AddKernel("tfrt_test.copy.with_delay.i32",
                      TFRT_KERNEL(TestCopyWithDelay<int32_t>));
  registry->AddKernel("tfrt_test.copy.with_delay.i64",
                      TFRT_KERNEL(TestCopyWithDelay<int64_t>));

  SetupStringRegistry(registry);
  SetupValueTrackingRegistry(registry);
}

}  // namespace tfrt
