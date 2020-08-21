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

//===- simple_test_kernels.cc ---------------------------------------------===//
//
// This library contains test kernels needed by example_kernels unit tests.
//
//===----------------------------------------------------------------------===//

#include "llvm_derived/Support/raw_ostream.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/host_context/shared_context.h"
#include "tfrt/host_context/sync_kernel_utils.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/logging.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/tensor_serialize_utils.h"
#include "tfrt/test_kernels.h"

namespace tfrt {

// This kernel produces an error.
static llvm::Expected<int32_t> TestFail() {
  return MakeStringError("something bad happened");
}

// This kernel produces only failure/success as Error.
static Error TestError() { return MakeStringError("something bad happened"); }

// This kernel produces a normal output and an error output.
static void TestPartialFail(Result<int32_t> one, Result<int32_t> error_out,
                            AsyncKernelFrame* frame) {
  one.Emplace(1);
  frame->ReportError("something bad happened");
}

// This kernel produces an error asynchronously.
static void TestReportErrorAsync(Result<int32_t> out,
                                 const ExecutionContext& exec_ctx,
                                 AsyncKernelFrame* frame) {
  exec_ctx.host()->EnqueueWork(
      [out_ref = out.Allocate(), frame_copy = *frame]() mutable {
        frame_copy.ReportError("something bad happened asynchronously");
      });
}

// This kernel cancels execution.
static void TestCancel(Argument<Chain> chain_in, Result<int> int_out,
                       Result<Chain> chain_out,
                       const ExecutionContext& exec_ctx) {
  // Calling RequestContext::Cancel() for testing the cancel behavior.
  // Do NOT do this in a normal kernel. RequestContext::Cancel() should be
  // called by a client external to the BEFExecutor.
  exec_ctx.request_ctx()->Cancel();
  int_out.Emplace(0);
  chain_out.Set(chain_in);
}

// This kernel expects a nested array attribute in the form:
//  [["string", [0 : i32, 1 : i32]], [1.0 : f32]]
// , and return a result for each leaf value.
static void TestFlat(Result<std::string> str_out, Result<int32_t> int_out_0,
                     Result<int32_t> int_out_1, Result<float> float_out,
                     AggregateAttr array) {
  auto a0 = array.GetAttributeOfType<AggregateAttr>(0);
  str_out.Emplace(a0.GetAttributeOfType<StringAttr>(0).GetValue().str());

  auto a01 = a0.GetAttributeOfType<ArrayAttr>(1).GetValue<int32_t>();
  int_out_0.Emplace(a01[0]);
  int_out_1.Emplace(a01[1]);

  auto a1 = array.GetAttributeOfType<ArrayAttr>(1).GetValue<float>();
  float_out.Emplace(a1[0]);
}

// A resource whose lifetime is explicitly managed by
// Test{Allocate,Deallocate}Resource.
class TestResource {
 public:
  TestResource() {
    tfrt::outs() << "Allocated TestResource\n";
    tfrt::outs().flush();
  }
  ~TestResource() {
    tfrt::outs() << "Destroyed TestResource\n";
    tfrt::outs().flush();
  }

  // Not copyable or movable.
  TestResource(const TestResource&) = delete;
  TestResource& operator=(const TestResource&) = delete;
  TestResource(TestResource&&) = delete;
  TestResource& operator=(TestResource&&) = delete;
};

// Allocate a TestResource. This returns a unique_ptr<TestResource>, so the
// corresponding Deallocate op can explicitly destroy the resource by resetting
// the unique_ptr. The unique_ptr also lets the executor automatically destroy
// the resource if the Deallocate kernel never runs - that can happen if an
// error occurs, or if execution is canceled.
static void TestAllocateResource(
    Argument<Chain> chain_in,
    Result<std::unique_ptr<TestResource>> resource_out,
    Result<Chain> chain_out) {
  resource_out.Emplace(new TestResource());
  chain_out.Set(chain_in);
}

static void TestDeallocateResource(
    Argument<std::unique_ptr<TestResource>> resource_in,
    Argument<Chain> chain_in, Result<Chain> chain_out) {
  resource_in->reset();
  chain_out.Set(chain_in);
  tfrt::outs() << "tfrt_test.deallocate_resource done\n";
  tfrt::outs().flush();
}

namespace {

// Helper structs for testing AsyncValue::get for non-polymorphic types
struct TestBase1 {
  std::string base1 = "base1";
};

struct TestChild1 : TestBase1 {
  std::string child1 = "child1";
};

// Helper structs for testing AsyncValue::get for polymorphic types
struct TestBase2 {
  virtual ~TestBase2() {}
  std::string base2 = "base2";
};

struct TestChild2 : TestBase2 {
  std::string child2 = "child2";
};

struct TestFinalClass final {
  std::string name = "final_class";
};
}  // namespace

static std::string TestAsyncValueGet(const ExecutionContext& exec_ctx) {
  std::string return_value;
  llvm::raw_string_ostream sstr(return_value);
  HostContext* host = exec_ctx.host();

  auto child1_av_ref = MakeAvailableAsyncValueRef<TestChild1>(host);
  AsyncValue* child1_av = child1_av_ref.GetAsyncValue();
  auto& base1 = child1_av->get<TestBase1>();
  sstr << base1.base1;

  auto child2_av_ref = MakeAvailableAsyncValueRef<TestChild2>(host);
  AsyncValue* child2_av = child2_av_ref.GetAsyncValue();
  auto& base2 = child2_av->get<TestBase2>();
  sstr << ":" << base2.base2;

  auto final_class_av_ref = MakeAvailableAsyncValueRef<TestFinalClass>(host);
  AsyncValue* final_class_av = final_class_av_ref.GetAsyncValue();
  auto& final_class_val = final_class_av->get<TestFinalClass>();
  sstr << ":" << final_class_val.name;

  auto int_av_ref = MakeAvailableAsyncValueRef<int>(host, 3);
  AsyncValue* int_av = int_av_ref.GetAsyncValue();
  auto& int_val = int_av->get<int>();
  sstr << ":" << int_val;

  return sstr.str();
}

// Return a string describing an AsyncValueRef<int>'s availability and value
// ("unavailable", "available:3", etc.)
static std::string AsyncValueRefToString(const AsyncValueRef<int>& ref) {
  std::string return_value;
  llvm::raw_string_ostream sstr(return_value);
  sstr << (ref.IsAvailable() ? "available" : "unavailable");
  if (ref.IsAvailable()) {
    sstr << ":" << ref.get();
  }
  return sstr.str();
}

static std::string TestAsyncValueRef(const ExecutionContext& exec_ctx) {
  //////////////////////////////////////////////////////////////////////
  // Sync usage.

  HostContext* host = exec_ctx.host();

  // Construct an available int with value 2.
  AsyncValueRef<int> two = MakeAvailableAsyncValueRef<int>(host, 2);

  std::string return_value;
  llvm::raw_string_ostream sstr(return_value);

  sstr << AsyncValueRefToString(two);

  // Construct an unavailable int.
  AsyncValueRef<int> three = MakeUnconstructedAsyncValueRef<int>(host);
  sstr << ";" << AsyncValueRefToString(three);

  three.AndThen([&sstr, &three]() { sstr << "(" << three.get() << ")"; });
  sstr << ";";

  // Set the unavailable int's value to 3.
  three.emplace(3);
  sstr << ";" << AsyncValueRefToString(three);

  //////////////////////////////////////////////////////////////////////
  // Async usage.
  //
  // Allocate an unavailable int.
  AsyncValueRef<int> four = MakeUnconstructedAsyncValueRef<int>(host);
  sstr << ";" << AsyncValueRefToString(four);

  // Copy the unavailable int, forming a new ref, and bind it to a lambda.
  // Ownership of this new ref transfers to the lambda.
  auto lambda = [four_copy = four.CopyRef()]() mutable {
    four_copy.emplace(4);
  };

  // We can synchronously return 'four' at this point.

  // The lambda can run asynchronously.
  lambda();

  sstr << ";" << AsyncValueRefToString(four);

  return sstr.str();
}

static Chain TestLogging(Argument<std::string> arg) {
  TFRT_LOG(INFO) << "from TFRT_LOG(INFO): " << *arg;
  TFRT_LOG(WARNING) << "from TFRT_LOG(WARNING): " << *arg;
  TFRT_LOG(ERROR) << "from TFRT_LOG(ERROR): " << *arg;
  TFRT_LOG_IF(INFO, true) << "from TFRT_LOG_IF(INFO, true): " << *arg;
  TFRT_LOG_IF(WARNING, true) << "from TFRT_LOG_IF(WARNING, true): " << *arg;
  TFRT_LOG_IF(ERROR, true) << "from TFRT_LOG_IF(ERROR, true): " << *arg;
  // The following will have no output.
  TFRT_LOG_IF(INFO, false) << "from TFRT_LOG_IF(INFO, false): " << *arg;
  TFRT_LOG_IF(WARNING, false) << "from TFRT_LOG_IF(WARNING, false): " << *arg;
  TFRT_LOG_IF(ERROR, false) << "from TFRT_LOG_IF(ERROR, false): " << *arg;
  return Chain();
}

namespace {

class SampleSharedContext : public SharedContext {
 public:
  // A SharedContext is required to have a constructor that takes a
  // HostContext*.
  explicit SampleSharedContext(HostContext* host) {
    // assert to make sure only one instance of the class is instantiated.
    assert(instance_count_ == 0);
    name_ += std::to_string(instance_count_.load());
    ++instance_count_;
  }

  const std::string& name() const { return name_; }

 private:
  static std::atomic<int> instance_count_;
  std::string name_ = "sample_shared_context";
};

std::atomic<int> SampleSharedContext::instance_count_{0};
}  // namespace

static void TestUseSampleSharedContext(Argument<Chain> chain,
                                       Result<std::string> name,
                                       Result<Chain> out_chain,
                                       const ExecutionContext& exec_ctx) {
  auto& shared_ctx =
      exec_ctx.host()->GetOrCreateSharedContext<SampleSharedContext>();
  name.Emplace(shared_ctx.name());
  out_chain.Set(chain);
}

static DenseHostTensor TestConstDenseAttr(DenseAttr dense_attr,
                                          const ExecutionContext& exec_ctx) {
  auto result =
      DeserializeDenseHostTensorFromDenseAttr(dense_attr, exec_ctx.host());
  assert(result);

  return std::move(*result);
}

// For testing RemainingSyncArguments
static int TestSyncSum(int a, RemainingSyncArguments other_args) {
  int sum = a;
  for (int i = 0; i < other_args.size(); ++i) {
    sum += other_args[i]->get<int>();
  }
  return sum;
}

// For testing RepeatedSyncArguments
static int TestSyncSum2(int a, RepeatedSyncArguments<int> other_args) {
  int sum = a;
  for (auto i : other_args) {
    sum += i;
  }
  return sum;
}

// For testing a large number of attributes
static void TestSyncSumAttributes(SyncKernelFrame* frame) {
  int sum = 0;
  for (int i = 0; i < frame->GetNumAttributes(); ++i) {
    sum += *frame->GetAttributeAt<int>(i);
  }

  frame->EmplaceResultAt<int>(0, sum);
}

void RegisterSimpleTestKernels(KernelRegistry* registry) {
  registry->AddKernel("tfrt_test.fail", TFRT_KERNEL(TestFail));
  registry->AddKernel("tfrt_test.partial_fail", TFRT_KERNEL(TestPartialFail));
  registry->AddKernel("tfrt_test.cancel", TFRT_KERNEL(TestCancel));
  registry->AddKernel("tfrt_test.flat", TFRT_KERNEL(TestFlat));
  registry->AddKernel("tfrt_test.async_value_get",
                      TFRT_KERNEL(TestAsyncValueGet));
  registry->AddKernel("tfrt_test.async_value_ref",
                      TFRT_KERNEL(TestAsyncValueRef));
  registry->AddKernel("tfrt_test.logging", TFRT_KERNEL(TestLogging));
  registry->AddKernel("tfrt_test.allocate_resource",
                      TFRT_KERNEL(TestAllocateResource));
  registry->AddKernel("tfrt_test.deallocate_resource",
                      TFRT_KERNEL(TestDeallocateResource));
  registry->AddKernel("tfrt_test.use_sample_shared_context",
                      TFRT_KERNEL(TestUseSampleSharedContext));
  registry->AddKernel("tfrt_test.report_error_async",
                      TFRT_KERNEL(TestReportErrorAsync));
  registry->AddKernel("tfrt_test.const_dense_attr",
                      TFRT_KERNEL(TestConstDenseAttr));
  registry->AddSyncKernel("tfrt_test.sync.const_dense_attr",
                          TFRT_SYNC_KERNEL(TestConstDenseAttr));

  registry->AddSyncKernel("tfrt_test.fail_s", TFRT_SYNC_KERNEL(TestFail));
  registry->AddSyncKernel("tfrt_test.error_s", TFRT_SYNC_KERNEL(TestError));
  registry->AddSyncKernel("tfrt_test.sync_sum", TFRT_SYNC_KERNEL(TestSyncSum));
  registry->AddSyncKernel("tfrt_test.sync_sum2",
                          TFRT_SYNC_KERNEL(TestSyncSum2));
  registry->AddSyncKernel("tfrt_test.sync_sum_attributes",
                          TestSyncSumAttributes);
}

}  // namespace tfrt
