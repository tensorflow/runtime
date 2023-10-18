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

// This file is a simple example showing how to use core_runtime API to execute
// some simple ops.

#include "driver.h"

#include <memory>

#include "benchmark/benchmark.h"
#include "gtest/gtest.h"
#include "llvm_derived/Support/raw_ostream.h"
#include "tfrt/core_runtime/core_runtime_op.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/core_runtime/tensor_handle.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/host_context.h"
#include "tfrt/host_context/location.h"
#include "tfrt/host_context/native_function.h"
#include "tfrt/support/logging.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "tfrt/tensor/dense_host_tensor_view.h"
#include "tfrt/tensor/tensor_metadata.h"

namespace tfrt {
namespace {

class CpuDriverTest : public testing::Test {
 protected:
  example::CoreRuntimeCpuDriver driver_;
};

TEST_F(CpuDriverTest, ReluTest_InputForward) {
  tfrt::OpAttrs attrs;
  attrs.SetArray("shape", tfrt::ArrayRef<Index>{2, 2});
  attrs.SetArray("values", tfrt::ArrayRef<float>{2.0});
  tfrt::TensorHandle a1;
  driver_.Execute(driver_.CreateExecutionContext(__FILE__, __LINE__),
                  "tfrt_test.create_dense_tensor", {}, attrs.freeze(), a1);
  auto buffer_pointer =
      a1.GetAsyncTensor()->get<DenseHostTensor>().buffer().get();

  tfrt::OpAttrs empty_attrs;
  tfrt::TensorHandle a2;
  driver_.Execute(driver_.CreateExecutionContext(__FILE__, __LINE__),
                  "tfrt_test.relu", a1, empty_attrs.freeze(), a2);

  ASSERT_EQ(a2.GetAsyncTensor()->get<DenseHostTensor>().buffer().get(),
            buffer_pointer);
}

TEST_F(CpuDriverTest, CompositeOpTest) {
  tfrt::OpAttrs attrs;
  attrs.SetArray("shape", tfrt::ArrayRef<Index>{1});
  attrs.SetArray("values", tfrt::ArrayRef<int32_t>{1});
  tfrt::TensorHandle a1;
  driver_.Execute(driver_.CreateExecutionContext(__FILE__, __LINE__),
                  "tfrt_test.create_dense_tensor", {}, attrs.freeze(), a1);

  tfrt::TensorHandle add_args[2] = {a1.CopyRef(), a1.CopyRef()};
  tfrt::TensorHandle a2;

  // Add 2 scalar int32 tensors, with TensorHandle as the input/output type.
  tfrt::NativeCallable add_callable = [](AsyncValue* const* arguments,
                                         int num_arguments,
                                         RCReference<AsyncValue>* results,
                                         int num_results, HostContext* host) {
    assert(num_arguments == 3);
    auto& a = arguments[1]->get<TensorHandle>();
    auto& b = arguments[2]->get<TensorHandle>();
    TFRT_DLOG(INFO) << "a is " << a;
    TFRT_DLOG(INFO) << "b is " << b;
    const auto& a_tensor = a.GetAsyncTensor()->get<DenseHostTensor>();
    const auto& b_tensor = b.GetAsyncTensor()->get<DenseHostTensor>();
    const int32_t result_value =
        DHTArrayView<int32_t>(&a_tensor).Elements()[0] +
        DHTArrayView<int32_t>(&b_tensor).Elements()[0];
    AsyncValueRef<DenseHostTensor> result =
        DenseHostTensor::MakeConstructedAsyncValueRef(a.GetAvailableMetadata(),
                                                      host);
    assert(result);
    MutableDHTArrayView<int32_t> result_view(&result.get());
    *result_view.data() = result_value;
    TFRT_DLOG(INFO) << "Result value is " << result_value;
    result.SetStateConcrete();

    assert(num_results == 2);
    results[0] = GetReadyChain();
    results[1] = MakeAvailableAsyncValueRef<TensorHandle>(
        host->GetHostDeviceRef(), a.GetAvailableMetadata(), std::move(result));
    TFRT_DLOG(INFO) << "result is " << results[1]->get<TensorHandle>()
                    << " with state " << results[1]->IsAvailable();
  };

  TypeName chain_type =
      driver_.GetHostContext()->GetKernelRegistry().GetType("!tfrt.chain");
  TypeName tensor_handle_type =
      driver_.GetHostContext()->GetKernelRegistry().GetType(
          CoreRuntime::kTensorHandleType);
  NativeFunction fn(
      "test_fn",
      /*argument_types=*/{chain_type, tensor_handle_type, tensor_handle_type},
      /*result_types=*/{chain_type, tensor_handle_type}, add_callable);
  driver_.WaitForHostContextQuiesce();

  auto op = driver_.MakeCompositeOp(&fn);
  op(driver_.CreateExecutionContext(__FILE__, __LINE__), add_args, OpAttrsRef(),
     a2, /*chain=*/nullptr);
  driver_.WaitForHostContextQuiesce();

  auto a2_metadata = a2.GetAvailableMetadata();
  ASSERT_EQ(a2_metadata.shape.GetRank(), 1);
  ASSERT_EQ(a2_metadata.shape.GetDimensionSize(0), 1);

  auto a2_view =
      DHTArrayView<int32_t>(&a2.GetAsyncTensor()->get<DenseHostTensor>());
  ASSERT_EQ(a2_view.Elements()[0], 2);
}

TEST_F(CpuDriverTest, NativeCompositeOpTest) {
  // Add 2 scalar int32.
  tfrt::NativeCallable add_callable =
      [](AsyncValue* const* arguments, int num_arguments,
         RCReference<AsyncValue>* results, int num_results, HostContext* host) {
        assert(num_arguments == 3);
        const int32_t result_value =
            arguments[1]->get<int32_t>() + arguments[2]->get<int32_t>();
        TFRT_DLOG(INFO) << "Result value is " << result_value;

        assert(num_results == 2);
        results[0] = GetReadyChain();
        results[1] = MakeAvailableAsyncValueRef<int32_t>(result_value);
      };

  TypeName chain_type =
      driver_.GetHostContext()->GetKernelRegistry().GetType("!tfrt.chain");
  TypeName i32_type =
      driver_.GetHostContext()->GetKernelRegistry().GetType("tfrt.i32");
  NativeFunction fn("test_fn",
                    /*argument_types=*/{chain_type, i32_type, i32_type},
                    /*result_types=*/{chain_type, i32_type}, add_callable);
  auto op = driver_.MakeNativeCompositeOp(&fn);

  auto a1 = MakeAvailableAsyncValueRef<int32_t>(1);
  tfrt::RCReference<AsyncValue> args[2] = {a1.CopyRCRef(), a1.CopyRCRef()};
  tfrt::RCReference<AsyncValue> a2;
  CompositeOpInvocation op_invocation{
      driver_.CreateExecutionContext(__FILE__, __LINE__), llvm::ArrayRef(args),
      a2, nullptr};
  op(op_invocation);
  driver_.WaitForHostContextQuiesce();

  ASSERT_EQ(a2->get<int32_t>(), 2);
}

}  // namespace
}  // namespace tfrt
