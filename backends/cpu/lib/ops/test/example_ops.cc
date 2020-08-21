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

//===- examples_ops.cc ----------------------------------------------------===//
//
// This file defines some example op implementations in the "tfrt_test."
// namespace.
//
//===----------------------------------------------------------------------===//

#include "../../kernels/cpu_kernels.h"
#include "llvm/Support/Casting.h"
#include "llvm_derived/Support/raw_ostream.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/core_runtime/op_utils.h"
#include "tfrt/cpu/core_runtime/cpu_op_registry.h"
#include "tfrt/cpu/ops/test/cpu_ops_and_kernels.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/parallel_for.h"
#include "tfrt/support/error_util.h"
#include "tfrt/tensor/dense_host_tensor_view.h"
#include "tfrt/tensor/scalar_host_tensor.h"

namespace tfrt {

//===----------------------------------------------------------------------===//
// test.odd_collector op
//===----------------------------------------------------------------------===//

// This implements a function "x = test.odd_collector(y)" that takes a 1d input
// and returns a 1d output containing only the odd values of the input.
//
// This is a test op for dynamic output results.
static Expected<DenseHostTensor> OddCollectorOp(
    const DenseHostTensor& input, const ExecutionContext& exec_ctx) {
  if (input.shape().GetRank() != 1)
    return MakeStringError("expected a 1D tensor input");
  if (input.dtype().kind() != DType::I32)
    return MakeStringError("expected a i32 element type");

  // Figure out how big the result tensor will be.
  DHTArrayView<int32_t> input_view(&input);
  size_t result_size = 0;
  for (auto& elt : input_view)
    if (elt & 1) ++result_size;

  // Allocate the result.
  auto result = DenseHostTensor::CreateUninitialized<int32_t>(
      TensorShape{result_size}, exec_ctx.host());
  if (!result.hasValue()) return MakeStringError("cannot allocate tensor");

  // Fill in the result.
  MutableDHTArrayView<int32_t> result_view(result.getPointer());
  size_t result_elt = 0;
  for (auto& elt : input_view)
    if (elt & 1) result_view[result_elt++] = elt;

  return std::move(result.getValue());
}

//===----------------------------------------------------------------------===//
// test.create_from_scalar op
//===----------------------------------------------------------------------===//

template <typename T>
static AsyncValueRef<HostTensor> DoCreateFromScalar(
    const TensorMetadata& dest_md, T value, const ExecutionContext& exec_ctx) {
  return MakeAvailableAsyncValueRef<ScalarHostTensor<T>>(exec_ctx.host(),
                                                         dest_md, value);
}

// result = test.create_from_scalar(value=V, shape=Shape)
//
static AsyncValueRef<HostTensor> TestCreateFromScalarOp(
    const OpAttrsRef& attrs, const TensorMetadata& dest_md,
    const ExecutionContext& exec_ctx) {
  auto& value = attrs.GetRawAsserting("value");
  assert(!value.IsArray() && "shape function failure");

  switch (value.type) {
    default:
      assert(0 && "shape function failure");
      return {};
#define OP_ATTR_TYPE(ENUM, CPP_TYPE)                                    \
  case OpAttrType::ENUM:                                                \
    return DoCreateFromScalar(dest_md, value.GetScalarData<CPP_TYPE>(), \
                              exec_ctx);
#include "tfrt/core_runtime/op_attr_type.def"
  }
}

//===----------------------------------------------------------------------===//
// test.add op
//===----------------------------------------------------------------------===//

namespace {
template <typename T>
AsyncValueRef<HostTensor> TestAddOpImpl(const HostTensor& lhs_ref,
                                        const HostTensor& rhs_ref,
                                        const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();

  auto* lhs = &lhs_ref;
  auto* rhs = &rhs_ref;

  // We handle Scalar+Scalar, Scalar+Dense, Dense+Dense below. Swap
  // Dense+Scalar to simplify the logic since add is commutative.
  if (isa<DenseHostTensor>(lhs) && isa<AnyScalarHostTensor>(rhs))
    std::swap(lhs, rhs);

  // Handle scalar+scalar.
  if (auto* srhs = dyn_cast<ScalarHostTensor<T>>(rhs)) {
    auto* slhs = cast<ScalarHostTensor<T>>(lhs);
    auto result = slhs->GetValue() + srhs->GetValue();
    return MakeAvailableAsyncValueRef<ScalarHostTensor<T>>(
        host, slhs->metadata(), result);
  }

  auto dest = DenseHostTensor::CreateUninitialized(lhs->metadata(), host);
  if (!dest)
    return MakeErrorAsyncValueRef(host, "out of memory allocating result");

  MutableDHTArrayView<T> dest_view(dest.getPointer());

  // Handle scalar+dense.
  DHTArrayView<T> rhs_view(cast<DenseHostTensor>(rhs));
  if (auto* slhs = dyn_cast<ScalarHostTensor<T>>(lhs)) {
    // Add a scalar to a dense tensor.
    auto lhs = slhs->GetValue();
    for (size_t i = 0, e = dest_view.NumElements(); i != e; ++i)
      dest_view[i] = lhs + rhs_view[i];
  } else {
    // Add two dense tensors.
    DHTArrayView<T> lhs_view(cast<DenseHostTensor>(lhs));
    for (size_t i = 0, e = dest_view.NumElements(); i != e; ++i)
      dest_view[i] = lhs_view[i] + rhs_view[i];
  }
  return MakeAvailableAsyncValueRef<DenseHostTensor>(
      host, std::move(dest.getValue()));
}
}  // namespace

// This implements the test.add op.
static AsyncValueRef<HostTensor> TestAddOp(const HostTensor& lhs,
                                           const HostTensor& rhs,
                                           const ExecutionContext& exec_ctx) {
  switch (lhs.dtype().kind()) {
    default:
      assert(0 && "shape function failure");
      return {};
#define DTYPE_TRIVIAL(ENUM) \
  case DType::ENUM:         \
    return TestAddOpImpl<TypeForDTypeKind<DType::ENUM>>(lhs, rhs, exec_ctx);
#include "tfrt/dtype/dtype.def"
  }
}

//===----------------------------------------------------------------------===//
// test.add.denseonly op
//===----------------------------------------------------------------------===//

template <typename T>
static void TestAddDenseOnlyImpl(DHTArrayView<T> lhs_view,
                                 DHTArrayView<T> rhs_view,
                                 MutableDHTArrayView<T> dest_view,
                                 size_t begin_idx, size_t end_idx) {
  // Add two dense tensors.
  for (size_t i = begin_idx; i != end_idx; ++i)
    dest_view[i] = lhs_view[i] + rhs_view[i];
}

// This implements the test.add.denseonly op, which is a simple add operation,
// but whose implementation intentionally only supports dense tensors.  This is
// used to test type conversion by the CPU device.
static Expected<DenseHostTensor> TestAddDenseOnlyOp(
    const DenseHostTensor& lhs, const DenseHostTensor& rhs,
    const ExecutionContext& exec_ctx) {
  auto dest =
      DenseHostTensor::CreateUninitialized(lhs.metadata(), exec_ctx.host());
  if (!dest) {
    return MakeStringError("out of memory allocating result");
  }

  auto* dest_ptr = dest.getPointer();
  switch (lhs.dtype().kind()) {
    default:
      assert(0 && "shape function failure");
      break;
#define DTYPE_TRIVIAL(ENUM)                                                    \
  case DType::ENUM:                                                            \
    TestAddDenseOnlyImpl<TypeForDTypeKind<DType::ENUM>>(&lhs, &rhs, dest_ptr,  \
                                                        0, lhs.NumElements()); \
    break;
#include "tfrt/dtype/dtype.def"
  }

  return std::move(dest.getValue());
}

// This implements the test.add.denseonly op, which is a simple add operation,
// but whose implementation intentionally only supports dense tensors.  This is
// used to test type conversion by the CPU device.
static AsyncValueRef<DenseHostTensor> TestAddDenseOnly2Op(
    const DenseHostTensor& lhs, const DenseHostTensor& rhs,
    const ExecutionContext& exec_ctx) {
  // Allocate the result buffer, and exit if an error occurred.
  auto dht = DenseHostTensor::MakeConstructedAsyncValueRef(lhs.metadata(),
                                                           exec_ctx.host());
  if (!dht) {
    return MakeErrorAsyncValueRef(exec_ctx.host(),
                                  "out of memory allocating result");
  }

  switch (lhs.dtype().kind()) {
    default:
      assert(0 && "shape function failure");
      break;
#define DTYPE_TRIVIAL(ENUM)                              \
  case DType::ENUM:                                      \
    TestAddDenseOnlyImpl<TypeForDTypeKind<DType::ENUM>>( \
        &lhs, &rhs, &dht.get(), 0, lhs.NumElements());   \
    break;
#include "tfrt/dtype/dtype.def"
  }
  dht.SetStateConcrete();
  return dht;
}

// This implements the test.add.denseonly op, which is a simple add operation,
// but whose implementation intentionally only supports dense tensors.  This is
// used to test type conversion by the CPU device.
static AsyncValueRef<DenseHostTensor> TestAddDenseOnly3Op(
    const DenseHostTensor& lhs, const DenseHostTensor& rhs,
    const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();
  // Allocate the result buffer, and exit if an error occurred.
  auto dht =
      DenseHostTensor::MakeConstructedAsyncValueRef(lhs.metadata(), host);
  if (!dht) {
    return MakeErrorAsyncValueRef(host, "out of memory allocating result");
  }

  // Note the captured dht is of type DenseHostTensor*. dht is guaranteed to be
  // alive since we capture the AsyncValueRef<DenseHostTensor> that contains the
  // dht in the ParallelFor call.
  auto add_impl = [dht = &dht.get(), lhs = lhs.CopyRef(), rhs = rhs.CopyRef()](
                      size_t begin_elt, size_t end_elt) mutable {
    switch (lhs.dtype().kind()) {
      default:
        assert(0 && "shape function failure");
        break;
#define DTYPE_TRIVIAL(ENUM)                                     \
  case DType::ENUM:                                             \
    return TestAddDenseOnlyImpl<TypeForDTypeKind<DType::ENUM>>( \
        &lhs, &rhs, dht, begin_elt, end_elt);
#include "tfrt/dtype/dtype.def"
    }
  };

  // Add two dense tensors in parallel. This one is intentionally using min
  // block size of 1 to trigger parallel block execution.
  ParallelFor(host).Execute(
      lhs.NumElements(), ParallelFor::BlockSizes::Min(1), std::move(add_impl),
      [dht = dht.CopyRef()]() mutable { dht.SetStateConcrete(); });

  return dht;
}

//===----------------------------------------------------------------------===//
// test.print op
//===----------------------------------------------------------------------===//

static AsyncValueRef<Chain> PrintOp(const HostTensor& input,
                                    const ExecutionContext& exec_ctx) {
  input.Print(tfrt::outs());
  tfrt::outs() << '\n';
  tfrt::outs().flush();
  return GetReadyChain(exec_ctx.host());
}

//===----------------------------------------------------------------------===//
// test.print_address op
//===----------------------------------------------------------------------===//

static AsyncValueRef<Chain> PrintAddressOp(const HostTensor& input,
                                           const ExecutionContext& exec_ctx) {
  if (auto* dht = dyn_cast<DenseHostTensor>(&input)) {
    tfrt::outs() << "DenseHostTensor: buffer=" << dht->buffer()->data();
  } else {
    tfrt::outs() << "Unsupported tensor type";
  }

  tfrt::outs() << '\n';
  tfrt::outs().flush();
  return GetReadyChain(exec_ctx.host());
}

//===----------------------------------------------------------------------===//
// test.identity op
//===----------------------------------------------------------------------===//

static DenseHostTensor IdentityOp(const HostTensor& input) {
  return llvm::cast<DenseHostTensor>(input).CopyRef();
}

//===----------------------------------------------------------------------===//
// test.async.noop op
//===----------------------------------------------------------------------===//

// Copy the input to the output and force it to happen asynchronously, for
// testing.
static RCReference<AsyncValue> AsyncNoopOp(const HostTensor& src,
                                           const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();
  auto dest_ind = MakeIndirectAsyncValue(host);

  auto copy = src.ConvertToHostTensor(host, ~uint32_t(0));

  host->EnqueueWork(
      [dest_ind = dest_ind.CopyRef(), copy = std::move(copy)]() mutable {
        dest_ind->ForwardTo(std::move(copy).ReleaseRCRef());
      });

  return dest_ind;
}

//===----------------------------------------------------------------------===//
// test.error.tensor op
//===----------------------------------------------------------------------===//

// This is an op that has a metadata function but whose op implementation
// produces an error for the tensor result.  This allows us to test handling of
// TensorHandle's that have a valid metadata but an error tensor value.
static RCReference<AsyncValue> ErrorTensorOp(const HostTensor& src,
                                             const ExecutionContext& exec_ctx) {
  return MakeErrorAsyncValueRef(exec_ctx.host(),
                                "error from test.error.tensor implementation");
}

//===----------------------------------------------------------------------===//
// Op registration
//===----------------------------------------------------------------------===//

void RegisterTestCpuOps(CpuOpRegistry* op_registry) {
  op_registry->AddOp("tfrt_test.odd_collector", TFRT_CPU_OP(OddCollectorOp),
                     CpuOpFlags::NoSideEffects);
  op_registry->AddOp("tfrt_test.create_from_scalar",
                     TFRT_CPU_OP(TestCreateFromScalarOp),
                     CpuOpFlags::NoSideEffects, {"value"});
  op_registry->AddOp("tfrt_test.add", TFRT_CPU_OP(TestAddOp),
                     CpuOpFlags::NoSideEffects | CpuOpFlags::AllowsScalar);
  op_registry->AddOp("tfrt_test.add.denseonly", TFRT_CPU_OP(TestAddDenseOnlyOp),
                     CpuOpFlags::NoSideEffects);
  op_registry->AddOp("tfrt_test.add.denseonly2",
                     TFRT_CPU_OP(TestAddDenseOnly2Op),
                     CpuOpFlags::NoSideEffects);
  op_registry->AddOp("tfrt_test.add.denseonly3",
                     TFRT_CPU_OP(TestAddDenseOnly3Op),
                     CpuOpFlags::NoSideEffects);
  op_registry->AddOp("tfrt_test.print", TFRT_CPU_OP(PrintOp),
                     CpuOpFlags::AllowsScalar | CpuOpFlags::AllowsString |
                         CpuOpFlags::AllowsCoo);
  op_registry->AddOp("tfrt_test.print_address", TFRT_CPU_OP(PrintAddressOp),
                     CpuOpFlags::AllowsScalar | CpuOpFlags::AllowsCoo);
  op_registry->AddOp("tfrt_test.identity", TFRT_CPU_OP(IdentityOp),
                     CpuOpFlags::NoSideEffects);
  op_registry->AddOp("tfrt_test.async.noop", TFRT_CPU_OP(AsyncNoopOp),
                     CpuOpFlags::NoSideEffects | CpuOpFlags::AllowsScalar);
  // Register another AsyncNoopOp but with no metadata function.
  op_registry->AddOp("tfrt_test.async.noop_no_md", TFRT_CPU_OP(AsyncNoopOp),
                     CpuOpFlags::NoSideEffects | CpuOpFlags::AllowsScalar);
  op_registry->AddOp("tfrt_test.error.tensor", TFRT_CPU_OP(ErrorTensorOp),
                     CpuOpFlags::NoSideEffects | CpuOpFlags::AllowsScalar);
}

}  // namespace tfrt
