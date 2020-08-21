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

//===- mnist_tensor_kernels.cc --------------------------------------------===//
//
// This file defines the tensor kernels for mnist.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <limits>

#include "../../kernels/cpu_kernels.h"
#include "tfrt/common/compat/eigen/eigen_dtype.h"
#include "tfrt/common/compat/eigen/eigen_kernel.h"
#include "tfrt/common/compat/eigen/tensor_types.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/core_runtime/op_utils.h"
#include "tfrt/cpu/core_runtime/cpu_op_registry.h"
#include "tfrt/cpu/ops/test/cpu_ops_and_kernels.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/tensor/btf.h"
#include "tfrt/tensor/dense_host_tensor_view.h"

using ::tfrt::compat::AsEigenConstTensor;
using ::tfrt::compat::AsEigenTensor;
using ::tfrt::compat::BinaryEigenKernelAsync;
using ::tfrt::compat::EigenHostContext;
using ::tfrt::compat::KeepBuffers;
using ::tfrt::compat::NullaryEigenKernelAsync;
using ::tfrt::compat::UnaryEigenKernelAsync;

namespace tfrt {

//===----------------------------------------------------------------------===//
// mnist.matmul op and kernels
//===----------------------------------------------------------------------===//

static void MatMulOp(const DenseHostTensor& lhs, const DenseHostTensor& rhs,
                     const OpAttrsRef& attrs, const TensorMetadata& dest_md,
                     RCReference<AsyncValue>* dest,
                     const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();

  auto dest_alloc = DenseHostTensor::CreateUninitialized(dest_md, host);
  if (!dest_alloc) {
    *dest = EmitErrorAsync(exec_ctx, "out of memory allocating result");
    return;
  }

  auto& dest_tensor = dest_alloc.getValue();

  // Handle attributes.
  bool transpose_a = attrs.GetAsserting<bool>("transpose_a");
  bool transpose_b = attrs.GetAsserting<bool>("transpose_b");

  // Computes C = A @ B.
  switch (lhs.dtype().kind()) {
    default:
      *dest = EmitErrorAsync(exec_ctx, "unsupported dtype for matmul");
      return;
#define DTYPE_TRIVIAL(ENUM)                                \
  case DType::ENUM:                                        \
    cpu::CallMatMulKernel<TypeForDTypeKind<DType::ENUM>>(  \
        lhs, rhs, &dest_tensor, transpose_a, transpose_b); \
    break;
#include "tfrt/dtype/dtype.def"
  }

  *dest =
      MakeAvailableAsyncValueRef<DenseHostTensor>(host, std::move(dest_tensor));
}

//===----------------------------------------------------------------------===//
// mnist.relu op and kernels
//===----------------------------------------------------------------------===//

template <typename T>
static AsyncValueRef<T> WaitForChain(T&& t, AsyncValueRef<Chain> chain,
                                     const ExecutionContext& exec_ctx) {
  auto result =
      MakeConstructedAsyncValueRef<T>(exec_ctx.host(), std::forward<T>(t));
  auto* chain_av = chain.GetAsyncValue();
  chain_av->AndThen(
      [chain = std::move(chain), result = result.CopyRef()]() mutable {
        if (chain.IsError()) {
          result.SetError(chain.GetError());
        } else {
          result.SetStateConcrete();
        }
      });
  return result;
}

static AsyncValueRef<Chain> ReluHelper(const DenseHostTensor& A,
                                       DenseHostTensor* dest,
                                       const ExecutionContext& exec_ctx) {
  switch (A.dtype().kind()) {
    default:
      return EmitErrorAsync(exec_ctx, "unsupported dtype for relu");
#define DTYPE_NUMERIC(ENUM) \
  case DType::ENUM:         \
    return cpu::Relu<EigenTypeForDTypeKind<DType::ENUM>>(A, dest, exec_ctx);
#include "tfrt/dtype/dtype.def"
  }
}

// Computes A = Relu(A).
template <typename T>
static AsyncValueRef<Chain> ReluInPlace(DenseHostTensor* A,
                                        const ExecutionContext& exec_ctx) {
  auto fn = [](auto& a) { return a.cwiseMax(static_cast<T>(0)); };
  return NullaryEigenKernelAsync<T>(A, std::move(fn), exec_ctx);
}

static AsyncValueRef<Chain> ReluInPlaceHelper(
    DenseHostTensor* A, const ExecutionContext& exec_ctx) {
  switch (A->dtype().kind()) {
    default:
      return EmitErrorAsync(exec_ctx, "unsupported dtype for relu");
#define DTYPE_NUMERIC(ENUM) \
  case DType::ENUM:         \
    return ReluInPlace<EigenTypeForDTypeKind<DType::ENUM>>(A, exec_ctx);
#include "tfrt/dtype/dtype.def"
  }
}

static AsyncValueRef<DenseHostTensor> ReluOp(Argument<DenseHostTensor> A,
                                             const TensorMetadata& B_md,
                                             const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();

  if (A.value()->IsUnique() && A->buffer()->IsUnique()) {
    DenseHostTensor dest(B_md, A->buffer().CopyRef());
    AsyncValueRef<Chain> chain = ReluInPlaceHelper(&dest, exec_ctx);
    return WaitForChain(std::move(dest), std::move(chain), exec_ctx);
  } else {
    auto dest = DenseHostTensor::CreateUninitialized(B_md, host);
    if (!dest) {
      return EmitErrorAsync(exec_ctx, "out of memory allocating result");
    }

    AsyncValueRef<Chain> chain =
        ReluHelper(A.get(), dest.getPointer(), exec_ctx);

    return WaitForChain(std::move(dest).getValue(), std::move(chain), exec_ctx);
  }
}

//===----------------------------------------------------------------------===//
// mnist.add kernels
//===----------------------------------------------------------------------===//

// Computes C = A + B.
template <typename T>
static AsyncValueRef<Chain> ElementwiseAdd(
    const DenseHostTensor& A, const DenseHostTensor& B,
    // `C` supplies the buffer for writing the output
    DenseHostTensor* C, const ExecutionContext& exec_ctx) {
  auto fn = [](auto& a, auto& b, auto& c) { return a + b; };
  return AsyncValueRef<Chain>(
      BinaryEigenKernelAsync<T, T>(A, B, C, std::move(fn), exec_ctx));
}

// Computes B += A.
// TODO(rmlarsen): Should we prefer B += A over C = A + B? Should we implement
// both?
template <typename T>
static AsyncValueRef<Chain> ElementwiseAddInPlace(
    Argument<DenseHostTensor> A,
    // `B` supplies the buffer for writing the output
    Argument<DenseHostTensor> B, const ExecutionContext& exec_ctx) {
  auto fn = [](auto& a, auto& b) { return a + b; };
  return UnaryEigenKernelAsync<T, T>(A.get(), &B.get(), std::move(fn),
                                     exec_ctx);
}

//===----------------------------------------------------------------------===//
// mnist.equal op and kernels
//===----------------------------------------------------------------------===//

template <typename T>
static AsyncValueRef<Chain> ElementwiseEqual(
    const DenseHostTensor& A, const DenseHostTensor& B,
    // `C` supplies the buffer for writing the output
    DenseHostTensor* C, const ExecutionContext& exec_ctx) {
  auto fn = [](auto& A, auto& B, auto& C) {
    return (A == B).template cast<T>();
  };
  return BinaryEigenKernelAsync<T, T>(A, B, C, std::move(fn), exec_ctx);
}

// Computes B = (A == B).
template <typename T>
static AsyncValueRef<Chain> ElementwiseEqualInPlace(
    Argument<DenseHostTensor> A,
    // `B` supplies the buffer for writing the output
    Argument<DenseHostTensor> B, const ExecutionContext& exec_ctx) {
  auto fn = [](auto& A, auto& B) { return (A == B).template cast<T>(); };
  return UnaryEigenKernelAsync<T, T>(A.get(), &B.get(), std::move(fn),
                                     exec_ctx);
}

static void ElementwiseEqualOp(const DenseHostTensor& lhs,
                               const DenseHostTensor& rhs,
                               RCReference<AsyncValue>* dest_tensor,
                               const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();
  auto dest = DenseHostTensor::CreateUninitialized(lhs.metadata(), host);
  if (!dest) {
    *dest_tensor = EmitErrorAsync(exec_ctx, "out of memory allocating result");
    return;
  }

  *dest_tensor =
      MakeUnconstructedAsyncValueRef<DenseHostTensor>(host).ReleaseRCRef();

  AsyncValueRef<Chain> chain;
  auto* dest_dht = dest.getPointer();
  switch (lhs.dtype().kind()) {
    default:
      chain = EmitErrorAsync(exec_ctx, "unsupported dtype for equal");
      break;
#define DTYPE_NUMERIC(ENUM)                                       \
  case DType::ENUM:                                               \
    chain = ElementwiseEqual<EigenTypeForDTypeKind<DType::ENUM>>( \
        lhs, rhs, dest_dht, exec_ctx);                            \
    break;
#include "tfrt/dtype/dtype.def"
  }

  auto* chain_av = chain.GetAsyncValue();
  chain_av->AndThen([dest = std::move(dest).getValue(),
                     chain = std::move(chain),
                     dest_tensor = dest_tensor->CopyRef()]() mutable {
    if (chain.IsError()) {
      dest_tensor->SetError(chain.GetError());
    } else {
      dest_tensor->emplace<DenseHostTensor>(std::move(dest));
    }
  });
}

//===----------------------------------------------------------------------===//
// mnist.cast op and kernels
//===----------------------------------------------------------------------===//

template <typename Tin, typename Tout>
static AsyncValueRef<Chain> Cast(const DenseHostTensor& A, DenseHostTensor* B,
                                 const ExecutionContext& exec_ctx) {
  auto fn = [](auto& a, auto& b) { return a.template cast<Tout>(); };
  return UnaryEigenKernelAsync<Tin, Tout>(A, B, std::move(fn), exec_ctx);
}

template <typename Tout>
static AsyncValueRef<Chain> CastForOutType(const DenseHostTensor& A,
                                           DenseHostTensor* B,
                                           const ExecutionContext& exec_ctx) {
  switch (A.dtype().kind()) {
    default:
      return EmitErrorAsync(exec_ctx, "unsupported dtype for cast");
#define DTYPE_NUMERIC(ENUM) \
  case DType::ENUM:         \
    return Cast<EigenTypeForDTypeKind<DType::ENUM>, Tout>(A, B, exec_ctx);
#include "tfrt/dtype/dtype.def"
  }
}

static void CastOp(const DenseHostTensor& A, const TensorMetadata& B_md,
                   RCReference<AsyncValue>* B_tensor,
                   const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();

  auto dest = DenseHostTensor::CreateUninitialized(B_md, host);
  if (!dest) {
    *B_tensor = EmitErrorAsync(exec_ctx, "out of memory allocating result");
    return;
  }

  *B_tensor =
      MakeUnconstructedAsyncValueRef<DenseHostTensor>(host).ReleaseRCRef();

  AsyncValueRef<Chain> chain;
  switch (B_md.dtype.kind()) {
    default:
      *B_tensor = EmitErrorAsync(exec_ctx, "unsupported dtype for cast");
      return;
#define DTYPE_NUMERIC(ENUM)                                     \
  case DType::ENUM:                                             \
    chain = CastForOutType<EigenTypeForDTypeKind<DType::ENUM>>( \
        A, dest.getPointer(), exec_ctx);                        \
    break;
#include "tfrt/dtype/dtype.def"
  }

  auto* chain_av = chain.GetAsyncValue();
  chain_av->AndThen([dest = std::move(dest).getValue(),
                     chain = std::move(chain),
                     B_tensor = B_tensor->CopyRef()]() mutable {
    if (chain.IsError()) {
      B_tensor->SetError(chain.GetError());
    } else {
      B_tensor->emplace<DenseHostTensor>(std::move(dest));
    }
  });
}

//===----------------------------------------------------------------------===//
// mnist.broadcast op and kernels
//===----------------------------------------------------------------------===//

// Computes B = tf.broadcast_to(A, tf.shape(B))
// A should be a 1-D tensor. B's last dimension should have the same size as A.
//
// As example usage, this kernel can be used to broatcast 1-D channel mean
// vector to shape of an image of NHWC format. But this can not be directly used
// to broadcast the 1-D channel mean vector to the shape of an image of NCHW
// format.
template <typename T, int N>
void Broadcast1DKernel(const DenseHostTensor& A, DenseHostTensor* B) {
  assert(A.shape().GetRank() == 1 && "only 1-D tensor is supported");
  DHTIndexableView<T, 1> A_view(&A);
  MutableDHTIndexableView<T, N> B_view(B);

  Eigen::array<Eigen::Index, N> reshape_dims;
  for (int i = 0; i < N - 1; i++) {
    reshape_dims[i] = static_cast<Eigen::Index>(1);
  }
  reshape_dims[N - 1] = static_cast<Eigen::Index>(A_view.FixedShape()[0]);

  Eigen::array<Eigen::Index, N> broadcast_dims;
  for (int i = 0; i < N - 1; i++) {
    broadcast_dims[i] = static_cast<Eigen::Index>(B_view.FixedShape()[i]);
  }
  broadcast_dims[N - 1] = static_cast<Eigen::Index>(1);

  auto in = AsEigenConstTensor(A_view);
  auto out = AsEigenTensor(B_view);
  out = in.reshape(reshape_dims).broadcast(broadcast_dims);
}

// Returns tf.broadcast_to(A, target_shape)
// A should be a 1-D tensor. target_shape's last dimension should have the same
// size as A.
template <typename T, int N>
static Expected<DenseHostTensor> Broadcast1D(const DenseHostTensor& A,
                                             const TensorShape& target_shape,
                                             const ExecutionContext& exec_ctx) {
  // TODO(fishx): Try to reuse Metadata fn.
  // This broadcast specializes for MNIST bias.
  DHTIndexableView<T, 1> A_view(&A);
  ssize_t target_dims[N];
  target_shape.GetDimensions(target_dims);
  const auto& shape_A = A_view.FixedShape();
  if (target_dims[N - 1] != shape_A[0]) {
    return MakeStringError(
        "broadcast input tensor dimension and target shape mismatch: ", shape_A,
        " vs. ", target_shape);
  }

  auto tensor =
      DenseHostTensor::CreateUninitialized<T>(target_shape, exec_ctx.host());
  if (!tensor.hasValue())
    return MakeStringError("cannot allocate result tensor");

  Broadcast1DKernel<T, N>(A, tensor.getPointer());

  return std::move(*tensor);
}

// Computes dest = tf.broadcast_to(src, dest_md.shape)
// src should be a 1-D tensor. dest_md should have a 2-D shape and its last
// dimension should have same size as src.
static void Broadcast1DOp(const DenseHostTensor& src,
                          const TensorMetadata& dest_md,
                          RCReference<AsyncValue>* dest,
                          const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();

  auto dest_alloc = DenseHostTensor::CreateUninitialized(dest_md, host);
  if (!dest_alloc) {
    *dest = EmitErrorAsync(exec_ctx, "out of memory allocating result");
    return;
  }

  auto& dest_tensor = dest_alloc.getValue();

  switch (src.dtype().kind()) {
    default:
      *dest = EmitErrorAsync(exec_ctx, "unsupported dtype for broadcast");
      return;
#define DTYPE_NUMERIC(ENUM)                                                 \
  case DType::ENUM:                                                         \
    Broadcast1DKernel<EigenTypeForDTypeKind<DType::ENUM>, 2>(src,           \
                                                             &dest_tensor); \
    break;
#include "tfrt/dtype/dtype.def"
  }
  *dest =
      MakeAvailableAsyncValueRef<DenseHostTensor>(host, std::move(dest_tensor));
}

//===----------------------------------------------------------------------===//
// mnist.argmax op and kernels
//===----------------------------------------------------------------------===//

// Take argmax along axis = Axis.
// Argmax result tensor must be int32, not float.
template <typename T, size_t Rank, size_t Axis>
static void ArgmaxKernel(const DenseHostTensor& A, DenseHostTensor* B) {
  DHTIndexableView<T, Rank> A_view(&A);
  MutableDHTIndexableView<int32_t, Rank - 1> B_view(B);

  auto in = AsEigenConstTensor(A_view);
  auto out = AsEigenTensor(B_view);
  out = in.argmax(Axis).template cast<int32_t>();
}

template <typename T, size_t Rank, size_t Axis = 1>
static Expected<DenseHostTensor> Argmax(const DenseHostTensor& A,
                                        const ExecutionContext& exec_ctx) {
  // TODO(fishx): Try to reuse Metadata fn.
  static_assert(Axis < Rank, "Axis < Rank");
  DHTIndexableView<T, Rank> A_view(&A);
  const auto& shape_A = A_view.FixedShape();
  std::array<ssize_t, Rank - 1> result_dims;
  size_t out_axis = 0;
  for (size_t in_axis = 0; in_axis < Rank; ++in_axis) {
    if (in_axis != Axis) {
      result_dims[out_axis++] = shape_A[in_axis];
    }
  }

  auto tensor = DenseHostTensor::CreateUninitialized<int32_t>(
      TensorShape(result_dims), exec_ctx.host());
  if (!tensor.hasValue()) {
    return MakeStringError("Cannot allocate result tensor.");
  }

  ArgmaxKernel<T, Rank, Axis>(A, tensor.getPointer());

  return std::move(*tensor);
}

template <size_t Rank, size_t Axis>
static void ArgmaxForAxisRank(const DenseHostTensor& A, DenseHostTensor* B) {
  switch (A.dtype().kind()) {
    default:
      assert(0 && "shape function mismatch");
#define DTYPE_NUMERIC(ENUM)                                             \
  case DType::ENUM:                                                     \
    ArgmaxKernel<EigenTypeForDTypeKind<DType::ENUM>, Rank, Axis>(A, B); \
    break;
#include "tfrt/dtype/dtype.def"
  }
}

template <size_t Axis>
static void ArgmaxForAxis(const DenseHostTensor& A, DenseHostTensor* B) {
  switch (A.shape().GetRank()) {
    case Axis + 1:
      return ArgmaxForAxisRank<Axis + 1, Axis>(A, B);
    case Axis + 2:
      return ArgmaxForAxisRank<Axis + 2, Axis>(A, B);
    default:
      assert(0 && "shape function mismatch");
  }
}

static void ArgmaxOp(const DenseHostTensor& src, const OpAttrsRef& attrs,
                     const TensorMetadata& dest_md,
                     RCReference<AsyncValue>* dest,
                     const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();

  auto dest_alloc = DenseHostTensor::CreateUninitialized(dest_md, host);
  if (!dest_alloc) {
    *dest = EmitErrorAsync(exec_ctx, "out of memory allocating result");
    return;
  }

  auto& dest_tensor = dest_alloc.getValue();

  switch (attrs.GetAsserting<int32_t>("axis")) {
    case 1:
      ArgmaxForAxis<1>(src, &dest_tensor);
      break;
    default:
      *dest = EmitErrorAsync(exec_ctx, "unsupported axis for argmax");
      return;
  }

  *dest =
      MakeAvailableAsyncValueRef<DenseHostTensor>(host, std::move(dest_tensor));
}

//===----------------------------------------------------------------------===//
// mnist.reduce_mean op and kernels
//===----------------------------------------------------------------------===//

template <typename T, size_t Rank, size_t Axis>
static void ReduceMeanKernel(const DenseHostTensor& A, DenseHostTensor* B) {
  DHTIndexableView<T, Rank> A_view(&A);
  MutableDHTIndexableView<T, Rank - 1> B_view(B);

  Eigen::IndexList<Eigen::type2index<Axis>> reduction_dims;
  auto in = AsEigenConstTensor(A_view);
  auto out = AsEigenTensor(B_view);
  out = in.mean(reduction_dims);
}

template <typename T, size_t Rank, size_t Axis = 0>
static Expected<DenseHostTensor> ReduceMean(const DenseHostTensor& A,
                                            const ExecutionContext& exec_ctx) {
  // TODO(fishx): Try to reuse Metadata fn.
  static_assert(Axis < Rank, "Axis < Rank");
  DHTIndexableView<T, Rank> A_view(&A);
  const auto& shape_A = A_view.FixedShape();
  std::array<ssize_t, Rank - 1> result_dims;
  size_t out_axis = 0;
  for (size_t in_axis = 0; in_axis < Rank; ++in_axis) {
    if (in_axis != Axis) {
      result_dims[out_axis++] = shape_A[in_axis];
    }
  }

  auto tensor = DenseHostTensor::CreateUninitialized<T>(
      TensorShape(result_dims), exec_ctx.host());
  if (!tensor.hasValue())
    return MakeStringError("cannot allocate result tensor");

  ReduceMeanKernel<T, Rank, Axis>(A, tensor.getPointer());

  return std::move(*tensor);
}

template <size_t Rank, size_t Axis>
static void ReduceMeanForAxisRank(const DenseHostTensor& A,
                                  DenseHostTensor* B) {
  switch (A.dtype().kind()) {
    default:
      assert(0 && "shape function mismatch");
#define DTYPE_NUMERIC(ENUM)                                                  \
  case DType::ENUM:                                                          \
    return ReduceMeanKernel<EigenTypeForDTypeKind<DType::ENUM>, Rank, Axis>( \
        A, B);
#include "tfrt/dtype/dtype.def"
  }
}

template <size_t Axis>
static void ReduceMeanForAxis(const DenseHostTensor& A, DenseHostTensor* B) {
  switch (A.shape().GetRank()) {
    case Axis + 1:
      return ReduceMeanForAxisRank<Axis + 1, Axis>(A, B);
    case Axis + 2:
      return ReduceMeanForAxisRank<Axis + 2, Axis>(A, B);
    default:
      assert(0 && "shape function mismatch");
  }
}

static void ReduceMeanOp(const DenseHostTensor& src, const OpAttrsRef& attrs,
                         const TensorMetadata& dest_md,
                         RCReference<AsyncValue>* dest,
                         const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();

  auto dest_alloc = DenseHostTensor::CreateUninitialized(dest_md, host);
  if (!dest_alloc) {
    *dest = EmitErrorAsync(exec_ctx, "out of memory allocating result");
    return;
  }

  auto& dest_tensor = dest_alloc.getValue();

  switch (attrs.GetAsserting<int32_t>("axis")) {
    case 0:
      ReduceMeanForAxis<0>(src, &dest_tensor);
      break;
    default:
      *dest = EmitErrorAsync(exec_ctx, "unsupported axis for reduce_mean");
      return;
  }

  *dest =
      MakeAvailableAsyncValueRef<DenseHostTensor>(host, std::move(dest_tensor));
}

//===----------------------------------------------------------------------===//
// mnist.create_dense_tensor op and kernels
//===----------------------------------------------------------------------===//

template <typename T>
static void CreateDenseTensorForType(const OpAttrsRef& attrs,
                                     DenseHostTensor* result) {
  MutableDHTArrayView<T> dst(result);
  ArrayRef<T> values = attrs.GetArrayAsserting<T>("values");
  if (values.size() == 1) {
    dst.Fill(values[0]);
  } else {
    assert(values.size() == dst.NumElements());
    std::copy(values.begin(), values.end(), dst.Elements().begin());
  }
}

static void CreateDenseTensorOp(const OpAttrsRef& attrs,
                                const TensorMetadata& dest_md,
                                RCReference<AsyncValue>* dest,
                                const ExecutionContext& exec_ctx) {
  HostContext* host = exec_ctx.host();

  auto dest_alloc = DenseHostTensor::CreateUninitialized(dest_md, host);
  if (!dest_alloc) {
    *dest = EmitErrorAsync(exec_ctx, "out of memory allocating result");
    return;
  }

  auto& dest_tensor = dest_alloc.getValue();
  switch (dest_md.dtype.kind()) {
#define DTYPE_TRIVIAL(ENUM)                                       \
  case DType::ENUM:                                               \
    CreateDenseTensorForType<EigenTypeForDTypeKind<DType::ENUM>>( \
        attrs, &dest_tensor);                                     \
    break;
#include "tfrt/dtype/dtype.def"
    default:
      llvm_unreachable("Tensors cannot have unknown dtype");
  }

  *dest =
      MakeAvailableAsyncValueRef<DenseHostTensor>(host, std::move(dest_tensor));
}

//===----------------------------------------------------------------------===//
// mnist.relu_grad_inplace kernel
//===----------------------------------------------------------------------===//
// Relu Gradient Inplace
// gradient = activation > 0 ? gradient : 0
template <typename T>
static AsyncValueRef<Chain> ReluGradInplace(const DenseHostTensor& activation,
                                            DenseHostTensor* gradient,
                                            const ExecutionContext& exec_ctx) {
  auto fn = [](auto& a, auto& b) {
    return b * (a > static_cast<T>(0)).template cast<T>();
  };
  return UnaryEigenKernelAsync<T, T>(activation, gradient, std::move(fn),
                                     exec_ctx);
}

//===----------------------------------------------------------------------===//
// tfrt_test.slice_inplace kernel
//===----------------------------------------------------------------------===//

// Computes output = tf.slice(input, begin, output.shape()) where both input and
// output are tensors with 'Rank' dimensions. 'begin' should be a vector of size
// 'Rank'.
template <typename T, size_t Rank>
static AsyncValueRef<Chain> SliceInPlace(const DenseHostTensor& input,
                                         DHTIndexableView<int64_t, 1> begin,
                                         const Chain& chain_in,
                                         DenseHostTensor* output,
                                         const ExecutionContext& exec_ctx) {
  auto input_view = DHTIndexableView<T, Rank>(&input);
  auto output_view = MutableDHTIndexableView<T, Rank>(output);

  if (begin.NumElements() != Rank) {
    return EmitErrorAsync(
        exec_ctx,
        "the size of 'begin_index' should match the input tensor rank");
  }

  auto output_shape = output_view.FixedShape();
  Eigen::DSizes<Eigen::Index, Rank> indices;
  Eigen::DSizes<Eigen::Index, Rank> sizes;
  for (int i = 0; i < Rank; ++i) {
    indices[i] = begin.ElementAt(i);
    sizes[i] = output_shape[i];
  }
  auto input_t = AsEigenConstTensor(input_view);
  auto output_t = AsEigenTensor(output_view);
  auto expr = input_t.slice(indices, sizes);
  return AsyncAssign(
      exec_ctx.host()->GetOrCreateSharedContext<EigenHostContext>(),
      std::move(output_t), std::move(expr), KeepBuffers::alive(&input, output));
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

template <typename T>
static void RegisterMNISTTensorKernelsForType(KernelRegistry* registry,
                                              const std::string& suffix) {
  registry->AddKernel("tfrt_test.relu." + suffix, TFRT_KERNEL(cpu::Relu<T>));
  registry->AddKernel("tfrt_test.relu_inplace." + suffix,
                      TFRT_KERNEL(ReluInPlace<T>));
  registry->AddKernel("tfrt_test.add." + suffix,
                      TFRT_KERNEL(ElementwiseAdd<T>));
  registry->AddKernel("tfrt_test.add_inplace." + suffix,
                      TFRT_KERNEL(ElementwiseAddInPlace<T>));
  registry->AddKernel("tfrt_test.equal." + suffix,
                      TFRT_KERNEL(ElementwiseEqual<T>));
  registry->AddKernel("tfrt_test.equal_inplace." + suffix,
                      TFRT_KERNEL(ElementwiseEqualInPlace<T>));
  registry->AddKernel("tfrt_test.matmul." + suffix + ".2",
                      TFRT_KERNEL(cpu::MatMul2D<T>));
  registry->AddKernel("tfrt_test.broadcast." + suffix + ".2",
                      TFRT_KERNEL(Broadcast1D<T, 2>));
  registry->AddKernel("tfrt_test.broadcast." + suffix + ".3",
                      TFRT_KERNEL(Broadcast1D<T, 3>));
  registry->AddKernel("tfrt_test.reduce_mean." + suffix + ".1",
                      TFRT_KERNEL(ReduceMean<T, 1>));
  registry->AddKernel("tfrt_test.argmax." + suffix + ".2",
                      TFRT_KERNEL(Argmax<T, 2>));
  registry->AddKernel("tfrt_test.relu_grad_inplace." + suffix,
                      TFRT_KERNEL(ReluGradInplace<T>));
  registry->AddKernel("tfrt_test.slice_inplace." + suffix + ".3",
                      TFRT_KERNEL(SliceInPlace<T, 3>));
}

void RegisterMNISTTensorKernels(KernelRegistry* registry) {
  RegisterMNISTTensorKernelsForType<float>(registry, "f32");
  RegisterMNISTTensorKernelsForType<int32_t>(registry, "i32");
  RegisterMNISTTensorKernelsForType<int64_t>(registry, "i64");
  registry->AddKernel("tfrt_test.cast.i32_to_f32",
                      TFRT_KERNEL(Cast<int32_t, float>));
}

void RegisterTestMnistCpuOps(CpuOpRegistry* op_registry) {
  op_registry->AddOp("tfrt_test.matmul", TFRT_CPU_OP(MatMulOp),
                     CpuOpFlags::NoSideEffects, {"transpose_a", "transpose_b"});
  op_registry->AddOp("tfrt_test.relu", TFRT_CPU_OP(ReluOp),
                     CpuOpFlags::NoSideEffects);
  op_registry->AddOp("tfrt_test.equal", TFRT_CPU_OP(ElementwiseEqualOp),
                     CpuOpFlags::NoSideEffects);
  op_registry->AddOp("tfrt_test.cast", TFRT_CPU_OP(CastOp),
                     CpuOpFlags::NoSideEffects);
  op_registry->AddOp("tfrt_test.broadcast", TFRT_CPU_OP(Broadcast1DOp),
                     CpuOpFlags::NoSideEffects);
  op_registry->AddOp("tfrt_test.argmax", TFRT_CPU_OP(ArgmaxOp),
                     CpuOpFlags::NoSideEffects, {"axis"});
  op_registry->AddOp("tfrt_test.reduce_mean", TFRT_CPU_OP(ReduceMeanOp),
                     CpuOpFlags::NoSideEffects, {"axis"});
  op_registry->AddOp("tfrt_test.create_dense_tensor",
                     TFRT_CPU_OP(CreateDenseTensorOp),
                     CpuOpFlags::NoSideEffects, {"shape", "values"});
}

}  // namespace tfrt
