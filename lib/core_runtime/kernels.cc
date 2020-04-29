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

//===- core_runtime/kernels.cc --------------------------------------------===//
//
// This library contains kernels that allows the bef_executor to drive the core
// runtime.
//
//===----------------------------------------------------------------------===//

#include "tfrt/core_runtime/kernels.h"

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/SmallString.h"
#include "tfrt/core_runtime/core_runtime.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/core_runtime/op_handler.h"
#include "tfrt/core_runtime/tensor_handle.h"
#include "tfrt/host_context/async_value.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/ref_count.h"
#include "tfrt/tensor/host_tensor.h"
#include "tfrt/tensor/string_host_tensor.h"

namespace tfrt {

// Convert a HostTensor (or subclass) into a TensorHandle for use by
// Core Runtime.
static void HTToTensorHandle(Argument<HostTensor> arg,
                             Result<TensorHandle> tensorhandle_output,
                             HostContext *host) {
  // Since we know the Tensor is present, we can access its metadata.
  tensorhandle_output.Emplace(arg->metadata(), arg.ValueRef());
}

static void TensorHandleToHT(Argument<TensorHandle> arg,
                             Result<HostTensor> ht_output, HostContext *host) {
  ht_output.Set(FormRef(arg->GetAsyncTensor()));
}

// Get TensorShape of a TensorHandle for use by Core Runtime.
static void TensorHandleToShape(Argument<TensorHandle> arg,
                                Result<TensorShape> tensorshape_result,
                                HostContext *host) {
  if (arg->IsMetadataAvailable()) {
    auto shape = arg->GetAvailableMetadata().shape;
    tensorshape_result.Emplace(std::move(shape));
    return;
  }
  // The metadata is not available yet.
  const AsyncValueRef<TensorMetadata> &metadata = arg->GetAsyncMetadata();

  auto value = tensorshape_result.AllocateIndirect();
  metadata.AndThen([value_ref = std::move(value),
                    metadata_ref = metadata.CopyRef(), host]() mutable {
    if (metadata_ref.IsError()) {
      value_ref->ForwardTo(metadata_ref.ReleaseRCRef());
      return;
    }
    auto shape = metadata_ref.get().shape;
    value_ref->ForwardTo(
        host->MakeConcreteAsyncValueRef<TensorShape>(std::move(shape)));
  });
}

// Convert a HostTensor (or subclass) into a TensorHandle for use by
// Core Runtime.
static Chain PrintTensorHandle(Argument<TensorHandle> arg) {
  llvm::SmallString<256> message;
  llvm::raw_svector_ostream(message) << arg.get() << "\n";
  printf("%s", message.c_str());
  fflush(stdout);
  return Chain();
}

static void CreateOpAttrs(Result<OpAttrs> result) { result.Emplace(); }

static Chain OpAttrsSetBool(Argument<OpAttrs> attrs, StringAttribute key,
                            Attribute<int8_t> value) {
  attrs->Set(key, static_cast<bool>(*value));
  return Chain();
}

template <typename T>
static Chain OpAttrsSet(Argument<OpAttrs> attrs, StringAttribute key,
                        Attribute<T> value) {
  attrs->Set(key, *value);
  return Chain();
}

static Chain OpAttrsSetDType(Argument<OpAttrs> attrs, StringAttribute key,
                             Attribute<BEFAttributeType> value) {
  attrs->Set(key, GetOpAttrTypeFromBEFAttributeType(*value));
  return Chain();
}

static Chain OpAttrsSetDense(Argument<OpAttrs> attrs, StringAttribute key,
                             DenseAttr value) {  // NOLINT
  attrs->Set(key, value);
  return Chain();
}

template <typename T>
static Chain OpAttrsSetArray(Argument<OpAttrs> attrs, StringAttribute key,
                             ArrayAttribute<T> value) {
  attrs->SetArray(key, value.data());
  return Chain();
}

static Chain OpAttrsSetString(Argument<OpAttrs> attrs, StringAttribute key,
                              StringAttribute value) {
  attrs->SetString(key, value.get());
  return Chain();
}

static llvm::Expected<TensorHandle> ConstStringTensor(
    ArrayAttribute<ssize_t> shape, AggregateAttribute value,
    HostContext *host) {
  TensorMetadata metadata(DType(DType::String), shape.data());

  auto tensor_ref =
      StringHostTensor::MakeConstructedAsyncValueRef(metadata, host);
  if (!tensor_ref)
    return MakeStringError("failed to allocate string host tensor");

  auto strings = tensor_ref.get().strings();
  assert(strings.size() == value.size());

  for (int i = 0, e = strings.size(); i != e; ++i) {
    strings[i] = value.GetStringAttribute(i).str();
  }

  tensor_ref.SetStateConcrete();

  return TensorHandle(metadata, std::move(tensor_ref));
}

static void ExecuteOpImpl(CoreRuntime *core_rt, OpHandler *op_handler,
                          ArrayRef<AsyncValue *> args,
                          AsyncValueRef<Chain> *op_chain,
                          MutableArrayRef<RCReference<AsyncValue>> results,
                          AggregateAttribute op_attr_array,
                          StringAttribute op_name, Location loc) {
  SmallVector<TensorHandle, 8> th_args;
  th_args.reserve(args.size());

  // TODO(clattner): This copies the input TensorHandle's.  While this is
  // correct, it would be better to *move* out of the input async value when
  // we know that we're the last user of the async value.
  for (auto *arg : args) th_args.push_back(arg->get<TensorHandle>().CopyRef());

  SmallVector<TensorHandle, 8> result_ths;
  result_ths.resize(results.size());

  // Set up OpAttrs.
  OpAttrs op_attrs;
  for (size_t i = 0, e = op_attr_array.size(); i != e; ++i) {
    auto pair = op_attr_array.GetAggregateAttribute(i);
    assert(pair.size() == 2);

    string_view key = pair.GetStringAttribute(0).get();

    auto value_and_kind = pair.GetRawAttribute(1);

    const void *value = value_and_kind.first;
    BEFAttributeType attribute_type = value_and_kind.second;

    if (IsArrayAttribute(attribute_type)) {
      auto type = GetOpAttrTypeFromBEFAttributeType(
          GetArrayAttributeElementType(attribute_type));

      op_attrs.SetRaw(key, value, DecodeArraySizeFromBEFAttributes(value),
                      type);
    } else if (attribute_type == BEFAttributeType::kString) {
      op_attrs.SetRaw(key, value, DecodeArraySizeFromBEFAttributes(value),
                      OpAttrType::CHAR);
    } else if (attribute_type == BEFAttributeType::kType) {
      BEFAttributeType type = *static_cast<const BEFAttributeType *>(value);
      assert(IsFixedAttribute(type));
      op_attrs.Set(key, GetOpAttrTypeFromBEFAttributeType(type));
    } else if (attribute_type == BEFAttributeType::kDenseElements) {
      DenseAttr dense_attr(value);
      auto r = op_attrs.Set(key, dense_attr);
      assert(r);
      (void)r;
    } else {
      assert(IsScalarAttribute(attribute_type));
      auto type = GetOpAttrTypeFromBEFAttributeType(attribute_type);
      op_attrs.SetRaw(key, value, OpAttrsRawEntry::kScalarSentinel, type);
    }
  }

  core_rt->Execute(op_name, op_handler, loc, th_args, OpAttrsRef(op_attrs),
                   result_ths, op_chain);

  // Return all of the TensorHandles in AsyncValue's.
  for (size_t i = 0, e = result_ths.size(); i != e; ++i) {
    auto &th_ref = result_ths[i];
    results[i]->emplace<TensorHandle>(std::move(th_ref));
  }
}

// ExecuteOp executes the `op_name` operation on the `op_handler`.
static void ExecuteOp(Argument<OpHandler *> op_handler, RemainingArguments args,
                      RemainingResults results,
                      AggregateAttribute op_attr_array, StringAttribute op_name,
                      KernelErrorHandler handler, KernelFrame *frame) {
  auto *host = frame->GetHostContext();
  auto *core_rt = CoreRuntime::GetFromHostContext(host);
  if (!core_rt) return handler.ReportError("no CoreRuntime available");

  for (int b = 0, e = results.size(); b < e; ++b)
    results.AllocateAt<TensorHandle>(b);

  ExecuteOpImpl(core_rt, op_handler.get(), args.values(),
                /*op_chain =*/nullptr, results.values(), op_attr_array, op_name,
                frame->GetLocation());
}

// ExecuteOpSeq executes the `op_name` operation on the `op_handler`. It takes
// an `in_op_chain` and produces an `out_op_chain` for sequencing op execution.
// The execution is only started when `in_op_chain` is ready, and the
// `out_op_chain` is ready only after the execution is finished.
static void ExecuteOpSeq(Argument<OpHandler *> op_handler,
                         Argument<Chain> in_op_chain, RemainingArguments args,
                         Result<Chain> out_op_chain, RemainingResults results,
                         AggregateAttribute op_attr_array,
                         StringAttribute op_name, KernelErrorHandler handler,
                         KernelFrame *frame) {
  auto *host = frame->GetHostContext();
  auto *core_rt = CoreRuntime::GetFromHostContext(host);
  if (!core_rt) return handler.ReportError("no CoreRuntime available");

  for (int b = 0, e = results.size(); b < e; ++b)
    results.AllocateAt<TensorHandle>(b);

  SmallVector<AsyncValue *, 4> async_args;
  if (!op_handler.value()->IsConcrete())
    async_args.push_back(op_handler.value());
  for (auto *arg_av : args.values())
    if (!arg_av->IsConcrete()) async_args.push_back(arg_av);

  // If all arguments except in_op_chain are ready, we can just execute the op.
  if (async_args.empty()) {
    auto op_chain = in_op_chain.ValueRef();
    ExecuteOpImpl(core_rt, op_handler.get(), args.values(), &op_chain,
                  results.values(), op_attr_array, op_name,
                  frame->GetLocation());
    out_op_chain.Set(std::move(op_chain));
    return;
  }

  // Otherwise, we need to create references to all arguments and asynchronouly
  // execute the op when they are ready.

  SmallVector<AsyncValueRef<TensorHandle>, 4> arg_refs;
  for (auto *av : args.values()) {
    arg_refs.push_back(AsyncValueRef<TensorHandle>(FormRef(av)));
  }

  SmallVector<RCReference<AsyncValue>, 4> result_refs;
  for (auto &av : results.values()) {
    result_refs.push_back(av.CopyRef());
  }

  host->RunWhenReady(
      async_args,
      [core_rt, op_handler = op_handler.ValueRef(),
       op_chain = in_op_chain.ValueRef(), arg_refs = std::move(arg_refs),
       result_refs = std::move(result_refs),
       out_op_chain = out_op_chain.Allocate(), op_name, op_attr_array,
       loc = frame->GetLocation()]() mutable {
        auto propgate_error = [&](const DecodedDiagnostic &diag) {
          out_op_chain.SetError(diag);
          for (auto &result_ref : result_refs) result_ref->SetError(diag);
        };

        if (op_handler.IsError()) return propgate_error(op_handler.GetError());
        if (op_chain.IsError()) return propgate_error(op_chain.GetError());

        SmallVector<AsyncValue *, 4> arg_avs;
        for (const auto &arg_ref : arg_refs) {
          if (arg_ref.IsError()) return propgate_error(arg_ref.GetError());
          arg_avs.push_back(arg_ref.GetAsyncValue());
        }

        ExecuteOpImpl(core_rt, op_handler.get(), arg_avs, &op_chain,
                      result_refs, op_attr_array, op_name, loc);

        auto *op_chain_av = op_chain.GetAsyncValue();
        op_chain_av->AndThen([op_chain = std::move(op_chain),
                              out_op_chain = out_op_chain.CopyRef()]() {
          // TODO(chky): we should have a version of AndThen that passes the
          // resolved state into the waiter.
          if (op_chain.IsError()) {
            out_op_chain.SetError(op_chain.GetError());
          } else {
            out_op_chain.emplace();
          }
        });
      });
}

static tfrt::Expected<OpHandler *> GetOpHandler(StringAttribute op_handler_name,
                                                HostContext *host) {
  auto *runtime = CoreRuntime::GetFromHostContext(host);
  assert(runtime);

  if (auto *op_handler = runtime->GetOpHandler(op_handler_name.get())) {
    return op_handler;
  }
  return tfrt::MakeStringError("op_handler not found.");
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void RegisterCoreRuntimeKernels(KernelRegistry *registry) {
  registry->AddKernel("corert.tensorhandle_to_shape",
                      TFRT_KERNEL(TensorHandleToShape));
  registry->AddKernel("corert.ht_to_tensorhandle",
                      TFRT_KERNEL(HTToTensorHandle));
  registry->AddKernel("corert.tensorhandle_to_ht",
                      TFRT_KERNEL(TensorHandleToHT));
  registry->AddKernel("corert.print_tensorhandle",
                      TFRT_KERNEL(PrintTensorHandle));
  registry->AddKernel("corert.create_op_attrs", TFRT_KERNEL(CreateOpAttrs));
  registry->AddKernel("corert.op_attrs_set.bool", TFRT_KERNEL(OpAttrsSetBool));
  registry->AddKernel("corert.op_attrs_set.i32",
                      TFRT_KERNEL(OpAttrsSet<int32_t>));
  registry->AddKernel("corert.op_attrs_set_array.i32",
                      TFRT_KERNEL(OpAttrsSetArray<int32_t>));
  registry->AddKernel("corert.op_attrs_set_array.i64",
                      TFRT_KERNEL(OpAttrsSetArray<int64_t>));
  registry->AddKernel("corert.op_attrs_set.f32",
                      TFRT_KERNEL(OpAttrsSet<float>));
  registry->AddKernel("corert.op_attrs_set_array.f32",
                      TFRT_KERNEL(OpAttrsSetArray<float>));
  registry->AddKernel("corert.op_attrs_set.dtype",
                      TFRT_KERNEL(OpAttrsSetDType));
  registry->AddKernel("corert.op_attrs_set.dense",
                      TFRT_KERNEL(OpAttrsSetDense));
  registry->AddKernel("corert.op_attrs_set.str", TFRT_KERNEL(OpAttrsSetString));
  registry->AddKernel("corert.executeop", TFRT_KERNEL(ExecuteOp));
  registry->AddKernel("corert.executeop.seq", TFRT_KERNEL(ExecuteOpSeq));
  // TODO(fishx): Rename it to corert.get_op_handler.
  registry->AddKernel("corert.get_device", TFRT_KERNEL(GetOpHandler));
  registry->AddKernel("corert.const_string_tensor",
                      TFRT_KERNEL(ConstStringTensor));
}

}  // namespace tfrt
