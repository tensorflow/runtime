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

// This library contains implementations for corert.executeop.

#include "tfrt/core_runtime/execute_op_impl.h"

#include "tfrt/core_runtime/core_runtime.h"
#include "tfrt/core_runtime/op_attrs.h"
#include "tfrt/core_runtime/op_handler.h"
#include "tfrt/core_runtime/tensor_handle.h"
#include "tfrt/host_context/async_value.h"
#include "tfrt/host_context/async_value_ref.h"
#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/kernel_utils.h"
#include "tfrt/host_context/sync_kernel_utils.h"
#include "tfrt/support/error_util.h"

namespace tfrt {

void SetUpOpAttrs(AggregateAttr op_attr_array, OpAttrs *op_attrs) {
  for (size_t i = 0, e = op_attr_array.GetNumElements(); i != e; ++i) {
    auto pair = op_attr_array.GetAttributeOfType<AggregateAttr>(i);
    assert(pair.GetNumElements() == 2);
    string_view key = pair.GetAttributeOfType<StringAttr>(0).GetValue();
    TypedAttrBase attr = pair.GetAttribute(1);

    BEFAttributeType attribute_type = attr.type();
    if (IsArrayAttribute(attribute_type)) {
      auto type = GetOpAttrTypeFromBEFAttributeType(
          GetElementAttributeType(attribute_type));
      auto array_attr = attr.cast<ArrayAttr>();
      op_attrs->SetRaw(key, array_attr.GetElements(),
                       array_attr.GetNumElements(), type);
    } else if (IsDenseAttribute(attribute_type)) {
      auto r = op_attrs->Set(key, attr.cast<DenseAttr>());
      assert(r);
      (void)r;
    } else if (IsDataTypeAttribute(attribute_type)) {
      switch (GetDataType(attribute_type)) {
        case DType::I1:
          op_attrs->Set(key, attr.cast<I1Attr>().GetValue());
          break;
        case DType::I32:
          op_attrs->Set(key, attr.cast<I32Attr>().GetValue());
          break;
        case DType::I64:
          op_attrs->Set(key, attr.cast<I64Attr>().GetValue());
          break;
        case DType::F32:
          op_attrs->Set(key, attr.cast<F32Attr>().GetValue());
          break;
        case DType::F64:
          op_attrs->Set(key, attr.cast<F64Attr>().GetValue());
          break;
        case DType::String:
          op_attrs->SetString(key, attr.cast<StringAttr>().GetValue());
          break;
        default:
          llvm_unreachable("unknown attribute type");
      }
    } else {
      switch (attribute_type) {
        case BEFAttributeType::kType: {
          auto type_attr = attr.cast<TypeAttr>();
          DType::Kind type = type_attr.GetValue();
          op_attrs->Set(key, GetOpAttrTypeFromDType(type));
          break;
        }
        case BEFAttributeType::kShape:
          op_attrs->Set(key, attr.cast<ShapeAttr>());
          break;
        case BEFAttributeType::kAggregate:
          op_attrs->Set(key, attr.cast<AggregateAttr>());
          break;
        default:
          llvm_unreachable("unknown attribute type");
      }
    }
  }
}

// Set up `op_attrs` with binary attributes in `op_attr_func_array`.
// `op_attr_func_array` is an array of string that denotes function attributes.
void SetUpOpFuncAttrs(AggregateAttr op_func_attr_array, OpAttrs *op_attrs) {
  for (size_t i = 0, e = op_func_attr_array.GetNumElements(); i != e; ++i) {
    auto pair = op_func_attr_array.GetAttributeOfType<AggregateAttr>(i);
    assert(pair.GetNumElements() == 2);
    string_view key = pair.GetAttributeOfType<StringAttr>(0).GetValue();
    TypedAttrBase attr = pair.GetAttribute(1);

    // The function attribute is string typed.
    assert(IsDataTypeAttribute(attr.type()) &&
           GetDataType(attr.type()) == DType::String);
    auto string_attr = attr.cast<StringAttr>().GetValue();
    op_attrs->SetFunc(key, {string_attr});
  }
}

void AsyncWaitForResultsFromTensorHandles(
    MutableArrayRef<RCReference<AsyncValue>> results,
    MutableArrayRef<TensorHandle> result_ths) {
  // Return all of the TensorHandles in AsyncValue's.
  for (size_t i = 0, e = result_ths.size(); i != e; ++i) {
    auto &th_ref = result_ths[i];
    auto *tensor_av = th_ref.GetAsyncTensor();

    // Only set the AsyncValue of TensorHandle to be available when the
    // underlying tensor is available. This is to avoid unnecessary async
    // dispatches in BEF execution.
    auto state = tensor_av->state();
    if (state.IsError()) {
      // Here we don't propagate errors to all results. We just faithfully
      // propagate the results from the op implementation. It is up to the op
      // implementation on how to set errors in its results.
      results[i]->SetError(tensor_av->GetError());
    } else if (state.IsAvailable()) {
      results[i]->emplace<TensorHandle>(std::move(th_ref));
    } else {
      tensor_av->AndThen([tensor_av, result = results[i].CopyRef(),
                          th_ref = std::move(th_ref)]() mutable {
        if (tensor_av->IsError()) {
          result->SetError(tensor_av->GetError());
          return;
        }
        result->emplace<TensorHandle>(std::move(th_ref));
      });
    }
  }
}

void ExecuteOpImpl(CoreRuntimeOp op, ArrayRef<AsyncValue *> args,
                   AsyncValueRef<Chain> *op_chain,
                   MutableArrayRef<RCReference<AsyncValue>> results,
                   AggregateAttr op_attr_array,
                   AggregateAttr op_func_attr_array,
                   const ExecutionContext &exec_ctx) {
  SmallVector<TensorHandle, 8> th_args;
  th_args.reserve(args.size());

  // Move the TensorHandle if we know that we are the last user of the async
  // value. This enables buffer forwading in ops implementation, because we
  // do not add redundant references to the tensor async value.
  for (auto *arg : args) {
    if (arg->IsUnique()) {
      th_args.push_back(std::move(arg->get<TensorHandle>()));
    } else {
      th_args.push_back(arg->get<TensorHandle>().CopyRef());
    }
  }

  SmallVector<TensorHandle, 8> result_ths;
  result_ths.resize(results.size());

  // Set up OpAttrs.
  OpAttrs op_attrs;
  SetUpOpAttrs(op_attr_array, &op_attrs);

  // Set up OpAttrs specifically for function attributes.
  SetUpOpFuncAttrs(op_func_attr_array, &op_attrs);

  op(exec_ctx, th_args, OpAttrsRef(op_attrs), result_ths, op_chain);

  AsyncWaitForResultsFromTensorHandles(results, result_ths);
}

void ExecuteOpImplSync(const CoreRuntimeOp &op,
                       RepeatedSyncArguments<TensorHandle> args,
                       AsyncValueRef<Chain> *op_chain, SyncKernelFrame *frame,
                       AggregateAttr op_attr_array,
                       const ExecutionContext &exec_ctx) {
  SmallVector<TensorHandle, 8> th_args;
  th_args.reserve(args.size());

  for (auto &arg : args) {
    th_args.push_back(arg.CopyRef());
  }

  SmallVector<TensorHandle, 8> result_ths;
  result_ths.resize(frame->GetNumResults());

  // Set up OpAttrs.
  OpAttrs op_attrs;
  SetUpOpAttrs(op_attr_array, &op_attrs);

  op(exec_ctx, th_args, OpAttrsRef(op_attrs), result_ths, op_chain);

  // Return all of the TensorHandles in AsyncValue's.
  for (size_t i = 0, e = result_ths.size(); i != e; ++i) {
    auto &th_ref = result_ths[i];
    frame->EmplaceResultAt<TensorHandle>(i, std::move(th_ref));
  }
}

}  // namespace tfrt
