/*
 * Copyright 2020 The TensorFlow Runtime Authors
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

// Declares the signature for metadata functions.  Metadata functions are
// executed synchronously by CoreRuntime to perform shape inference and error
// checking.

#ifndef TFRT_CORE_RUNTIME_OP_METADATA_FUNCTION_H_
#define TFRT_CORE_RUNTIME_OP_METADATA_FUNCTION_H_

#include "tfrt/support/forward_decls.h"

namespace tfrt {
class AsyncValue;
class ExecutionContext;
class OpAttrsRef;
class TensorMetadata;

// Computes the metadata (e.g. shape) of the result values of an op, and emits
// any errors about invalid shapes or attributes.  A metadata function should
// have the same number of inputs and outputs as the op, but takes them as
// TensorMetadata objects instead of TensorHandle's.  The result of this
// function is null in the successful case, or the 'error' AsyncValue produced
// by emitting a diagnostic in the error case.

using OpMetadataFn = RCReference<AsyncValue> (*)(
    const ExecutionContext& exec_ctx, ArrayRef<TensorMetadata> inputs,
    const OpAttrsRef& attrs, MutableArrayRef<TensorMetadata> results);
}  // namespace tfrt

#endif  // TFRT_CORE_RUNTIME_OP_METADATA_FUNCTION_H_
