/*
 * Copyright 2022 The TensorFlow Runtime Authors
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

#ifndef TFRT_BACKENDS_JITRT_CUSTOM_CALLS_CUSTOM_CALLS_TESTLIB_H_
#define TFRT_BACKENDS_JITRT_CUSTOM_CALLS_CUSTOM_CALLS_TESTLIB_H_

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"

namespace tfrt {
namespace jitrt {

llvm::orc::SymbolMap CustomCallsTestlibSymbolMap(
    llvm::orc::MangleAndInterner mangle);

}  // namespace jitrt
}  // namespace tfrt

#endif  // TFRT_BACKENDS_JITRT_CUSTOM_CALLS_CUSTOM_CALLS_TESTLIB_H_
