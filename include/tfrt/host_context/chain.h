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

//===- chain.h - Control dependence representation --------------*- C++ -*-===//
//
// Chain is a control dependence between kernels. Its runtime representation is
// a zero sized value.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_HOST_CONTEXT_CHAIN_H_
#define TFRT_HOST_CONTEXT_CHAIN_H_

namespace tfrt {

struct Chain {};

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_CHAIN_H_
