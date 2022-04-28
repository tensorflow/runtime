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

#include <utility>

#include "tfrt/jitrt/custom_call.h"

namespace tfrt {
namespace jitrt {

static void TimesTwo(MemrefDesc input, MemrefDesc output) {
  assert(input.dtype == DType::F32 && output.dtype == DType::F32);
  assert(input.sizes == output.sizes);

  int64_t num_elements = 1;
  for (int64_t d : input.sizes) num_elements *= d;

  float* input_data = reinterpret_cast<float*>(input.data);
  float* output_data = reinterpret_cast<float*>(output.data);

  for (int64_t i = 0; i < num_elements; ++i)
    output_data[i] = input_data[i] * 2.0;
}

void RegisterCustomCallTestLib(CustomCallRegistry* registry) {
  registry->Register(CustomCall::Bind("testlib.times_two")
                         .Arg<MemrefDesc>()  // input
                         .Arg<MemrefDesc>()  // output
                         .To(TimesTwo));
}

}  // namespace jitrt
}  // namespace tfrt
