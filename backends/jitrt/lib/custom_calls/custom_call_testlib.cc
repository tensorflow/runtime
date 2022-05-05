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

#include <string>
#include <utility>

#include "mlir/Support/LogicalResult.h"
#include "tfrt/jitrt/custom_call.h"
#include "tfrt/support/string_util.h"

namespace tfrt {
namespace jitrt {

using mlir::failure;
using mlir::LogicalResult;
using mlir::success;

using llvm::StringRef;

static LogicalResult Multiply(MemrefDesc input, MemrefDesc output, float cst) {
  // TODO(ezhulenev): Support all floating point dtypes.
  if (input.dtype() != output.dtype() || input.sizes() != output.sizes() ||
      input.dtype() != DType::F32)
    return failure();

  int64_t num_elements = 1;
  for (int64_t d : input.sizes()) num_elements *= d;

  float* input_data = reinterpret_cast<float*>(input.data());
  float* output_data = reinterpret_cast<float*>(output.data());

  for (int64_t i = 0; i < num_elements; ++i)
    output_data[i] = input_data[i] * cst;

  return success();
}

// A custom call for testing attributes encoding/decoding.
static LogicalResult PrintAttrs(const char* caller, int32_t i32, int64_t i64,
                                float f32, double f64,
                                ArrayRef<int32_t> i32_arr,
                                ArrayRef<int64_t> i64_arr,
                                ArrayRef<float> f32_arr,
                                ArrayRef<double> f64_arr, StringRef str) {
  llvm::outs() << caller << "\n";

  llvm::outs() << "i32: " << i32 << "\n";
  llvm::outs() << "i64: " << i64 << "\n";
  llvm::outs() << "f32: " << f32 << "\n";
  llvm::outs() << "f64: " << f64 << "\n";

  auto print_arr = [](llvm::StringRef type, auto arr) {
    llvm::outs() << type << "[" << arr.size() << "] " << Join(arr, ", ")
                 << "\n";
  };

  print_arr("i32", i32_arr);
  print_arr("i64", i64_arr);
  print_arr("f32", f32_arr);
  print_arr("f64", f64_arr);

  llvm::outs() << "str: " << str << "\n";

  return success();
}

void RegisterCustomCallTestLib(CustomCallRegistry* registry) {
  registry->Register(CustomCall::Bind("testlib.multiply")
                         .Arg<MemrefDesc>()  // input
                         .Arg<MemrefDesc>()  // output
                         .Attr<float>("cst")
                         .To(Multiply));

  registry->Register(CustomCall::Bind("testlib.print_attrs")
                         .UserData<const char*>()
                         .Attr<int32_t>("i32")
                         .Attr<int64_t>("i64")
                         .Attr<float>("f32")
                         .Attr<double>("f64")
                         .Attr<ArrayRef<int32_t>>("i32_arr")
                         .Attr<ArrayRef<int64_t>>("i64_arr")
                         .Attr<ArrayRef<float>>("f32_arr")
                         .Attr<ArrayRef<double>>("f64_arr")
                         .Attr<StringRef>("str")
                         .To(PrintAttrs));
}

}  // namespace jitrt
}  // namespace tfrt
