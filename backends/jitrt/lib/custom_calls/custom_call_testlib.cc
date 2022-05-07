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

#include "tfrt/jitrt/custom_calls/custom_call_testlib.h"

#include <string>
#include <utility>

#include "llvm_derived/Support/raw_ostream.h"
#include "mlir/Support/LogicalResult.h"
#include "tfrt/jitrt/custom_call.h"
#include "tfrt/jitrt/jitrt.h"
#include "tfrt/jitrt/runtime.h"
#include "tfrt/support/string_util.h"

namespace tfrt {
namespace jitrt {

using mlir::failure;
using mlir::LogicalResult;
using mlir::success;

using llvm::StringRef;

// NoOp custom call for benchmarking arguments/attributes encoding.
static LogicalResult NoOp(FlatMemrefView, FlatMemrefView, FlatMemrefView,
                          FlatMemrefView, StringRef, float, double) {
  return success();
}

static LogicalResult Multiply(MemrefView input, MemrefView output, float cst) {
  // TODO(ezhulenev): Support all floating point dtypes.
  if (input.dtype != output.dtype || input.sizes != output.sizes ||
      input.dtype != DType::F32)
    return failure();

  int64_t num_elements = 1;
  for (int64_t d : input.sizes) num_elements *= d;

  float* input_data = reinterpret_cast<float*>(input.data);
  float* output_data = reinterpret_cast<float*>(output.data);

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
  tfrt::outs() << caller << "\n";

  tfrt::outs() << "i32: " << i32 << "\n";
  tfrt::outs() << "i64: " << i64 << "\n";
  tfrt::outs() << "f32: " << f32 << "\n";
  tfrt::outs() << "f64: " << f64 << "\n";

  auto print_arr = [](llvm::StringRef type, auto arr) {
    tfrt::outs() << type << "[" << arr.size() << "] " << Join(arr, ", ")
                 << "\n";
  };

  print_arr("i32", i32_arr);
  print_arr("i64", i64_arr);
  print_arr("f32", f32_arr);
  print_arr("f64", f64_arr);

  tfrt::outs() << "str: " << str << "\n";
  tfrt::outs().flush();

  return success();
}

static LogicalResult PrintVariadicArgs(CustomCall::RemainingArgs args) {
  tfrt::outs() << "Number of variadic arguments: " << args.size() << "\n";

  for (unsigned i = 0; i < args.size(); ++i) {
    tfrt::outs() << "arg[" << i << "]: ";

    if (args.isa<int32_t>(i)) {
      tfrt::outs() << "i32: " << args.get<int32_t>(i);
    } else if (args.isa<int64_t>(i)) {
      tfrt::outs() << "i64: " << args.get<int64_t>(i);
    } else if (args.isa<float>(i)) {
      tfrt::outs() << "f32: " << args.get<float>(i);
    } else if (args.isa<double>(i)) {
      tfrt::outs() << "f64: " << args.get<double>(i);
    } else if (args.isa<MemrefView>(i)) {
      tfrt::outs() << args.get<MemrefView>(i) << " / "
                   << args.get<FlatMemrefView>(i);
    } else {
      tfrt::outs() << "<unknown type>";
    }

    tfrt::outs() << "\n";
    tfrt::outs().flush();
  }
  return success();
}

static LogicalResult PrintMemrefAndVariadicArgs(
    MemrefView arg, CustomCall::RemainingArgs args) {
  tfrt::outs() << "arg: " << arg << "\n";
  return PrintVariadicArgs(args);
}

// Custom call handler for testing direct custom call compilation.
static bool DirectCustomCall(runtime::KernelContext* ctx, void** args,
                             void** attrs) {
  internal::DecodedArgs decoded_args(args);
  internal::DecodedAttrs decoded_attrs(attrs);
  CustomCall::UserData* user_data = Executable::GetUserData(ctx);
  auto caller = user_data->getIfExists<const char*>();
  tfrt::outs() << "Direct custom call: num_args=" << decoded_args.size()
               << "; num_attrs=" << decoded_attrs.size()
               << "; str=" << (caller ? *caller : "<unknown>") << "\n";
  return true;
}

// Direct NoOp custom call for benchmarking arguments/attributes encoding.
static bool DirectNoOp(runtime::KernelContext* ctx, void** args, void** attrs) {
  static CustomCall* call = CustomCall::Bind("testlib.noop")
                                .Arg<FlatMemrefView>()
                                .Arg<FlatMemrefView>()
                                .Arg<FlatMemrefView>()
                                .Arg<FlatMemrefView>()
                                .Attr<StringRef>("str")
                                .Attr<float>("f32")
                                .Attr<double>("f64")
                                .To<CustomCall::RuntimeChecks::kNone>(NoOp)
                                .release();

  return mlir::succeeded(call->call(args, attrs, Executable::GetUserData(ctx)));
}

// Direct PrintAttrs custom call for testing disabled attributes checks.
static bool DirectPrintAttrs(runtime::KernelContext* ctx, void** args,
                             void** attrs) {
  static CustomCall* call =
      CustomCall::Bind("testlib.print_attrs")
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
          .To<CustomCall::RuntimeChecks::kNone>(PrintAttrs)
          .release();

  return mlir::succeeded(call->call(args, attrs, Executable::GetUserData(ctx)));
}

void RegisterCustomCallTestLib(CustomCallRegistry* registry) {
  registry->Register(CustomCall::Bind("testlib.noop")
                         .Arg<FlatMemrefView>()
                         .Arg<FlatMemrefView>()
                         .Arg<FlatMemrefView>()
                         .Arg<FlatMemrefView>()
                         .Attr<StringRef>("str")
                         .Attr<float>("f32")
                         .Attr<double>("f64")
                         .To(NoOp));

  registry->Register(CustomCall::Bind("testlib.multiply")
                         .Arg<MemrefView>()  // input
                         .Arg<MemrefView>()  // output
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

  registry->Register(CustomCall::Bind("testlib.variadic_args")
                         .RemainingArgs()  // variadic args
                         .To(PrintVariadicArgs));

  registry->Register(CustomCall::Bind("testlib.memref_and_variadic_args")
                         .Arg<MemrefView>()
                         .RemainingArgs()  // variadic args
                         .To(PrintMemrefAndVariadicArgs));
}

llvm::orc::SymbolMap CustomCallsTestlibSymbolMap(
    llvm::orc::MangleAndInterner mangle) {
  llvm::orc::SymbolMap symbol_map;

  auto bind = [&](llvm::StringRef name, auto symbol_ptr) {
    symbol_map[mangle(name)] = llvm::JITEvaluatedSymbol(
        llvm::pointerToJITTargetAddress(symbol_ptr), llvm::JITSymbolFlags());
  };

  bind("testlib.direct_call", &DirectCustomCall);
  bind("testlib.noop", &DirectNoOp);
  bind("testlib.print_attrs", &DirectPrintAttrs);

  return symbol_map;
}

}  // namespace jitrt
}  // namespace tfrt
