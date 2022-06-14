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

#include "llvm/ADT/TypeSwitch.h"
#include "llvm_derived/Support/raw_ostream.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "tfrt/jitrt/conversion/custom_call_to_llvm.h"
#include "tfrt/jitrt/custom_call.h"
#include "tfrt/jitrt/custom_call_registry.h"
#include "tfrt/jitrt/jitrt.h"
#include "tfrt/jitrt/runtime.h"
#include "tfrt/support/string_util.h"

// clang-format off
#include "tfrt/jitrt/custom_calls/custom_call_testlib_enums.cc.inc"
// clang-format on

#define GET_ATTRDEF_CLASSES
#include "tfrt/jitrt/custom_calls/custom_call_testlib_attrs.cc.inc"

namespace tfrt {
namespace jitrt {

using mlir::Attribute;
using mlir::DialectAsmParser;
using mlir::DialectAsmPrinter;
using mlir::failure;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::success;
using mlir::TypeID;

using llvm::StringRef;

TestlibDialect::TestlibDialect(MLIRContext* context)
    : Dialect(getDialectNamespace(), context, TypeID::get<TestlibDialect>()) {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "tfrt/jitrt/custom_calls/custom_call_testlib_attrs.cc.inc"
      >();
}

// Entry point for Attribute parsing, TableGen generated code will handle the
// dispatch to the individual classes.
Attribute TestlibDialect::parseAttribute(DialectAsmParser& parser,
                                         mlir::Type type) const {
  StringRef attr_tag;
  if (failed(parser.parseKeyword(&attr_tag))) return Attribute();
  {
    Attribute attr;
    auto parse_result = generatedAttributeParser(parser, attr_tag, type, attr);
    if (parse_result.hasValue()) return attr;
  }
  parser.emitError(parser.getNameLoc(), "unknown testlib attribute");
  return Attribute();
}

// Entry point for Attribute printing, TableGen generated code will handle the
// dispatch to the individual classes.
void TestlibDialect::printAttribute(Attribute attr,
                                    DialectAsmPrinter& os) const {
  LogicalResult result = generatedAttributePrinter(attr, os);
  (void)result;
  assert(succeeded(result));
}

// Explicitly register attributes encoding for enums passed to the custom calls.
void PopulateCustomCallAttrEncoding(CustomCallAttrEncodingSet& encoding) {
  encoding.Add<EnumAttrEncoding<EnumTypeAttr, EnumType>>();
  encoding.Add<EnumAttrEncoding<EnumType2Attr, EnumType2, RuntimeEnumType>>(
      [](EnumType2 enum_value) {
        switch (enum_value) {
          case EnumType2::Foo:
            return RuntimeEnumType::kFoo;
          case EnumType2::Bar:
            return RuntimeEnumType::kBar;
          case EnumType2::Baz:
            return RuntimeEnumType::kBaz;
        }
      });

  // Encode `PairOfDimsAttr` as an aggregate attribute.
  encoding.Add<AggregateAttrEncoding<PairOfDimsAttr, RuntimePairOfDims>>(
      encoding, AggregateAttrDef<PairOfDimsAttr>()
                    .Add("rank", &PairOfDimsAttr::getRank)
                    .Add("a", &PairOfDimsAttr::getA)
                    .Add("b", &PairOfDimsAttr::getB));
}

static std::string StringifyEnumType(RuntimeEnumType value) {
  switch (value) {
    case RuntimeEnumType::kFoo:
      return "RuntimeFoo";
    case RuntimeEnumType::kBar:
      return "RuntimeBar";
    case RuntimeEnumType::kBaz:
      return "RuntimeBaz";
  }
}

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

template <typename Array>
static void print_arr(llvm::StringRef type, Array arr) {
  tfrt::outs() << type << "[" << arr.size() << "] " << Join(arr, ", ") << "\n";
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

  print_arr<ArrayRef<int32_t>>("i32", i32_arr);
  print_arr<ArrayRef<int64_t>>("i64", i64_arr);
  print_arr<ArrayRef<float>>("f32", f32_arr);
  print_arr<ArrayRef<double>>("f64", f64_arr);

  tfrt::outs() << "str: " << str << "\n";
  tfrt::outs().flush();

  return success();
}

static LogicalResult PrintVariantAttrs(CustomCall::VariantAttr attr1,
                                       CustomCall::VariantAttr attr2,
                                       CustomCall::VariantAttr attr3) {
  std::vector<CustomCall::VariantAttr> attrs = {attr1, attr2, attr3};
  for (auto attr : attrs) {
    if (attr.isa<int32_t>()) {
      tfrt::outs() << "i32: " << attr.get<int32_t>();
    } else if (attr.isa<int64_t>()) {
      tfrt::outs() << "i64: " << attr.get<int64_t>();
    } else if (attr.isa<float>()) {
      tfrt::outs() << "f32: " << attr.get<float>();
    } else if (attr.isa<double>()) {
      tfrt::outs() << "f64: " << attr.get<double>();
    } else if (attr.isa<ArrayRef<int32_t>>()) {
      print_arr<ArrayRef<int32_t>>("i32",
                                   attr.get<ArrayRef<int32_t>>().getValue());
    } else if (attr.isa<ArrayRef<int32_t>>()) {
      print_arr<ArrayRef<int64_t>>("i64",
                                   attr.get<ArrayRef<int64_t>>().getValue());
    } else if (attr.isa<ArrayRef<int32_t>>()) {
      print_arr("f32", attr.get<ArrayRef<float>>().getValue());
    } else if (attr.isa<ArrayRef<int32_t>>()) {
      print_arr<ArrayRef<double>>("f64",
                                  attr.get<ArrayRef<double>>().getValue());
    } else if (attr.isa<StringRef>()) {
      tfrt::outs() << "str: " << attr.get<StringRef>();
    } else {
      tfrt::outs() << "<unknown type>";
    }

    tfrt::outs() << "\n";
    tfrt::outs().flush();
  }

  return success();
}

static LogicalResult PrintDialectAttrs(EnumType enum_value,
                                       RuntimeEnumType runtime_enum,
                                       RuntimePairOfDims runtime_pair_of_dims) {
  tfrt::outs() << "Enum: " << stringifyEnumType(enum_value) << "\n";
  tfrt::outs() << "Runtime Enum: " << StringifyEnumType(runtime_enum) << "\n";
  tfrt::outs() << "PairOfDims: rank = " << runtime_pair_of_dims.rank;
  tfrt::outs() << " a = [" << Join(runtime_pair_of_dims.a, ", ") << "]";
  tfrt::outs() << " b = [" << Join(runtime_pair_of_dims.b, ", ") << "]\n";
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
    } else if (args.isa<StridedMemrefView>(i) || args.isa<MemrefView>(i)) {
      tfrt::outs() << args.get<StridedMemrefView>(i) << " / "
                   << args.get<MemrefView>(i) << " / "
                   << args.get<FlatMemrefView>(i);
    } else {
      tfrt::outs() << "<unknown type>";
    }

    tfrt::outs() << "\n";
    tfrt::outs().flush();
  }
  return success();
}

static LogicalResult PrintVariantArg(CustomCall::VariantArg arg1,
                                     CustomCall::VariantArg arg2,
                                     CustomCall::VariantArg arg3) {
  std::vector<CustomCall::VariantArg> args = {arg1, arg2, arg3};
  for (auto arg : args) {
    if (arg.isa<int32_t>()) {
      tfrt::outs() << "i32: " << arg.get<int32_t>();
    } else if (arg.isa<int64_t>()) {
      tfrt::outs() << "i64: " << arg.get<int64_t>();
    } else if (arg.isa<float>()) {
      tfrt::outs() << "f32: " << arg.get<float>();
    } else if (arg.isa<double>()) {
      tfrt::outs() << "f64: " << arg.get<double>();
    } else if (arg.isa<StridedMemrefView>() || arg.isa<MemrefView>()) {
      tfrt::outs() << arg.get<StridedMemrefView>() << " / "
                   << arg.get<MemrefView>() << " / "
                   << arg.get<FlatMemrefView>();
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
  const char* caller = user_data->getIfExists<const char>();
  tfrt::outs() << "Direct custom call: num_args=" << decoded_args.size()
               << "; num_attrs=" << decoded_attrs.size()
               << "; str=" << (caller ? caller : "<unknown>") << "\n";
  return true;
}

// Direct NoOp custom call for benchmarking arguments/attributes encoding.
static bool DirectNoOp(runtime::KernelContext* ctx, void** args, void** attrs) {
  auto noop = [](FlatMemrefView, FlatMemrefView, FlatMemrefView, FlatMemrefView,
                 StringRef, float, double) { return success(); };

  static auto* call = CustomCall::Bind("testlib.noop")
                          .Arg<FlatMemrefView>()
                          .Arg<FlatMemrefView>()
                          .Arg<FlatMemrefView>()
                          .Arg<FlatMemrefView>()
                          .Attr<StringRef>("str")
                          .Attr<float>("f32")
                          .Attr<double>("f64")
                          .To<CustomCall::RuntimeChecks::kNone>(noop)
                          .release();

  return mlir::succeeded(call->call(args, attrs, Executable::GetUserData(ctx)));
}

// Direct PrintAttrs custom call for testing disabled attributes checks.
static bool DirectPrintAttrs(runtime::KernelContext* ctx, void** args,
                             void** attrs) {
  static auto* call = CustomCall::Bind("testlib.print_attrs")
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

  registry->Register(CustomCall::Bind("testlib.multiply.x3")
                         .Arg<MemrefView>()  // input
                         .Arg<MemrefView>()  // output
                         .Value(3.0)         // cst
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

  registry->Register(CustomCall::Bind("testlib.print_dialect_attrs")
                         .Attr<EnumType>("enum")
                         .Attr<RuntimeEnumType>("runtime_enum")
                         .Attr<RuntimePairOfDims>("dims")
                         .To(PrintDialectAttrs));

  registry->Register(CustomCall::Bind("testlib.variadic_args")
                         .RemainingArgs()  // variadic args
                         .To(PrintVariadicArgs));

  registry->Register(CustomCall::Bind("testlib.memref_and_variadic_args")
                         .Arg<MemrefView>()
                         .RemainingArgs()  // variadic args
                         .To(PrintMemrefAndVariadicArgs));

  registry->Register(CustomCall::Bind("testlib.variant_arg")
                         .Arg<CustomCall::VariantArg>()
                         .Arg<CustomCall::VariantArg>()
                         .Arg<CustomCall::VariantArg>()
                         .To(PrintVariantArg));

  registry->Register(CustomCall::Bind("testlib.print_variant_attrs")
                         .Attr<CustomCall::VariantAttr>("i32")
                         .Attr<CustomCall::VariantAttr>("f32")
                         .Attr<CustomCall::VariantAttr>("str")
                         .To(PrintVariantAttrs));
}

DirectCustomCallLibrary CustomCallTestlib() {
  DirectCustomCallLibrary lib;
  lib.Insert("testlib.direct_call", &DirectCustomCall);
  lib.Insert("testlib.print_attrs", &DirectPrintAttrs);
  lib.Insert("testlib.noop", &DirectNoOp);
  return lib;
}

}  // namespace jitrt
}  // namespace tfrt
