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

#include <cstdint>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "llvm_derived/Support/raw_ostream.h"
#include "mlir/Support/LogicalResult.h"
#include "tfrt/support/error_util.h"
#include "tfrt/support/string_util.h"
#include "third_party/tensorflow/compiler/xla/mlir/runtime/transforms/custom_call_encoding.h"
#include "third_party/tensorflow/compiler/xla/runtime/custom_call.h"
#include "third_party/tensorflow/compiler/xla/runtime/custom_call_registry.h"
#include "third_party/tensorflow/compiler/xla/runtime/executable.h"

XLA_RUNTIME_DEFINE_EXPLICIT_TYPE_ID(tfrt::jitrt::RuntimeEnumType);

namespace tfrt {
namespace jitrt {

using mlir::LogicalResult;
using mlir::succeeded;
using mlir::success;

using namespace xla::runtime;  // NOLINT

void PopulateCustomCallTypeIdNames(xla::runtime::TypeIDNameRegistry& registry) {
  registry.Register<Tagged<EnumType>>("__type_id_enumtype");
  registry.Register<Tagged<RuntimeEnumType>>("__type_id_runtime_enumtype");
  registry.Register<Tagged<RuntimePairOfDims>>("__type_id_runtime_pairofdims");
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
                          FlatMemrefView, std::string_view, float, double) {
  return success();
}

static absl::Status Multiply(MemrefView input, MemrefView output, float cst) {
  // TODO(ezhulenev): Support all floating point dtypes.
  if (input.dtype != output.dtype || input.sizes != output.sizes ||
      input.dtype != xla::PrimitiveType::F32)
    return absl::InvalidArgumentError("Unsupported floating point dtype");

  int64_t num_elements = 1;
  for (int64_t d : input.sizes) num_elements *= d;

  float* input_data = reinterpret_cast<float*>(input.data);
  float* output_data = reinterpret_cast<float*>(output.data);

  for (int64_t i = 0; i < num_elements; ++i)
    output_data[i] = input_data[i] * cst;

  return absl::OkStatus();
}

template <typename Array>
static void print_arr(std::string_view type, Array arr) {
  tfrt::outs() << type << "[" << arr.size() << "] " << Join(arr, ", ") << "\n";
}

template <typename T>
static void TensorToString(absl::Span<const int64_t> shapes,
                           absl::Span<const T> data,
                           llvm::raw_string_ostream& os) {
  // Print all elements if it is a vector.
  if (shapes.size() == 1) {
    os << "(" + Join(data, ", ") + ")";
    return;
  }

  size_t stride = data.size() / shapes[0];
  absl::Span<const int64_t> inner_shape(shapes.data() + 1, shapes.size() - 1);

  os << "(";
  for (size_t i = 0; i < data.size(); i += stride) {
    auto start = data.begin() + i;
    absl::Span<const T> inner_data(start, stride);
    TensorToString(inner_shape, inner_data, os);
    if (i + stride < data.size()) os << ", ";
  }
  os << ")";
}

template <typename T>
static void PrintTensor(std::string_view type,
                        CustomCall::TensorRef<T> tensor) {
  std::string tensor_str;
  llvm::raw_string_ostream os(tensor_str);
  TensorToString<T>(tensor.shape, tensor.data, os);
  tfrt::outs() << type << "[" << Join(tensor.shape, "x") << "] " << os.str()
               << "\n";
}

// A custom call for testing attributes encoding/decoding.
static LogicalResult PrintAttrs(
    const char* caller, int32_t i32, int64_t i64, float f32, double f64,
    absl::Span<const int32_t> i32_arr, absl::Span<const int64_t> i64_arr,
    absl::Span<const float> f32_arr, absl::Span<const double> f64_arr,
    CustomCall::TensorRef<int64_t> i64_2d_arr,
    absl::Span<const int32_t> i32_array, absl::Span<const int64_t> i64_array,
    absl::Span<const float> f32_array, absl::Span<const double> f64_array,
    absl::Span<const int64_t> i64_dense_array,
    absl::Span<const int64_t> empty_array, std::string_view str) {
  tfrt::outs() << caller << "\n";

  tfrt::outs() << "i32: " << i32 << "\n";
  tfrt::outs() << "i64: " << i64 << "\n";
  tfrt::outs() << "f32: " << f32 << "\n";
  tfrt::outs() << "f64: " << f64 << "\n";

  print_arr<absl::Span<const int32_t>>("i32", i32_arr);
  print_arr<absl::Span<const int64_t>>("i64", i64_arr);
  print_arr<absl::Span<const float>>("f32", f32_arr);
  print_arr<absl::Span<const double>>("f64", f64_arr);
  PrintTensor<int64_t>("i64", i64_2d_arr);

  print_arr<absl::Span<const int32_t>>("i32", i32_array);
  print_arr<absl::Span<const int64_t>>("i64", i64_array);
  print_arr<absl::Span<const float>>("f32", f32_array);
  print_arr<absl::Span<const double>>("f64", f64_array);
  print_arr<absl::Span<const int64_t>>("i64", i64_dense_array);
  print_arr<absl::Span<const int64_t>>("i64", empty_array);

  tfrt::outs() << "str: " << str << "\n";
  tfrt::outs().flush();

  return success();
}

static LogicalResult PrintVariantAttrs(CustomCall::VariantAttr attr1,
                                       CustomCall::VariantAttr attr2,
                                       CustomCall::VariantAttr attr3,
                                       CustomCall::VariantAttr attr4,
                                       CustomCall::VariantAttr attr5) {
  std::vector<CustomCall::VariantAttr> attrs = {attr1, attr2, attr3, attr4,
                                                attr5};
  for (auto attr : attrs) {
    if (attr.isa<int32_t>()) {
      tfrt::outs() << "i32: " << attr.get<int32_t>();
    } else if (attr.isa<int64_t>()) {
      tfrt::outs() << "i64: " << attr.get<int64_t>();
    } else if (attr.isa<float>()) {
      tfrt::outs() << "f32: " << attr.get<float>();
    } else if (attr.isa<double>()) {
      tfrt::outs() << "f64: " << attr.get<double>();
    } else if (attr.isa<absl::Span<const int32_t>>()) {
      print_arr<absl::Span<const int32_t>>(
          "i32", attr.get<absl::Span<const int32_t>>().value());
    } else if (attr.isa<absl::Span<const int64_t>>()) {
      print_arr<absl::Span<const int64_t>>(
          "i64", attr.get<absl::Span<const int64_t>>().value());
    } else if (attr.isa<absl::Span<const float>>()) {
      print_arr("f32", attr.get<absl::Span<const float>>().value());
    } else if (attr.isa<absl::Span<const double>>()) {
      print_arr<absl::Span<const double>>(
          "f64", attr.get<absl::Span<const double>>().value());
    } else if (attr.isa<std::string_view>()) {
      tfrt::outs() << "str: " << attr.get<std::string_view>();
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
static bool DirectCustomCall(xla::runtime::ExecutionContext* ctx, void** args,
                             void** attrs, void** rets) {
  xla::runtime::internal::DecodedArgs decoded_args(args);
  xla::runtime::internal::DecodedAttrs decoded_attrs(attrs);
  xla::runtime::internal::DecodedRets decoded_rets(rets);
  const CustomCall::UserData* user_data = Executable::GetUserData(ctx);
  const char* caller = user_data->getIfExists<const char>();
  tfrt::outs() << "Direct custom call: num_args=" << decoded_args.size()
               << "; num_attrs=" << decoded_attrs.size()
               << "; num_rets=" << decoded_rets.size()
               << "; str=" << (caller ? caller : "<unknown>") << "\n";
  return true;
}

// Direct NoOp custom call for benchmarking arguments/attributes encoding.
static bool DirectNoOp(xla::runtime::ExecutionContext* ctx, void** args,
                       void** attrs, void** rets) {
  auto noop = [](FlatMemrefView, FlatMemrefView, FlatMemrefView, FlatMemrefView,
                 std::string_view, float, double) { return success(); };

  static auto* call = CustomCall::Bind("testlib.noop")
                          .Arg<FlatMemrefView>()
                          .Arg<FlatMemrefView>()
                          .Arg<FlatMemrefView>()
                          .Arg<FlatMemrefView>()
                          .Attr<std::string_view>("str")
                          .Attr<float>("f32")
                          .Attr<double>("f64")
                          .To<CustomCall::RuntimeChecks::kNone>(noop)
                          .release();

  return succeeded(Executable::Call(ctx, *call, args, attrs, rets));
}

// Direct PrintAttrs custom call for testing disabled attributes checks.
static bool DirectPrintAttrs(xla::runtime::ExecutionContext* ctx, void** args,
                             void** attrs, void** rets) {
  static auto* call = CustomCall::Bind("testlib.print_attrs", {false})
                          .UserData<const char*>()
                          .Attr<int32_t>("i32")
                          .Attr<int64_t>("i64")
                          .Attr<float>("f32")
                          .Attr<double>("f64")
                          .Attr<absl::Span<const int32_t>>("i32_dense")
                          .Attr<absl::Span<const int64_t>>("i64_dense")
                          .Attr<absl::Span<const float>>("f32_dense")
                          .Attr<absl::Span<const double>>("f64_dense")
                          .Attr<CustomCall::TensorRef<int64_t>>("i64_2d_dense")
                          .Attr<absl::Span<const int32_t>>("i32_array")
                          .Attr<absl::Span<const int64_t>>("i64_array")
                          .Attr<absl::Span<const float>>("f32_array")
                          .Attr<absl::Span<const double>>("f64_array")
                          .Attr<absl::Span<const int64_t>>("i64_dense_array")
                          .Attr<absl::Span<const int64_t>>("empty_array")
                          .Attr<std::string_view>("str")
                          .To<CustomCall::RuntimeChecks::kNone>(PrintAttrs)
                          .release();

  return succeeded(Executable::Call(ctx, *call, args, attrs, rets));
}

void RegisterDynamicCustomCallTestLib(DynamicCustomCallRegistry& registry) {
  registry.Register(CustomCall::Bind("testlib.noop")
                        .Arg<FlatMemrefView>()
                        .Arg<FlatMemrefView>()
                        .Arg<FlatMemrefView>()
                        .Arg<FlatMemrefView>()
                        .Attr<std::string_view>("str")
                        .Attr<float>("f32")
                        .Attr<double>("f64")
                        .To(NoOp));

  registry.Register(CustomCall::Bind("testlib.multiply")
                        .Arg<MemrefView>()  // input
                        .Arg<MemrefView>()  // output
                        .Attr<float>("cst")
                        .To(Multiply));

  registry.Register(CustomCall::Bind("testlib.multiply.x3")
                        .Arg<MemrefView>()  // input
                        .Arg<MemrefView>()  // output
                        .Value(3.0)         // cst
                        .To(Multiply));

  registry.Register(CustomCall::Bind("testlib.print_attrs", {false})
                        .UserData<const char*>()
                        .Attr<int32_t>("i32")
                        .Attr<int64_t>("i64")
                        .Attr<float>("f32")
                        .Attr<double>("f64")
                        .Attr<absl::Span<const int32_t>>("i32_dense")
                        .Attr<absl::Span<const int64_t>>("i64_dense")
                        .Attr<absl::Span<const float>>("f32_dense")
                        .Attr<absl::Span<const double>>("f64_dense")
                        .Attr<CustomCall::TensorRef<int64_t>>("i64_2d_dense")
                        .Attr<absl::Span<const int32_t>>("i32_array")
                        .Attr<absl::Span<const int64_t>>("i64_array")
                        .Attr<absl::Span<const float>>("f32_array")
                        .Attr<absl::Span<const double>>("f64_array")
                        .Attr<absl::Span<const int64_t>>("i64_dense_array")
                        .Attr<absl::Span<const int64_t>>("empty_array")
                        .Attr<std::string_view>("str")
                        .To(PrintAttrs));

  registry.Register(CustomCall::Bind("testlib.print_dialect_attrs")
                        .Attr<EnumType>("enum")
                        .Attr<RuntimeEnumType>("runtime_enum")
                        .Attr<RuntimePairOfDims>("dims")
                        .To(PrintDialectAttrs));

  registry.Register(CustomCall::Bind("testlib.variadic_args")
                        .RemainingArgs()  // variadic args
                        .To(PrintVariadicArgs));

  registry.Register(CustomCall::Bind("testlib.memref_and_variadic_args")
                        .Arg<MemrefView>()
                        .RemainingArgs()  // variadic args
                        .To(PrintMemrefAndVariadicArgs));

  registry.Register(CustomCall::Bind("testlib.variant_arg")
                        .Arg<CustomCall::VariantArg>()
                        .Arg<CustomCall::VariantArg>()
                        .Arg<CustomCall::VariantArg>()
                        .To(PrintVariantArg));

  registry.Register(CustomCall::Bind("testlib.print_variant_attrs")
                        .Attr<CustomCall::VariantAttr>("i32")
                        .Attr<CustomCall::VariantAttr>("f32")
                        .Attr<CustomCall::VariantAttr>("i32_array")
                        .Attr<CustomCall::VariantAttr>("i64_array")
                        .Attr<CustomCall::VariantAttr>("str")
                        .To(PrintVariantAttrs));
}

void RegisterDirectCustomCallTestLib(DirectCustomCallRegistry& registry) {
  registry.Register("testlib.direct_call", &DirectCustomCall);
  registry.Register("testlib.print_attrs", &DirectPrintAttrs);
  registry.Register("testlib.noop", &DirectNoOp);
}

}  // namespace jitrt
}  // namespace tfrt
