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

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Transforms/Passes.h"
#include "tfrt/host_context/async_dispatch.h"
#include "tfrt/host_context/async_value.h"
#include "tfrt/jitrt/custom_calls/custom_call_testlib.h"
#include "tfrt/jitrt/jitrt_compiler.h"
#include "tfrt/jitrt/results.h"
#include "tfrt/tensor/dense_host_tensor.h"
#include "third_party/tensorflow/compiler/xla/mlir/runtime/transforms/calling_convention.h"
#include "third_party/tensorflow/compiler/xla/mlir/runtime/transforms/compiler.h"
#include "third_party/tensorflow/compiler/xla/mlir/runtime/transforms/custom_call_encoding.h"
#include "third_party/tensorflow/compiler/xla/runtime/arguments.h"
#include "third_party/tensorflow/compiler/xla/runtime/custom_call_registry.h"
#include "third_party/tensorflow/compiler/xla/runtime/executable.h"
#include "third_party/tensorflow/compiler/xla/runtime/jit_executable.h"

namespace tfrt {
namespace jitrt {

using namespace xla::runtime;  // NOLINT

using absl::StatusOr;
using llvm::SmallVector;
using mlir::failure;
using mlir::FailureOr;
using mlir::LogicalResult;
using mlir::success;
using mlir::TypeID;
using xla::PrimitiveType;

namespace LLVM = mlir::LLVM;

// Features supported in JitRt but missing in this example:
//   1. Launching async tasks.
//   2. Returning async results from the compiled function.

// TODO(ezhulenev): Show all the features supported by JitRt?

//===----------------------------------------------------------------------===//
// Compiled program written in a mix of MLIR dialects.
//===----------------------------------------------------------------------===//

// JitRt input program can be defined in arbitrary dialects, the only
// requirement is that the user must pass a pipeline that can lower the input
// program to the LLVM dialect (see `create_compilation_pipeline` option below).
//
// In this example we use Tosa to define the compute function body becase it's
// available upstream, and transpose operation can showcase the input value
// specialization: Tosa can lower to Linalg (and then to LLVM) only transpose
// operations with constant permutation, without input value specialization this
// program can't be lowered to LLVM and executed.
//
// We also use `testlib` dialect for showing how to register custom type for
// passing in as a compiled function argument, and passing it back to the
// custom call handler, which requires to specify its lowering to LLVM.
static const char* mlir_module = R"(
  module {
    // Declare your own "runtime" intrinsics library in the compiled module.
    func.func private @my.runtime.intrinsic(%arg: !testlib.custom_arg)
      attributes { rt.dynamic, rt.custom_call = "my.runtime.intrinsic" }

    // Permutation argument annotated with a constraint, which means that
    // before compiling the function body, argument must be sunk into the
    // function body as a constant. Otherwise tosa.transpose will not be lowered
    // to Linalg operation.
    func.func @compute(
      %arg: !testlib.custom_arg,
      %input: tensor<?x?xf32>,
      %perm: tensor<2xi32> { rt.constraint = "value" }
    ) -> tensor<?x?xf32> {

      // Pass custom argument and attributes to the runtime intrinsics.
      func.call @my.runtime.intrinsic(%arg) { api_version = 1 : i32 }
        : (!testlib.custom_arg) -> ()

      // Transpose input tensor and return result to the caller.
      %transposed = "tosa.transpose"(%input, %perm)
        : (tensor<?x?xf32>, tensor<2xi32>)  -> (tensor<?x?xf32>)

      func.return %transposed : tensor<?x?xf32>
    }
  })";

static const char* entrypoint = "compute";

//===----------------------------------------------------------------------===//
// Declare run-time type/argument for user-defined type.
//===----------------------------------------------------------------------===//

// Convert custom argument type to the LLVM type that will be used during module
// compilation. For simplicity we will pass custom arguments as an opaque LLVM
// pointer (`!llvm.ptr`).
static mlir::Type ConvertCustomArg(CustomArgType type) {
  return LLVM::LLVMPointerType::get(type.getContext());
}

// Run time type corresponding to the `!testlib.custom_arg` type. Run time type
// definition decouples Executable from the MLIR dependency, and also defines
// the `ArgumentAbi` for passing values of this type to the compiled executable.
struct CustomArgRtType : public llvm::RTTIExtends<CustomArgRtType, Type> {
  static constexpr char ID = 0;  // NOLINT

  // We pass custom argument as an opaque pointer (`void*` or `!llvm.ptr`).
  StatusOr<ArgumentAbi> AsArgument() const final { return ArgumentAbi{1}; }

  std::string ToString() const final { return "!testlib.custom_arg"; }
};

// Run time argument corresponding to the `!testlib.custom_arg` type. In this
// particular example the `!testlib.custom_arg` at run time is a `std::string`
// that we want to pass back to our custom call. However we decided that we want
// to hide it behind the opaque pointer, so the packing function adds a pointer
// to the pointer to a string to the arguments array (as a `void*` C++ pointer).
struct CustomArgument
    : public llvm::RTTIExtends<CustomArgument, xla::runtime::Argument> {
  static constexpr char ID = 0;  // NOLINT

  explicit CustomArgument(std::string message) : message(std::move(message)) {}

  // Check that argument matches the expected type.
  absl::Status Verify(const Type& type) const final {
    if (isa<CustomArgRtType>(type)) return absl::OkStatus();
    return absl::InvalidArgumentError(
        absl::StrCat("expected custom arg type, got: ", type.ToString()));
  }

  // Packs an indirect pointer to the string message to the arguments array.
  void Pack(absl::Span<void*> args) const final {
    args[0] = const_cast<void*>(reinterpret_cast<const void*>(&ptr));
  }

  std::string ToString() const final { return "custom_arg: " + message + "\n"; }

  std::string message;

  // This is the pointer that we'll pass to the compiled module as a "custom
  // argument representation" (opaque pointer). The semantics of compiled
  // function arguments is "pointers to arguments", and because the argument is
  // not a string, but the pointer to it, we should not be packing the pointer
  // to `message`, but a pointer to the pointer to message.
  std::string* ptr = &message;
};

//===----------------------------------------------------------------------===//
// Define encoding of custom arguments to custom call arguments.
//===----------------------------------------------------------------------===//

// Forward declare.
struct CustomArg;

// Packs value on the stack. Returns allocation holding the value.
static LLVM::AllocaOp PackValue(mlir::ImplicitLocOpBuilder& b,
                                mlir::Value value) {
  mlir::Type ptr = mlir::LLVM::LLVMPointerType::get(b.getContext());

  auto one = b.create<mlir::arith::ConstantOp>(b.getI32IntegerAttr(1));
  auto mem = b.create<mlir::LLVM::AllocaOp>(ptr, value.getType(), one, 0);

  b.create<mlir::LLVM::StoreOp>(value, mem);

  return mem;
}

// Custom argument encoding passed to the `rt-to-llvm` pipeline and responsible
// for encoding custom argument value for passing to the custom call. Because we
// chose an opaque pointer implementation, we just pass it directly to the call.
//
// TODO(ezhulenev): This opaque pointer encoding with a user TypeID can be added
// to the `custom_call_to_llvm` library if required in some other place.
class CustomArgEncoding : public CustomCallArgEncoding {
 public:
  LogicalResult Match(mlir::Value value, mlir::Value) const final {
    return success(value.getType().isa<CustomArgType>());
  }

  FailureOr<Encoded> Encode(Globals& g, Allocas& a,
                            mlir::ImplicitLocOpBuilder& b, mlir::Value,
                            mlir::Value converted) const final {
    Encoded encoded;
    encoded.type_id = EncodeTypeId(g, b, TypeID::get<Tagged<CustomArg>>());
    encoded.value = PackValue(b, converted);
    return encoded;
  }
};

//===----------------------------------------------------------------------===//
// Register custom call with a runtime.
//===----------------------------------------------------------------------===//

// Context structure that encapsulats all the state that has to be available
// to your runtime intrinsics.
struct MyRuntimeContext {
  std::vector<std::string> custom_args;
};

// Custom argument passed to the compiled function as it seen by the custom call
// handler. We encode custom argument as a single pointer to the `message`, and
// in the custom call we decode it back to the C++ type.

// Custom argument (defined above) passed  to the compiled function as a pointer
// to the `message`. When this pointer passed to the custom call, we "decode" it
// to the structure that wraps the pointer.
struct CustomArg {
  const std::string* message;
};

// Implement your runtime intrinsic as a regular C++ function.
static LogicalResult MyRuntimeIntrinsic(MyRuntimeContext* ctx,
                                        CustomArg custom_arg,
                                        int32_t api_version) {
  ctx->custom_args.push_back(*custom_arg.message);
  return success();
}

// Register your runtime support library with JitRt as custom calls.
void RegisterMyRuntimeIntrinsics(DynamicCustomCallRegistry* registry) {
  registry->Register(CustomCall::Bind("my.runtime.intrinsic")
                         .UserData<MyRuntimeContext*>()
                         .Arg<CustomArg>()
                         .Attr<int32_t>("api_version")
                         .To(MyRuntimeIntrinsic));
}

// Register type id to unique name mapping for custom types.
void RegisterMyRuntimeTypeNames(TypeIDNameRegistry& registry) {
  registry.Register<Tagged<CustomArg>>("__type_id_customarg");
}

//===----------------------------------------------------------------------===//
// The end-to-end test itself that compiles and executes the MLIR module.
//===----------------------------------------------------------------------===//

TEST(EndToEndExampleTest, CompiledAndExecute) {
  // Step by step guide for compiling and executing your programs on top of the
  // JitRt library.

  // ------------------------------------------------------------------------ //
  // 1. Set up options for the JitRt executable compilation/recompilation.
  // ------------------------------------------------------------------------ //
  JitExecutable::Options opts;

  // Because one of the arguments requires value specialization, we must enable
  // specialization to be able to compile the executable.
  opts.specialization = JitExecutable::Specialization::kEnabled;

  // Define what dialects are supported in the input IR module. If you have your
  // own custom dialects in the input IR you must pass a callback that registers
  // all the dialects that are considered legal for your input program.
  //
  // In this example in addition to "standard" JitRt dialects we add only Tosa.
  opts.compiler.register_dialects =
      [](xla::runtime::DialectRegistry& dialects) {
        // For testing value specialization.
        dialects->insert<mlir::tosa::TosaDialect>();

        // For testing passing custom arguments back to custom calls.
        dialects->insert<TestlibDialect>();

        RegisterDefaultJitRtDialects(dialects);
      };

  // Convert all tensors in the compute function signature to memrefs, because
  // tensors do not have any runtime representation and can't be passed across
  // the ABI boundary. The expectation is that compiler pipeline will act
  // according to this calling convention, and the entrypoint will have the same
  // function signature.
  opts.compiler.calling_convention = xla::runtime::DefaultCallingConvention(
      mlir::bufferization::BufferizeTypeConverter());

  // Add a mapping from the custom type id to symbol name.
  opts.compiler.symbols_binding =
      ToSymbolsBinding(/*custom_calls=*/nullptr, RegisterMyRuntimeTypeNames);

  // Add a conversion from the `!testlib.custom_arg` MLIR type to the run time
  // type corresponding to a custom argument.
  opts.compiler.type_converter.AddConversion(
      [](CustomArgType arg) { return std::make_unique<CustomArgRtType>(); });

  // ------------------------------------------------------------------------ //
  // 2. Set up compilation pipeline that lowers input module to LLVM.
  // ------------------------------------------------------------------------ //

  // As a first step we lower from Tosa to Linalg on buffers, and then we rely
  // on a default JitRt compilation pipeline to lower further to LLVM.
  opts.compiler.create_compilation_pipeline =
      [&](xla::runtime::PassManager& passes) {
        // 1. Lower Tosa to Linalg in tensors.
        passes->addNestedPass<mlir::func::FuncOp>(
            mlir::tosa::createTosaToLinalg());

        // 2. Lower Linalg on tensors to Linalg on buffers.
        passes->addPass(mlir::func::createFuncBufferizePass());
        passes->addNestedPass<mlir::func::FuncOp>(
            mlir::createLinalgBufferizePass());

        // 3. Clean up IR after lowering to Linalg on buffers.
        passes->addPass(mlir::createCSEPass());
        passes->addPass(mlir::createCanonicalizerPass());

        // 4. Continue compilation using the default JitRt pipeline.
        CompilationPipelineOptions copts;

        // Register type id to unique symbol name mappings.
        copts.populate_type_id_names = RegisterMyRuntimeTypeNames;

        // Register type conversions from custom types (!testlib.custom_arg).
        copts.populate_type_conversions = [](mlir::TypeConverter& converter) {
          converter.addConversion(ConvertCustomArg);
        };

        // Add custom call arguments encoding for custom types
        // (!testlib.custom_arg).
        copts.populate_arg_encodings = [](CustomCallArgEncodingSet& encoding) {
          encoding.Add<CustomArgEncoding>();
        };

        CreateDefaultJitRtCompilationPipeline(passes, copts);
      };

  // If your input IR requires specialization, you'll also need to define the
  // `opts.compiler.create_compilation_pipeline` callback. In this test we rely
  // on the fact that "value-specialized" arguments will be materialized as
  // constants in the function body.

  // ------------------------------------------------------------------------ //
  // 3. Instantiate JitExecutable from the input MLIR source.
  // ------------------------------------------------------------------------ //

  // JitExecutable does compilation/recompilation from the input source to the
  // Executable artifact.
  absl::StatusOr<JitExecutable> jit_executable =
      JitExecutable::Instantiate(mlir_module, entrypoint, opts);
  ASSERT_TRUE(jit_executable.ok()) << jit_executable.status().message();

  // In this example default executable will be in error state, because the
  // program requires value specialization and can't be compiled without it.
  AsyncValuePtr<Executable> default_exec = jit_executable->DefaultExecutable();
  ASSERT_TRUE(default_exec.IsError());

  // ------------------------------------------------------------------------ //
  // 4. Prepare input data for the compiled program.
  // ------------------------------------------------------------------------ //

  // JitRt Executable knows how to pass MemrefDesc to the compiled program
  // according to the MLIR C ABI (memrefs passed as `StridedMemRefType` struct).
  //
  // For the custom argument (!testlib.custom_arg) it relies on the ABI and
  // argument packing defined by the `CustomArgument` type above.
  //
  // For "real" programs instead of vectors we should have tensors flying
  // around.

  // Allocate storage for arguments.
  std::vector<float> input = {1.0, 2.0, 3.0, 4.0};
  std::vector<int32_t> perm = {1, 0};

  // Prepare arguments for the executable.
  Arguments<CustomArgument, MemrefDesc> args(3);
  args.emplace_back<CustomArgument>("hello from the other side");

  // Input is a 2x2 memref.
  std::array<int64_t, 2> sizes = {2, 2};
  std::array<int64_t, 2> strides = {2, 1};
  args.emplace_back<MemrefDesc>(PrimitiveType::F32, input.data(), 0, sizes,
                                strides);

  // Perm is a vector of size 2.
  std::array<int64_t, 1> vec_size = {2};
  std::array<int64_t, 1> vec_stride = {1};
  args.emplace_back<MemrefDesc>(PrimitiveType::S32, perm.data(), 0, vec_size,
                                vec_stride);

  // ------------------------------------------------------------------------ //
  // 5. Prepare options for executing the JitRt executable.
  // ------------------------------------------------------------------------ //

  Executable::ExecuteOpts execute_opts;

  // We don't expect to launch any async tasks in this example.
  execute_opts.async_task_runner =
      reinterpret_cast<jitrt::AsyncTaskRunner*>(0XDEADBEEF);

  // Pass runtime context to all runtime intrinsics handlers.
  MyRuntimeContext runtime_context;

  CustomCall::UserData user_data;
  user_data.insert(&runtime_context);
  execute_opts.custom_call_data = &user_data;

  DynamicCustomCallRegistry custom_call_registry;
  RegisterMyRuntimeIntrinsics(&custom_call_registry);
  execute_opts.custom_call_registry = &custom_call_registry;

  // ------------------------------------------------------------------------ //
  // 6. Get executable specialized for the concrete operands.
  // ------------------------------------------------------------------------ //

  // At this point we trigger compilation of the original input program for
  // the concrete value of the transpose permutation vector.
  absl::StatusOr<AsyncValuePtr<Executable>> executable =
      jit_executable->GetExecutable(args);

  // Await the successful compilation completion.
  ASSERT_TRUE(executable.ok()) << executable.status().message();
  Await(executable->value());

  // ------------------------------------------------------------------------ //
  // 7. Define how to convert returned values back to C++ objects.
  // ------------------------------------------------------------------------ //

  // Conversion context allows to pass data from the caller to the result
  // conversion function (e.g. auxiliary data structures to distinguish newly
  // allocated memrefs from forwarded arguments). In this example we don't pass
  // anything to the conversion functions.
  struct ResultConversionCtx {};
  ResultConversionCtx conversion_ctx;

  // TODO(ezhulenev): We should decouple JitRt from TFRT specific
  // RemainingResults, and do not force clients to deal with returned
  // AsyncValues.

  // Placeholders for returned values.
  unsigned num_results = (*executable)->num_results();
  llvm::SmallVector<RCReference<AsyncValue>> result_values(num_results);
  RemainingResults results(result_values);

  // If execution failed errors will be automatically allocated for all results.
  RemainingResultsConverter<ResultConversionCtx> converter(results,
                                                           conversion_ctx);
  converter.AddConversion(ReturnMemrefAsDenseHostTensor<ResultConversionCtx>);

  // ------------------------------------------------------------------------ //
  // 8. Call JitRt executable with the prepared operands.
  // ------------------------------------------------------------------------ //

  // Execute Jit compiled executable.
  auto executed = (*executable)->Execute(args, converter, execute_opts);
  ASSERT_TRUE(executed.ok())
      << "Failed to execute: " << executed.status().message();

  // Check the result returned from the compiled function.
  ASSERT_TRUE(result_values[0]->IsAvailable());

  // Result must be a DenseHostTensor.
  auto& result_tensor = result_values[0]->get<tfrt::DenseHostTensor>();
  ASSERT_EQ(result_tensor.dtype(), DType::F32);
  ASSERT_EQ(result_tensor.NumElements(), 4);

  ArrayRef<float> data = {result_tensor.data<float>(), 4};
  std::vector<float> expected = {1.0, 3.0, 2.0, 4.0};
  EXPECT_EQ(data.vec(), std::vector<float>(expected.begin(), expected.end()));

  // Check that custom argument was correctly passed to the custom call.
  ASSERT_EQ(runtime_context.custom_args.size(), 1);
  EXPECT_EQ(runtime_context.custom_args[0], "hello from the other side");

  // ------------------------------------------------------------------------ //
  // 8. Saving/Restoring JitRt executable to/from object file.
  // ------------------------------------------------------------------------ //

  // See `aot_compilation_test` for an example of serializing JitRt executable
  // as an object file.
}

}  // namespace jitrt
}  // namespace tfrt

namespace xla {
namespace runtime {

using tfrt::jitrt::CustomArg;

// Register custom argument decoding (must be in ::xla::runtime namespace).
template <CustomCall::RuntimeChecks checks>
struct CustomCallArgDecoding<CustomArg, checks> {
  static FailureOr<CustomArg> Decode(TypeID type_id, void* value) {
    if (!CustomCall::Isa<CustomArg>(checks, type_id)) return failure();
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(value, sizeof(void*));
    return CustomArg{*reinterpret_cast<const std::string**>(value)};
  }
};

}  // namespace runtime
}  // namespace xla
