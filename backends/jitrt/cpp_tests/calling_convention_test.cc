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

#include "gtest/gtest.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "tfrt/jitrt/jitrt.h"
#include "tfrt/jitrt/jitrt_compiler.h"

namespace tfrt {
namespace jitrt {
namespace {

static const char* mlir_module = R"(
    #map = affine_map<(d0) -> (d0)>
    func.func @log2_1d(%arg0: memref<?xf32>) -> memref<?xf32> {
      %c0 = arith.constant 0 : index
     %0 = memref.dim %arg0, %c0 : memref<?xf32>
      %1 = memref.alloc(%0) : memref<?xf32>
      linalg.generic { indexing_maps = [#map, #map],
                       iterator_types = ["parallel"] }
                     ins(%arg0 : memref<?xf32>) outs(%1 : memref<?xf32>) {
      ^bb0(%arg1: f32, %arg2: f32):
        %2 = math.log2 %arg1 : f32
        linalg.yield %2 : f32
      }
      return %1 : memref<?xf32>
    })";

static const char* entrypoint = "log2_1d";

struct CallingConventionTestCase {
  std::string test_name;
  CallingConvention calling_convention;
  int expected_num_results;
  int expected_num_operands;
};

using CallingConventionTest =
    ::testing::TestWithParam<CallingConventionTestCase>;

// Test that the compiled runtime signature matches what is specified in the
// calling convention.
TEST_P(CallingConventionTest, TestSignature) {
  const CallingConventionTestCase& test_case = GetParam();

  JitExecutable::Options opts;
  opts.specialization = JitExecutable::Specialization::kEnabled;
  opts.compiler.calling_convention = test_case.calling_convention;
  opts.compiler.register_dialects = [](mlir::DialectRegistry& registry) {
    RegisterDefaultJitRtDialects(registry);
  };
  opts.compiler.create_compilation_pipeline = [&](mlir::PassManager& pm) {
    CreateDefaultJitRtCompilationPipeline(pm, {});
  };

  llvm::Expected<JitExecutable> jit_executable =
      JitExecutable::Instantiate(mlir_module, entrypoint, opts);
  if (auto err = jit_executable.takeError()) ASSERT_FALSE(err) << StrCat(err);

  // The default executable is enough for us to inspect the runtime signature.
  llvm::Expected<AsyncValuePtr<Executable>> executable =
      jit_executable->DefaultExecutable();
  // Await the successful compilation completion.
  if (auto err = executable.takeError()) ASSERT_FALSE(err) << StrCat(err);
  Await(executable->value());

  auto is_dynamic_memref = [](const Type* type) {
    // Get the underlying value type from the async value.
    while (auto* value = dyn_cast<AsyncValueType>(type))
      type = &value->value_type();

    auto* memref = llvm::dyn_cast<MemrefType>(type);
    if (!memref) return false;
    return llvm::any_of(memref->sizes(), mlir::ShapedType::isDynamic);
  };

  const FunctionType& signature = (*executable)->runtime_signature();
  EXPECT_EQ(signature.num_results(), test_case.expected_num_results);
  EXPECT_EQ(signature.num_operands(), test_case.expected_num_operands);

  // All results and operands, except the kernel context, should be dynamic
  // memrefs in the default executable.
  if (signature.num_results() > 0) {
    ASSERT_EQ(signature.num_results(), 1);
    const Type* type = signature.result(0);

    EXPECT_TRUE(is_dynamic_memref(type));
  }
  for (unsigned i = 0; i < signature.num_operands(); ++i) {
    const Type* type = signature.operand(i);

    if (i == 0)
      EXPECT_TRUE(llvm::isa<KernelContextOperandType>(type));
    else
      EXPECT_TRUE(is_dynamic_memref(type));
  }
}

INSTANTIATE_TEST_SUITE_P(
    CallingConventionTest, CallingConventionTest,
    testing::ValuesIn<CallingConventionTestCase>({
        {"DefaultCallingConvention",
         xla::runtime::DefaultCallingConvention(
             mlir::bufferization::BufferizeTypeConverter()),
         /*expected_num_results=*/1, /*expected_num_operands=*/2},
        {"ResultsToOutsCallingConvention",
         xla::runtime::ResultsToOutsCallingConvention(
             mlir::bufferization::BufferizeTypeConverter()),
         /*expected_num_results=*/0, /*expected_num_operands=*/3},
    }),
    [](const testing::TestParamInfo<CallingConventionTest::ParamType>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace jitrt
}  // namespace tfrt
