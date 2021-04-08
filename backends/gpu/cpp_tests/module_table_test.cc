// Copyright 2020 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Unit test for GPU ModuleTable.
#include "tfrt/gpu/module_table.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tfrt/cpp_tests/error_util.h"
#include "tfrt/gpu/stream/stream_wrapper.h"

namespace tfrt {
namespace gpu {

namespace {

class ModuleTableTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_TRUE(IsSuccess(Init(stream::Platform::CUDA)));
    TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(stream::Platform::CUDA));
    ASSERT_GT(count, 0);
    TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(stream::Platform::CUDA, 0));
    TFRT_ASSERT_AND_ASSIGN(context_, DevicePrimaryCtxRetain(device));
  }

  stream::OwningContext context_;
};

TEST_F(ModuleTableTest, SingleModuleSingleKernel) {
  constexpr string_view kernel_ptx = R"(
        .version 6.0
        .target sm_35
        .address_size 64

        .visible .entry Kernel() {
          ret;
        })";
  const ModuleTable::Spec spec{{{kernel_ptx, {"Kernel"}}}};

  TFRT_ASSERT_AND_ASSIGN(auto current, CtxSetCurrent(context_.get()));
  TFRT_ASSERT_AND_ASSIGN(auto module_table,
                         gpu::ModuleTable::Create(current, spec));
  auto function = module_table->GetFunction(ModuleFuncHandle{0});
  auto stream = stream::Stream(nullptr, stream::Platform::CUDA);
  EXPECT_TRUE(
      IsSuccess(LaunchKernel(current, function, /*grid_dim=*/{{1, 1, 1}},
                             /*block_dim=*/{{1, 1, 1}},
                             /*shared_memory_size_bytes=*/0, stream)));
  EXPECT_TRUE(IsSuccess(stream::CtxSynchronize(current)));
}

// PTX for module containing trivial saxpy and vector addition kernels.
// Generated from the following:
// __global__ void saxpy(int n, float a, float* x, float* y) {
//   int i = blockIdx.x  * blockDim.x + threadIdx.x;
//   if (i < n) y[i] = a*x[i] + y[i];
// }
//
// __global__ void vector_add(int n, const float* x, const float* y, float* z)
// {
//   int i = blockIdx.x  * blockDim.x + threadIdx.x;
//   if (i < n) z[i] = x[i] + y[i];
// }
constexpr string_view kMultiKernelPtx = R"(
    .version 6.4
    .target sm_30
    .address_size 64

      // .globl	saxpy

    .visible .entry saxpy(
      .param .u32 saxpy_param_0,
      .param .f32 saxpy_param_1,
      .param .u64 saxpy_param_2,
      .param .u64 saxpy_param_3
    )
    {
      .reg .pred 	%p<2>;
      .reg .f32 	%f<5>;
      .reg .b32 	%r<6>;
      .reg .b64 	%rd<8>;


      ld.param.u32 	%r2, [saxpy_param_0];
      ld.param.f32 	%f1, [saxpy_param_1];
      ld.param.u64 	%rd1, [saxpy_param_2];
      ld.param.u64 	%rd2, [saxpy_param_3];
      mov.u32 	%r3, %ctaid.x;
      mov.u32 	%r4, %ntid.x;
      mov.u32 	%r5, %tid.x;
      mad.lo.s32 	%r1, %r4, %r3, %r5;
      setp.ge.s32	%p1, %r1, %r2;
      @%p1 bra 	BB0_2;

      cvta.to.global.u64 	%rd3, %rd2;
      cvta.to.global.u64 	%rd4, %rd1;
      mul.wide.s32 	%rd5, %r1, 4;
      add.s64 	%rd6, %rd4, %rd5;
      ld.global.f32 	%f2, [%rd6];
      add.s64 	%rd7, %rd3, %rd5;
      ld.global.f32 	%f3, [%rd7];
      fma.rn.f32 	%f4, %f2, %f1, %f3;
      st.global.f32 	[%rd7], %f4;

    BB0_2:
      ret;
    }

     // .globl	vector_add
    .visible .entry vector_add(
      .param .u32 vector_add_param_0,
      .param .u64 vector_add_param_1,
      .param .u64 vector_add_param_2,
      .param .u64 vector_add_param_3
    )
    {
      .reg .pred 	%p<2>;
      .reg .f32 	%f<4>;
      .reg .b32 	%r<6>;
      .reg .b64 	%rd<11>;


      ld.param.u32 	%r2, [vector_add_param_0];
      ld.param.u64 	%rd1, [vector_add_param_1];
      ld.param.u64 	%rd2, [vector_add_param_2];
      ld.param.u64 	%rd3, [vector_add_param_3];
      mov.u32 	%r3, %ctaid.x;
      mov.u32 	%r4, %ntid.x;
      mov.u32 	%r5, %tid.x;
      mad.lo.s32 	%r1, %r4, %r3, %r5;
      setp.ge.s32	%p1, %r1, %r2;
      @%p1 bra 	BB1_2;

      cvta.to.global.u64 	%rd4, %rd1;
      mul.wide.s32 	%rd5, %r1, 4;
      add.s64 	%rd6, %rd4, %rd5;
      cvta.to.global.u64 	%rd7, %rd2;
      add.s64 	%rd8, %rd7, %rd5;
      ld.global.f32 	%f1, [%rd8];
      ld.global.f32 	%f2, [%rd6];
      add.f32 	%f3, %f2, %f1;
      cvta.to.global.u64 	%rd9, %rd3;
      add.s64 	%rd10, %rd9, %rd5;
      st.global.f32 	[%rd10], %f3;

    BB1_2:
      ret;
    })";

TEST_F(ModuleTableTest, SingleModuleMultiKernel) {
  const ModuleTable::Spec spec{{{kMultiKernelPtx, {"saxpy", "vector_add"}}}};

  TFRT_ASSERT_AND_ASSIGN(auto current, CtxSetCurrent(context_.get()));
  TFRT_ASSERT_AND_ASSIGN(auto module_table,
                         gpu::ModuleTable::Create(current, spec));
  const auto saxpy = module_table->GetFunction(ModuleFuncHandle{0});
  const auto vector_add = module_table->GetFunction(ModuleFuncHandle{1});

  TFRT_ASSERT_AND_ASSIGN(
      auto stream,
      stream::StreamCreate(current, stream::StreamFlags::NON_BLOCKING));

  constexpr int kVecSize{1024 * 1024};
  constexpr int kVecBytes{kVecSize * sizeof(float)};
  TFRT_ASSERT_AND_ASSIGN(
      stream::HostMemory<float> x,
      stream::MemHostAlloc<float>(current, kVecSize,
                                  stream::MemHostAllocFlags::DEVICEMAP));
  TFRT_ASSERT_AND_ASSIGN(
      stream::HostMemory<float> y,
      stream::MemHostAlloc<float>(current, kVecSize,
                                  stream::MemHostAllocFlags::DEVICEMAP));

  for (int i = 0; i < kVecSize; ++i) {
    x.get().raw()[i] = i;
    y.get().raw()[i] = -i;
  }

  TFRT_ASSERT_AND_ASSIGN(auto x_dev, stream::MemHostGetDevicePointer(x.get()));
  TFRT_ASSERT_AND_ASSIGN(auto y_dev, stream::MemHostGetDevicePointer(y.get()));
  TFRT_ASSERT_AND_ASSIGN(stream::DeviceMemory<float> z_dev,
                         stream::MemAlloc<float>(current, kVecSize));

  // First we do z <- x + y
  EXPECT_TRUE(IsSuccess(
      LaunchKernel(current, vector_add, /*grid_dim=*/{{kVecSize / 256, 1, 1}},
                   /*block_dim=*/{{256, 1, 1}},
                   /*shared_memory_size_bytes=*/0, stream.get(), kVecSize,
                   x_dev.raw(), y_dev.raw(), z_dev.get().raw())));
  // Then y <- 2 * x + y
  const float alpha = 2.0;
  EXPECT_TRUE(IsSuccess(
      LaunchKernel(current, saxpy, /*grid_dim=*/{{kVecSize / 256, 1, 1}},
                   /*block_dim=*/{{256, 1, 1}},
                   /*shared_memory_size_bytes=*/0, stream.get(), kVecSize,
                   alpha, x_dev.raw(), y_dev.raw())));

  // Copy z_dev to x.
  ASSERT_TRUE(IsSuccess(stream::MemcpyAsync(current, x.get(), z_dev.get(),
                                            kVecBytes, stream.get())));

  ASSERT_TRUE(IsSuccess(stream::StreamSynchronize(stream.get())));

  for (int i = 0; i < kVecSize; ++i) {
    // Since z was x + y and x = -y, then z should be all zeros.
    ASSERT_THAT(x.get().raw()[i], testing::FloatNear(0, 0.01))
        << "Expected value to be 0, but found " << x.get().raw()[i]
        << " for index " << i;

    // Since y was assigned 2x +y and x = -y, we expect y to be the original x.
    ASSERT_THAT(y.get().raw()[i], testing::FloatNear(i, 0.01))
        << "Expected value to match index, but found " << y.get().raw()[i]
        << " for index " << i;
  }
}

// This module is generated from the following code:
// __global__ void vector_sub(int n, const float* x, const float* y, float* z) {
//   int i = blockIdx.x  * blockDim.x + threadIdx.x;
//   if (i < n) z[i] = x[i] - y[i];
// }
//
// __global__ void scale(int n, float alpha,  float* x) {
//   int i = blockIdx.x  * blockDim.x + threadIdx.x;
//   if (i < n) x[i] = alpha * x[i];
// }
constexpr string_view kMultiKernelPtxAlt = R"(
.version 6.4
.target sm_30
.address_size 64

  // .globl	vector_sub

.visible .entry vector_sub(
  .param .u32 vector_sub_param_0,
  .param .u64 vector_sub_param_1,
  .param .u64 vector_sub_param_2,
  .param .u64 vector_sub_param_3
)
{
  .reg .pred 	%p<2>;
  .reg .f32 	%f<4>;
  .reg .b32 	%r<6>;
  .reg .b64 	%rd<11>;


  ld.param.u32 	%r2, [vector_sub_param_0];
  ld.param.u64 	%rd1, [vector_sub_param_1];
  ld.param.u64 	%rd2, [vector_sub_param_2];
  ld.param.u64 	%rd3, [vector_sub_param_3];
  mov.u32 	%r3, %ctaid.x;
  mov.u32 	%r4, %ntid.x;
  mov.u32 	%r5, %tid.x;
  mad.lo.s32 	%r1, %r4, %r3, %r5;
  setp.ge.s32	%p1, %r1, %r2;
  @%p1 bra 	BB0_2;

  cvta.to.global.u64 	%rd4, %rd1;
  mul.wide.s32 	%rd5, %r1, 4;
  add.s64 	%rd6, %rd4, %rd5;
  cvta.to.global.u64 	%rd7, %rd2;
  add.s64 	%rd8, %rd7, %rd5;
  ld.global.f32 	%f1, [%rd8];
  ld.global.f32 	%f2, [%rd6];
  sub.f32 	%f3, %f2, %f1;
  cvta.to.global.u64 	%rd9, %rd3;
  add.s64 	%rd10, %rd9, %rd5;
  st.global.f32 	[%rd10], %f3;

BB0_2:
  ret;
}

  // .globl	scale
.visible .entry scale(
  .param .u32 scale_param_0,
  .param .f32 scale_param_1,
  .param .u64 scale_param_2
)
{
  .reg .pred 	%p<2>;
  .reg .f32 	%f<4>;
  .reg .b32 	%r<6>;
  .reg .b64 	%rd<5>;


  ld.param.u32 	%r2, [scale_param_0];
  ld.param.f32 	%f1, [scale_param_1];
  ld.param.u64 	%rd1, [scale_param_2];
  mov.u32 	%r3, %ctaid.x;
  mov.u32 	%r4, %ntid.x;
  mov.u32 	%r5, %tid.x;
  mad.lo.s32 	%r1, %r4, %r3, %r5;
  setp.ge.s32	%p1, %r1, %r2;
  @%p1 bra 	BB1_2;

  cvta.to.global.u64 	%rd2, %rd1;
  mul.wide.s32 	%rd3, %r1, 4;
  add.s64 	%rd4, %rd2, %rd3;
  ld.global.f32 	%f2, [%rd4];
  mul.f32 	%f3, %f2, %f1;
  st.global.f32 	[%rd4], %f3;

BB1_2:
	ret;
})";

TEST_F(ModuleTableTest, MultiModuleMultiKernel) {
  // Load declarations out of declaration order. The module table indexes them
  // by the order they appear in the spec.
  const ModuleTable::Spec spec{{{kMultiKernelPtxAlt, {"scale", "vector_sub"}},
                                {kMultiKernelPtx, {"vector_add", "saxpy"}}}};

  TFRT_ASSERT_AND_ASSIGN(auto current, CtxSetCurrent(context_.get()));
  TFRT_ASSERT_AND_ASSIGN(auto module_table,
                         gpu::ModuleTable::Create(current, spec));
  const auto vector_sub = module_table->GetFunction(ModuleFuncHandle{1});
  const auto vector_add = module_table->GetFunction(ModuleFuncHandle{2});
  const auto saxpy = module_table->GetFunction(ModuleFuncHandle{3});

  TFRT_ASSERT_AND_ASSIGN(
      auto stream,
      stream::StreamCreate(current, stream::StreamFlags::NON_BLOCKING));

  constexpr int kVecSize{1024 * 1024};
  constexpr int kVecBytes{kVecSize * sizeof(float)};
  TFRT_ASSERT_AND_ASSIGN(
      stream::HostMemory<float> x,
      stream::MemHostAlloc<float>(current, kVecSize,
                                  stream::MemHostAllocFlags::DEVICEMAP));
  TFRT_ASSERT_AND_ASSIGN(
      stream::HostMemory<float> y,
      stream::MemHostAlloc<float>(current, kVecSize,
                                  stream::MemHostAllocFlags::DEVICEMAP));

  TFRT_ASSERT_AND_ASSIGN(
      stream::HostMemory<float> add_result,
      stream::MemHostAlloc<float>(current, kVecSize,
                                  stream::MemHostAllocFlags::DEFAULT));

  for (int i = 0; i < kVecSize; ++i) {
    x.get().raw()[i] = i;
    y.get().raw()[i] = -i;
  }

  TFRT_ASSERT_AND_ASSIGN(auto x_dev, stream::MemHostGetDevicePointer(x.get()));
  TFRT_ASSERT_AND_ASSIGN(auto y_dev, stream::MemHostGetDevicePointer(y.get()));
  TFRT_ASSERT_AND_ASSIGN(stream::DeviceMemory<float> z_dev,
                         stream::MemAlloc<float>(current, kVecSize));

  // First we do z <- x + y
  EXPECT_TRUE(IsSuccess(
      LaunchKernel(current, vector_add, /*grid_dim=*/{{kVecSize / 256, 1, 1}},
                   /*block_dim=*/{{256, 1, 1}},
                   /*shared_memory_size_bytes=*/0, stream.get(), kVecSize,
                   x_dev.raw(), y_dev.raw(), z_dev.get().raw())));

  // Copy out z to add_result.
  ASSERT_TRUE(IsSuccess(stream::MemcpyAsync(
      current, add_result.get(), z_dev.get(), kVecBytes, stream.get())));

  // Then z <- y - x, which should be -2x.
  EXPECT_TRUE(IsSuccess(
      LaunchKernel(current, vector_sub, /*grid_dim=*/{{kVecSize / 256, 1, 1}},
                   /*block_dim=*/{{256, 1, 1}},
                   /*shared_memory_size_bytes=*/0, stream.get(), kVecSize,
                   y_dev.raw(), x_dev.raw(), z_dev.get().raw())));

  // Then y <- 2 * x + y
  const float alpha = 2.0;
  EXPECT_TRUE(IsSuccess(
      LaunchKernel(current, saxpy, /*grid_dim=*/{{kVecSize / 256, 1, 1}},
                   /*block_dim=*/{{256, 1, 1}},
                   /*shared_memory_size_bytes=*/0, stream.get(), kVecSize,
                   alpha, x_dev.raw(), y_dev.raw())));

  // Copy out z to x.
  ASSERT_TRUE(IsSuccess(stream::MemcpyAsync(current, x.get(), z_dev.get(),
                                            kVecBytes, stream.get())));

  ASSERT_TRUE(IsSuccess(stream::StreamSynchronize(stream.get())));

  for (int i = 0; i < kVecSize; ++i) {
    // Since z was x + y and x = -y, then z should be all zeros.
    ASSERT_THAT(add_result.get().raw()[i], testing::FloatNear(0, 0.01))
        << "Expected value to be 0, but found " << add_result.get().raw()[i]
        << " for index " << i;

    // Since y was assigned 2x +y and x = -y, we expect y to be the original x.
    ASSERT_THAT(y.get().raw()[i], testing::FloatNear(i, 0.01))
        << "Expected value to match index, but found " << y.get().raw()[i]
        << " for index " << i;

    ASSERT_THAT(x.get().raw()[i], testing::FloatNear(-2 * i, 0.01))
        << "Expected value to match 2  * -index but found " << x.get().raw()[i]
        << " for index " << i;
  }
}

TEST_F(ModuleTableTest, MultiDeviceModuleTable) {
  class DummyModuleTable : public ModuleTable {
   public:
    stream::Function GetFunction(ModuleFuncHandle handle) const override {
      return {};
    }
  };

  std::array<stream::Device, 3> device{
      stream::Device(0, stream::Platform::CUDA),
      stream::Device(1, stream::Platform::CUDA),
      stream::Device(2, stream::Platform::CUDA),
  };
  std::array<std::unique_ptr<ModuleTable>, 3> owned_module_tables{
      std::make_unique<DummyModuleTable>(),
      std::make_unique<DummyModuleTable>(),
      std::make_unique<DummyModuleTable>(),
  };
  std::array<ModuleTable*, 3> module_tables{
      owned_module_tables[0].get(),
      owned_module_tables[1].get(),
      owned_module_tables[2].get(),
  };

  auto multi_device_table = MultiDeviceModuleTable::Create();
  ASSERT_TRUE(IsSuccess(multi_device_table->AddTable(
      device[0], std::move(owned_module_tables[0]))));
  EXPECT_TRUE(multi_device_table->GetTable(device[0]).hasValue());
  EXPECT_EQ(*multi_device_table->GetTable(device[0]), module_tables[0]);

  ASSERT_TRUE(IsSuccess(multi_device_table->AddTable(
      device[2], std::move(owned_module_tables[2]))));
  // Verify that pre-existing table is unaffected.
  EXPECT_TRUE(multi_device_table->GetTable(device[0]).hasValue());
  EXPECT_EQ(*multi_device_table->GetTable(device[0]), module_tables[0]);

  EXPECT_TRUE(multi_device_table->GetTable(device[2]).hasValue());
  EXPECT_EQ(*multi_device_table->GetTable(device[2]), module_tables[2]);

  ASSERT_TRUE(IsSuccess(multi_device_table->AddTable(
      device[1], std::move(owned_module_tables[1]))));
  // Verify pre-existing tables
  EXPECT_TRUE(multi_device_table->GetTable(device[0]).hasValue());
  EXPECT_EQ(*multi_device_table->GetTable(device[0]), module_tables[0]);
  EXPECT_TRUE(multi_device_table->GetTable(device[2]).hasValue());
  EXPECT_EQ(*multi_device_table->GetTable(device[2]), module_tables[2]);

  EXPECT_TRUE(multi_device_table->GetTable(device[1]).hasValue());
  EXPECT_EQ(*multi_device_table->GetTable(device[1]), module_tables[1]);
}

}  // namespace
}  // namespace gpu
}  // namespace tfrt
