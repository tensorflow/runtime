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

// Unit test for driver wrapper (abstraction layer for CUDA and HIP).

#include "tfrt/gpu/wrapper/driver_wrapper.h"

#include "common.h"
#include "tfrt/gpu/wrapper/cuda_wrapper.h"
#include "tfrt/gpu/wrapper/hip_wrapper.h"

namespace tfrt {
namespace gpu {
namespace wrapper {
using ::testing::AllOf;
using ::testing::AssertionFailure;
using ::testing::AssertionResult;
using ::testing::AssertionSuccess;
using ::testing::ContainsRegex;
using ::testing::Eq;
using ::testing::HasSubstr;

static AssertionResult IsInvalidContextError(llvm::Error&& error) {
  if (!error) return AssertionFailure() << error;
  if (auto handled = llvm::handleErrors(
          std::move(error),
          [](std::unique_ptr<ErrorInfo<CUresult>> info) -> llvm::Error {
            if (GetResult(*info) == CUDA_ERROR_INVALID_CONTEXT)
              return llvm::Error::success();
            return llvm::Error(std::move(info));
          },
          [](std::unique_ptr<ErrorInfo<hipError_t>> info) -> llvm::Error {
            if (GetResult(*info) == hipErrorInvalidContext)
              return llvm::Error::success();
            return llvm::Error(std::move(info));
          }))
    return AssertionFailure() << handled;
  return AssertionSuccess();
}

// Helper function to clear current context.
static AssertionResult SetNullContext(Platform platform) {
  for (;;) {
    // Setting null context requires multiple tries because it may at first
    // restore a primary context.
    auto current = CtxSetCurrent({nullptr, platform});
    if (!current) return AssertionFailure() << current.takeError();
    if (*current == nullptr) return AssertionSuccess();
  }
}

TEST_P(Test, TestLogError) {
  auto platform = GetParam();
  ASSERT_THAT(Init(platform), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  std::string error_str = toString(DeviceGet(platform, count).takeError());
  if (platform == Platform::CUDA) {
    EXPECT_THAT(error_str, AllOf(HasSubstr("cuDeviceGet"),
                                 HasSubstr("CUDA_ERROR_INVALID_DEVICE"),
                                 HasSubstr("invalid device ordinal")));
  }
  if (platform == Platform::ROCm) {
    EXPECT_THAT(error_str,
                AllOf(HasSubstr("hipDeviceGet"), HasSubstr("hipInvalidDevice"),
                      HasSubstr("invalid device ordinal")));
  }
}

TEST_P(Test, DriverVersionIsGreaterZero) {
  auto platform = GetParam();
  ASSERT_THAT(Init(platform), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto version, DriverGetVersion(platform));
  EXPECT_GT(version, 0);
}

TEST_P(Test, TestDeviceProperties) {
  auto platform = GetParam();
  ASSERT_THAT(Init(platform), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  TFRT_ASSERT_AND_ASSIGN(auto name, DeviceGetName(device));
  EXPECT_FALSE(name.empty());
  TFRT_ASSERT_AND_ASSIGN(auto mem_size, DeviceTotalMem(device));
  EXPECT_GT(mem_size, 0);
  if (platform == Platform::CUDA) {
    TFRT_ASSERT_AND_ASSIGN(
        auto warp_size,
        CuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_WARP_SIZE, device));
    EXPECT_EQ(warp_size, 32);
  } else {
    TFRT_ASSERT_AND_ASSIGN(
        auto warp_size,
        HipDeviceGetAttribute(hipDeviceAttributeWarpSize, device));
    EXPECT_EQ(warp_size, 64);
  }
}

TEST_P(Test, CreatedContextIsCurrent) {
  auto platform = GetParam();
  ASSERT_THAT(Init(platform), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, CtxCreate(CtxFlags::SCHED_AUTO, device));
  EXPECT_NE(context.get(), nullptr);
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxGetCurrent());
  EXPECT_EQ(context.get(), current.context());
}

TEST_P(Test, RetainedPrimaryContextIsNotCurrent) {
  auto platform = GetParam();
  ASSERT_THAT(Init(platform), IsSuccess());
  EXPECT_TRUE(SetNullContext(platform));
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, DevicePrimaryCtxRetain(device));
  EXPECT_NE(context.get(), nullptr);
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxGetCurrent());
  EXPECT_EQ(current, nullptr);
}

TEST_P(Test, TestMultipleContexts) {
  auto platform = GetParam();
  ASSERT_THAT(Init(platform), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context1,
                         CtxCreate(CtxFlags::SCHED_AUTO, device));
  EXPECT_EQ(CtxGetCurrent()->context(), context1.get());
  TFRT_ASSERT_AND_ASSIGN(auto context2,
                         CtxCreate(CtxFlags::SCHED_AUTO, device));
  EXPECT_EQ(CtxGetCurrent()->context(), context2.get());
  TFRT_ASSERT_AND_ASSIGN(auto current1, CtxSetCurrent(context1.get()));
  EXPECT_NE(current1.context(), context2.get());
}

TEST_P(Test, TestDestroyResourcesWithoutCurrentContext) {
  auto platform = GetParam();
  ASSERT_THAT(Init(platform), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, CtxCreate(CtxFlags::SCHED_AUTO, device));

  TFRT_ASSERT_AND_ASSIGN(auto device_ptr,
                         MemAlloc(Current(), /*size_bytes=*/8));
  TFRT_ASSERT_AND_ASSIGN(
      auto host_ptr,
      MemHostAlloc(Current(), /*size_bytes=*/8, MemHostAllocFlags::PORTABLE));
  TFRT_ASSERT_AND_ASSIGN(
      auto managed_ptr,
      MemAllocManaged(Current(), /*size_bytes=*/8, MemAttachFlags::GLOBAL));
  char buffer[8];
  TFRT_ASSERT_AND_ASSIGN(auto registered_ptr,
                         MemHostRegister(Current(), buffer, sizeof(buffer),
                                         MemHostRegisterFlags::DEVICEMAP));

  EXPECT_TRUE(SetNullContext(platform));
}

TEST_P(Test, TestStreamsWithoutCurrentContext) {
  auto platform = GetParam();
  ASSERT_THAT(Init(platform), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, CtxCreate(CtxFlags::SCHED_AUTO, device));

  TFRT_ASSERT_AND_ASSIGN(auto stream,
                         StreamCreate(Current(), StreamFlags::DEFAULT));
  EXPECT_TRUE(SetNullContext(platform));

  TFRT_ASSERT_AND_ASSIGN(auto ready, StreamQuery(stream.get()));
  EXPECT_EQ(ready, true);

  EXPECT_THAT(StreamSynchronize(stream.get()), IsSuccess());
  EXPECT_TRUE(IsInvalidContextError(StreamSynchronize({nullptr, platform})));
}

TEST_P(Test, TestEventsWithoutCurrentContext) {
  auto platform = GetParam();
  ASSERT_THAT(Init(platform), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, CtxCreate(CtxFlags::SCHED_AUTO, device));

  TFRT_ASSERT_AND_ASSIGN(auto stream,
                         StreamCreate(Current(), StreamFlags::DEFAULT));
  TFRT_ASSERT_AND_ASSIGN(auto event,
                         EventCreate(Current(), EventFlags::DEFAULT));
  EXPECT_TRUE(SetNullContext(platform));

  TFRT_ASSERT_AND_ASSIGN(auto ready, EventQuery(event.get()));
  EXPECT_EQ(ready, true);

  EXPECT_THAT(EventRecord(event.get(), stream.get()), IsSuccess());
  EXPECT_THAT(StreamWaitEvent(stream.get(), event.get()), IsSuccess());

  EXPECT_THAT(EventSynchronize(event.get()), IsSuccess());

  Stream null_stream(nullptr, platform);
  EXPECT_TRUE(IsInvalidContextError(EventRecord(event.get(), null_stream)));
  EXPECT_TRUE(IsInvalidContextError(StreamWaitEvent(null_stream, event.get())));
}

TEST_P(Test, TestContextFlags) {
  auto platform = GetParam();
  ASSERT_THAT(Init(platform), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  auto flags = CtxFlags::SCHED_SPIN | CtxFlags::LMEM_RESIZE_TO_MAX;
  TFRT_ASSERT_AND_ASSIGN(auto context, CtxCreate(flags, device));
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxGetCurrent());
  TFRT_ASSERT_AND_ASSIGN(auto get_flags, CtxGetFlags(current));
  EXPECT_EQ(flags, get_flags);
}

TEST_P(Test, TestContextDevice) {
  auto platform = GetParam();
  ASSERT_THAT(Init(platform), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, CtxCreate(CtxFlags::SCHED_AUTO, device));
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxGetCurrent());
  TFRT_ASSERT_AND_ASSIGN(auto get_device, CtxGetDevice(current));
  EXPECT_EQ(device, get_device);
}

TEST_F(Test, TestContextLimitCUDA) {
  auto platform = Platform::CUDA;
  ASSERT_THAT(Init(platform), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, CtxCreate(CtxFlags::SCHED_AUTO, device));
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxGetCurrent());
  auto limit = CU_LIMIT_PRINTF_FIFO_SIZE;
  size_t value = 1024 * 1024;
  EXPECT_THAT(CuCtxSetLimit(current, limit, value), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto get_value, CuCtxGetLimit(current, limit));
  EXPECT_EQ(value, get_value);
}

TEST_F(Test, TestContextCacheConfigCUDA) {
  auto platform = Platform::CUDA;
  ASSERT_THAT(Init(platform), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, CtxCreate(CtxFlags::SCHED_AUTO, device));
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxGetCurrent());
  auto cache_cfg = CU_FUNC_CACHE_PREFER_SHARED;
  EXPECT_THAT(CuCtxSetCacheConfig(current, cache_cfg), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto get_cache_cfg, CuCtxGetCacheConfig(current));
  EXPECT_EQ(cache_cfg, get_cache_cfg);
}

TEST_F(Test, TestContextSharedConfigCUDA) {
  auto platform = Platform::CUDA;
  ASSERT_THAT(Init(platform), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, CtxCreate(CtxFlags::SCHED_AUTO, device));
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxGetCurrent());
  auto shread_cfg = CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE;
  EXPECT_THAT(CuCtxSetSharedMemConfig(current, shread_cfg), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto get_shread_cfg, CuCtxGetSharedMemConfig(current));
  EXPECT_EQ(shread_cfg, get_shread_cfg);
}

TEST_P(Test, ContextApiVersionIsGreaterZero) {
  auto platform = GetParam();
  ASSERT_THAT(Init(platform), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, CtxCreate(CtxFlags::SCHED_AUTO, device));
  TFRT_ASSERT_AND_ASSIGN(auto version, CtxGetApiVersion(context.get()));
  EXPECT_GT(version, 0);
}

TEST_P(Test, TestContextStreamPriorityRange) {
  auto platform = GetParam();
  ASSERT_THAT(Init(platform), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, CtxCreate(CtxFlags::SCHED_AUTO, device));
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxGetCurrent());
  TFRT_ASSERT_AND_ASSIGN(auto range, CtxGetStreamPriorityRange(current));
  EXPECT_GE(range.least, 0);
  EXPECT_LE(range.greatest, 0);
}

TEST_P(Test, TestPrimaryContextState) {
  auto platform = GetParam();
  ASSERT_THAT(Init(platform), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  EXPECT_THAT(DevicePrimaryCtxReset(device), IsSuccess());
  auto flags = CtxFlags::SCHED_SPIN | CtxFlags::LMEM_RESIZE_TO_MAX;
  EXPECT_THAT(DevicePrimaryCtxSetFlags(device, flags), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto context, DevicePrimaryCtxRetain(device));
  TFRT_ASSERT_AND_ASSIGN(auto state, DevicePrimaryCtxGetState(device));
  EXPECT_NE(state.active, 0);
  EXPECT_EQ(state.flags, flags);
  // Prevent cleanup from corrupting memory. See b/154999929.
  EXPECT_TRUE(SetNullContext(platform));
}

TEST_P(Test, TestPrimaryContextReset) {
  auto platform = GetParam();
  ASSERT_THAT(Init(platform), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, DevicePrimaryCtxRetain(device));
  EXPECT_THAT(DevicePrimaryCtxReset(device), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxSetCurrent(context.get()));
  ASSERT_EQ(current.platform(), platform);
}

TEST_P(Test, TestNoCurrentContext) {
#ifdef NDEBUG
  GTEST_SKIP() << "CheckNoCurrentContext not implemented in NDEBUG builds";
#endif
  auto platform = GetParam();
  ASSERT_THAT(Init(platform), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, DevicePrimaryCtxRetain(device));
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxSetCurrent(context.get()));
  (void)current;
  // It's valid to create multiple CurrentContext instances.
  EXPECT_THAT(CtxGetCurrent().takeError(), IsSuccess());

  const char* error_str = "Existing CurrentContext instance(s) in same thread.";
  EXPECT_THAT(toString(DevicePrimaryCtxRelease(device)), Eq(error_str));
  EXPECT_THAT(toString(DevicePrimaryCtxReset(device)), Eq(error_str));
  EXPECT_THAT(toString(CtxCreate(CtxFlags::SCHED_AUTO, device).takeError()),
              Eq(error_str));
  EXPECT_THAT(toString(CtxSetCurrent(context.get()).takeError()),
              Eq(error_str));
  EXPECT_THAT(toString(CtxDestroy(context.get())), Eq(error_str));
}

TEST_P(Test, TestDestroyNullResources) {
  auto platform = GetParam();
  EXPECT_THAT(CtxDestroy({nullptr, platform}), IsSuccess());
  EXPECT_THAT(ModuleUnload({nullptr, platform}), IsSuccess());
  EXPECT_THAT(StreamDestroy({nullptr, platform}), IsSuccess());
  EXPECT_THAT(EventDestroy({nullptr, platform}), IsSuccess());
}

TEST_F(Test, TestModuleLoadDataCUDA) {
  auto platform = Platform::CUDA;
  ASSERT_THAT(Init(platform), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, CtxCreate(CtxFlags::SCHED_AUTO, device));
  unsigned int log_buffer_size = 1024 * 1024;
  auto log_buffer = std::make_unique<char[]>(log_buffer_size);
  std::vector<CUjit_option> jit_options = {
      CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, CU_JIT_INFO_LOG_BUFFER,
      CU_JIT_LOG_VERBOSE, CU_JIT_WALL_TIME};
  std::vector<void*> jit_values = {
      static_cast<char*>(nullptr) + log_buffer_size, log_buffer.get(),
      static_cast<char*>(nullptr) + 1, nullptr};
  int max_threads_per_multiprocessor;
  int max_threads_per_block;
  int shared_mem_per_block;
  int multiprocessor_count;

  TFRT_ASSERT_AND_ASSIGN(
      max_threads_per_multiprocessor,
      CuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
                           device));
  TFRT_ASSERT_AND_ASSIGN(
      max_threads_per_block,
      CuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device));
  TFRT_ASSERT_AND_ASSIGN(
      shared_mem_per_block,
      CuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK,
                           device));
  TFRT_ASSERT_AND_ASSIGN(
      multiprocessor_count,
      CuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));

  // PTX string of an empty kernel.
  const char* kernel_ptx = R"(
        .version 6.0
        .target sm_35
        .address_size 64

        .visible .entry Kernel() {
          ret;
        })";

  // Note: requesting logs causes memory leaks.
  TFRT_ASSERT_AND_ASSIGN(
      auto module,
      CuModuleLoadDataEx(Current(), kernel_ptx, jit_options, jit_values));

  EXPECT_GT(strlen(log_buffer.get()), 0);
  float wall_time =
      reinterpret_cast<const std::array<float, 2>&>(jit_values[3])[0];
  EXPECT_GT(wall_time, 0.0f);

  EXPECT_THAT(CtxSetCurrent({nullptr, platform}).takeError(),
              IsSuccess());  // Verify no context needed.
  TFRT_ASSERT_AND_ASSIGN(auto function,
                         ModuleGetFunction(module.get(), "Kernel"));

  TFRT_ASSERT_AND_ASSIGN(auto current, CtxSetCurrent(context.get()));
  int block_size = 128;
  TFRT_ASSERT_AND_ASSIGN(auto max_blocks_per_sm,
                         OccupancyMaxActiveBlocksPerMultiprocessor(
                             current, function, block_size,
                             /*dynamic_shared_memory_size=*/4 * 1024));
  EXPECT_EQ(max_blocks_per_sm * block_size, max_threads_per_multiprocessor);

  size_t shared_memory_per_thread = 256;
  TFRT_ASSERT_AND_ASSIGN(
      auto max_potential_block_size,
      CuOccupancyMaxPotentialBlockSize(
          current, function,
          [&](int block_size) { return block_size * shared_memory_per_thread; },
          max_threads_per_block));
  // Not 'EXPECT_EQ' because using the maximum amount of shared memory disables
  // half of the multi-processors on some architectures, see
  // https://docs.nvidia.com/cuda/maxwell-tuning-guide/index.html#shared-memory-capacity
  EXPECT_LE(max_potential_block_size.block_size * shared_memory_per_thread,
            shared_mem_per_block);
  // We expect to use at least 32kB.
  EXPECT_GE(max_potential_block_size.block_size * shared_memory_per_thread,
            32 * 1024);
  EXPECT_LE(max_potential_block_size.min_num_blocks *
                max_potential_block_size.block_size,
            multiprocessor_count * max_threads_per_multiprocessor);

  auto stream = Stream(nullptr, platform);
  EXPECT_THAT(LaunchKernel(current, function, /*grid_dim=*/{{1, 1, 1}},
                           /*block_dim=*/{{1, 1, 1}},
                           /*shared_memory_size_bytes=*/0, stream),
              IsSuccess());
  EXPECT_THAT(CtxSynchronize(current), IsSuccess());
}

TEST_P(Test, MemcpyRequeriesContext) {
  auto platform = GetParam();
  ASSERT_THAT(Init(platform), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, CtxCreate(CtxFlags::SCHED_AUTO, device));
  size_t size_bytes = 1024 * 1024;

  TFRT_ASSERT_AND_ASSIGN(auto src, MemAlloc(Current(), size_bytes));
  TFRT_ASSERT_AND_ASSIGN(auto dst, MemAlloc(Current(), size_bytes));

  EXPECT_TRUE(SetNullContext(platform));
  TFRT_ASSERT_AND_ASSIGN(auto no_current, CtxGetCurrent());
  EXPECT_EQ(no_current, nullptr);
  EXPECT_TRUE(IsInvalidContextError(
      Memcpy(no_current, dst.get(), src.get(), size_bytes)));
}

TEST_P(Test, TestStreamProperties) {
  auto platform = GetParam();
  ASSERT_THAT(Init(platform), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, CtxCreate(CtxFlags::SCHED_AUTO, device));
  auto flags = StreamFlags::NON_BLOCKING;
  int priority = -1;

  TFRT_ASSERT_AND_ASSIGN(auto stream, StreamCreate(Current(), flags, priority));
  EXPECT_THAT(CtxSetCurrent({nullptr, platform}).takeError(),
              IsSuccess());  // Verify no context needed.
  TFRT_ASSERT_AND_ASSIGN(auto get_priority, StreamGetPriority(stream.get()));
  EXPECT_EQ(priority, get_priority);
  TFRT_ASSERT_AND_ASSIGN(auto get_flags, StreamGetFlags(stream.get()));
  EXPECT_EQ(flags, get_flags);
}

TEST_F(Test, TestEventsCUDA) {
  auto platform = Platform::CUDA;
  ASSERT_THAT(Init(platform), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, CtxCreate(CtxFlags::SCHED_AUTO, device));
  EXPECT_THAT(CtxGetCurrent().takeError(), IsSuccess());

  int expected = 1;
  int desired = 2;

  TFRT_ASSERT_AND_ASSIGN(auto stream,
                         StreamCreate(Current(), StreamFlags::DEFAULT));
  TFRT_ASSERT_AND_ASSIGN(auto start,
                         EventCreate(Current(), EventFlags::DEFAULT));
  TFRT_ASSERT_AND_ASSIGN(auto stop,
                         EventCreate(Current(), EventFlags::DEFAULT));
  EXPECT_THAT(EventRecord(start.get(), stream.get()), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto dst, MemAlloc<int>(Current(), 1));
  EXPECT_THAT(MemsetD32(Current(), dst.get(), /*value=*/0, /*count=*/1),
              IsSuccess());

  // PTX string of a kernel which blocks until *address == expected and then
  // writes desired to *address.
  const char* kernel_ptx = R"(
        .version 6.0
        .target sm_35
        .address_size 64

        .visible .entry Kernel(
          .param .u64 _address,
          .param .u32 _expected,
          .param .u32 _desired
        ) {
          .reg .b64  %address;
          .reg .b32  %expected;
          .reg .b32  %desired;
          .reg .b32  %value;
          .reg .pred %pred;

          ld.param.u64 %address,  [_address];
          ld.param.u32 %expected, [_expected];
          ld.param.u32 %desired,  [_desired];

        while:
          atom.cas.b32 %value, [%address], %expected, %desired;
          setp.ne.s32  %pred, %value, %expected;
          @%pred bra while;

          ret;
        })";
  TFRT_ASSERT_AND_ASSIGN(auto module, ModuleLoadData(Current(), kernel_ptx));

  TFRT_ASSERT_AND_ASSIGN(auto function,
                         ModuleGetFunction(module.get(), "Kernel"));
  ASSERT_NE(function, nullptr);

  // Launch kernel that blocks until 'dst' is set to 1 below.
  EXPECT_THAT(CuLaunchKernel(Current(), function, /*grid_dim=*/{{1, 1, 1}},
                             /*block_dim=*/{{1, 1, 1}},
                             /*shared_memory_size_bytes=*/0, stream.get(),
                             dst.get().raw(platform), expected, desired),
              IsSuccess());

  EXPECT_THAT(CtxSetCurrent({nullptr, platform}).takeError(),
              IsSuccess());  // Verify no context needed.
  EXPECT_THAT(EventRecord(stop.get(), stream.get()), IsSuccess());
  EXPECT_THAT(EventSynchronize(start.get()), IsSuccess());

  bool stream_ready, event_ready;
  // Stream and event are not ready because kernel execution is blocking.
  TFRT_ASSERT_AND_ASSIGN(stream_ready, StreamQuery(stream.get()));
  EXPECT_FALSE(stream_ready);
  TFRT_ASSERT_AND_ASSIGN(event_ready, EventQuery(stop.get()));
  EXPECT_FALSE(event_ready);

  EXPECT_THAT(CtxSetCurrent(context.get()).takeError(), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto other_stream,
                         StreamCreate(Current(), StreamFlags::DEFAULT));
  // Unblock kernel execution.
  EXPECT_THAT(MemsetD32Async(Current(), dst.get(), expected,
                             /*count=*/1, other_stream.get()),
              IsSuccess());

  EXPECT_THAT(CtxSetCurrent({nullptr, platform}).takeError(),
              IsSuccess());  // Verify no context needed.

  EXPECT_THAT(StreamSynchronize(stream.get()), IsSuccess());
  // Stream and event are ready after stream synchronization.
  TFRT_ASSERT_AND_ASSIGN(stream_ready, StreamQuery(stream.get()));
  EXPECT_TRUE(stream_ready);
  TFRT_ASSERT_AND_ASSIGN(event_ready, EventQuery(stop.get()));
  EXPECT_TRUE(event_ready);

  TFRT_ASSERT_AND_ASSIGN(auto time_ms,
                         EventElapsedTime(start.get(), stop.get()));
  EXPECT_GT(time_ms, 0.0f);
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxSetCurrent(context.get()));
  int result;
  EXPECT_THAT(Memcpy(current, {&result, platform}, dst.get(), sizeof(int)),
              IsSuccess());
  EXPECT_EQ(result, desired);
}

TEST_P(Test, UnalignedPointeeType) {
  auto platform = GetParam();
  Pointer<const char>(reinterpret_cast<const char*>(0x1), platform);
  Pointer<unsigned char>(reinterpret_cast<unsigned char*>(0x1), platform);
}

TEST_P(Test, MemHostGetDevicePointer) {
  auto platform = GetParam();
  ASSERT_THAT(Init(platform), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, CtxCreate(CtxFlags::SCHED_AUTO, device));
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxGetCurrent());
  char buffer[32];
  TFRT_ASSERT_AND_ASSIGN(auto registered_ptr,
                         MemHostRegister(current, buffer, sizeof(buffer),
                                         MemHostRegisterFlags::DEVICEMAP));
  auto host_ptr = static_cast<Pointer<char>>(registered_ptr.get());
  TFRT_ASSERT_AND_ASSIGN(auto device_ptr, MemHostGetDevicePointer(host_ptr));
  EXPECT_NE(device_ptr, Pointer<char>(nullptr, platform));
}

TEST_P(Test, MemGetAddressRange) {
  auto platform = GetParam();
  ASSERT_THAT(Init(platform), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, CtxCreate(CtxFlags::SCHED_AUTO, device));
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxGetCurrent());
  size_t size_bytes = 32;
  TFRT_ASSERT_AND_ASSIGN(auto pointer, MemAlloc(current, size_bytes));
  auto char_ptr = static_cast<Pointer<char>>(pointer.get());
  TFRT_ASSERT_AND_ASSIGN(auto range, MemGetAddressRange(current, char_ptr));
  EXPECT_EQ(char_ptr, range.base);
  EXPECT_EQ(size_bytes, range.size_bytes);
}

TEST_P(Test, OutOfMemory) {
  auto platform = GetParam();
  ASSERT_THAT(Init(platform), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));
  TFRT_ASSERT_AND_ASSIGN(auto context, CtxCreate(CtxFlags::SCHED_AUTO, device));
  TFRT_ASSERT_AND_ASSIGN(auto current, CtxGetCurrent());
  size_t size_bytes = 1ull << 40;
  std::string log_string;
  llvm::raw_string_ostream(log_string)
      << MemAlloc(current, size_bytes).takeError();
  EXPECT_THAT(log_string,
              HasSubstr("Out of memory trying to allocate 1.00TiB"));
}

static void CheckDanglingResources(
    Expected<OwningContext> context,
    std::function<llvm::Error(Context)> destroy) {
  EXPECT_THAT(context.takeError(), IsSuccess());
  EXPECT_THAT(CtxSetCurrent(context->get()).takeError(), IsSuccess());

  TFRT_ASSERT_AND_ASSIGN(auto stream,
                         StreamCreate(Current(), StreamFlags::DEFAULT));
  TFRT_ASSERT_AND_ASSIGN(auto event,
                         EventCreate(Current(), EventFlags::DEFAULT));
  const size_t size_bytes = 32;
  TFRT_ASSERT_AND_ASSIGN(auto device_mem, MemAlloc(Current(), size_bytes));
  TFRT_ASSERT_AND_ASSIGN(
      auto host_mem,
      MemHostAlloc(Current(), size_bytes, MemHostAllocFlags::DEFAULT));
  char array[size_bytes];
  TFRT_ASSERT_AND_ASSIGN(auto registered_mem,
                         MemHostRegister(Current(), array, size_bytes,
                                         MemHostRegisterFlags::DEFAULT));

  // Release resources and let the driver destroy them with the context.
  stream.release();
  event.release();
  device_mem.release();
  host_mem.release();
  registered_mem.release();

  std::string log_string;
  llvm::raw_string_ostream(log_string) << destroy(context->release());

#ifndef NDEBUG  // Dangling resources are only detected in !NDEBUG builds.
  EXPECT_THAT(log_string,
              AllOf(HasSubstr("has dangling resources"),
                    ContainsRegex("0x[0-9a-f]+ \\(stream\\)"),
                    ContainsRegex("0x[0-9a-f]+ \\(event\\)"),
                    ContainsRegex("0x[0-9a-f]+ \\(device memory\\)"),
                    ContainsRegex("0x[0-9a-f]+ \\(host memory\\)"),
                    ContainsRegex("0x[0-9a-f]+ \\(registered memory\\)")));
#endif
}

TEST_P(Test, DanglingResources) {
  auto platform = GetParam();
  ASSERT_THAT(Init(platform), IsSuccess());
  TFRT_ASSERT_AND_ASSIGN(auto count, DeviceGetCount(platform));
  ASSERT_GT(count, 0);
  TFRT_ASSERT_AND_ASSIGN(auto device, DeviceGet(platform, 0));

  CheckDanglingResources(CtxCreate(CtxFlags::SCHED_AUTO, device), &CtxDestroy);
  CheckDanglingResources(DevicePrimaryCtxRetain(device), [&](Context) {
    return DevicePrimaryCtxRelease(device);
  });
  CheckDanglingResources(DevicePrimaryCtxRetain(device), [&](Context) {
    return DevicePrimaryCtxReset(device);
  });
}

}  // namespace wrapper
}  // namespace gpu
}  // namespace tfrt
