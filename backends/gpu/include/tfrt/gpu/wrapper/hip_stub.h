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

// Mostly auto-generated HIP API header.
#ifndef TFRT_GPU_WRAPPER_HIP_STUB_H_
#define TFRT_GPU_WRAPPER_HIP_STUB_H_

#include <array>
#include <cstdint>
#include <cstdlib>

#include "tfrt/gpu/wrapper/hip_forwards.h"

struct hipDim3_t {
  uint32_t x = 1;
  uint32_t y = 1;
  uint32_t z = 1;
};

// Define macros from hip_runtime.h used in hip_stub.h.inc.
#define __dparm(x)
#define DEPRECATED(x)
#define dim3 hipDim3_t

// Declare types from hip_runtime.h used in hip_stub.h.inc.
using hipDeviceptr_t = void*;
struct hipDeviceProp_t;
struct hipPointerAttribute_t;
struct hipFuncAttributes;
struct hipLaunchParams;

extern "C" {
#include "hip_stub.h.inc"
}

#undef dim3
#undef __dparm
#undef DEPRECATED

const char* hipGetErrorName(hipError_t hip_error);
const char* hipGetErrorString(hipError_t hip_error);
const char *hiprtcGetErrorString(hiprtcResult result);
hiprtcResult hiprtcVersion(int* major, int* minor);
hiprtcResult hiprtcAddNameExpression(hiprtcProgram prog, const char* name_expression);
hiprtcResult hiprtcCompileProgram(
                                   hiprtcProgram prog,
                                   int numOptions,
                                   const char** options);
hiprtcResult hiprtcCreateProgram(
                                  hiprtcProgram* prog,
                                  const char* src,
                                  const char* name,
                                  int numberHeaders,
                                  char** headers,
                                  const char** includeNames);
hiprtcResult hiprtcDestroyProgram(hiprtcProgram* prog);
hiprtcResult hiprtcGetLoweredName(
                                  hiprtcProgram prog,
                                  const char* name_expression,
                                  const char** lowered_name);
hiprtcResult hiprtcGetProgramLog(hiprtcProgram prog, char* log);
hiprtcResult hiprtcGetProgramLogSize(hiprtcProgram prog, size_t* logSizeRet);
hiprtcResult hiprtcGetCode(hiprtcProgram prog, char* code);
hiprtcResult hiprtcGetCodeSize(hiprtcProgram prog, size_t* codeSizeRet);

// Enums for corresponding #defines in the HIP headers.
enum hipDeviceFlags_t {
  hipDeviceScheduleAuto = 0x0,
  hipDeviceScheduleSpin = 0x1,
  hipDeviceScheduleYield = 0x2,
  hipDeviceScheduleBlockingSync = 0x4,
  hipDeviceMapHost = 0x8,
  hipDeviceLmemResizeToMax = 0x10,
};
enum hipStreamFlags_t {
  hipStreamDefault = 0x0,
  hipStreamNonBlocking = 0x1,
};
enum hipEventFlags_t {
  hipEventDefault = 0x0,
  hipEventBlockingSync = 0x1,
  hipEventDisableTiming = 0x2,
  hipEventInterprocess = 0x4,
};
enum hipHostMallocFlags_t {
  hipHostMallocDefault = 0x0,
  hipHostMallocPortable = 0x1,
  hipHostMallocMapped = 0x2,
  hipHostMallocWriteCombined = 0x4,
  hipHostMallocCoherent = 0x40000000,
  hipHostMallocNonCoherent = 0x80000000,
};
enum hipHostRegisterFlags_t {
  hipHostRegisterDefault = 0x0,
  hipHostRegisterMapped = 0x2,
  hipExtHostRegisterCoarseGrained = 0x8,
};
enum hipMemAttachFlags_t {
  hipMemAttachGlobal = 1,
  hipMemAttachHost = 2,
};

typedef struct {
  // 32-bit Atomics
  unsigned
      hasGlobalInt32Atomics : 1;  ///< 32-bit integer atomics for global memory.
  unsigned hasGlobalFloatAtomicExch : 1;  ///< 32-bit float atomic exch for
                                          ///< global memory.
  unsigned
      hasSharedInt32Atomics : 1;  ///< 32-bit integer atomics for shared memory.
  unsigned hasSharedFloatAtomicExch : 1;  ///< 32-bit float atomic exch for
                                          ///< shared memory.
  unsigned hasFloatAtomicAdd : 1;  ///< 32-bit float atomic add in global and
                                   ///< shared memory.

  // 64-bit Atomics
  unsigned
      hasGlobalInt64Atomics : 1;  ///< 64-bit integer atomics for global memory.
  unsigned
      hasSharedInt64Atomics : 1;  ///< 64-bit integer atomics for shared memory.

  // Doubles
  unsigned hasDoubles : 1;  ///< Double-precision floating point.

  // Warp cross-lane operations
  unsigned hasWarpVote : 1;     ///< Warp vote instructions (__any, __all).
  unsigned hasWarpBallot : 1;   ///< Warp ballot instructions (__ballot).
  unsigned hasWarpShuffle : 1;  ///< Warp shuffle operations. (__shfl_*).
  unsigned
      hasFunnelShift : 1;  ///< Funnel two words into one with shift&mask caps.

  // Sync
  unsigned hasThreadFenceSystem : 1;  ///< __threadfence_system.
  unsigned hasSyncThreadsExt : 1;     ///< __syncthreads_count, syncthreads_and,
                                      ///< syncthreads_or.

  // Misc
  unsigned hasSurfaceFuncs : 1;  ///< Surface functions.
  unsigned has3dGrid : 1;  ///< Grid and group dims are 3D (rather than 2D).
  unsigned hasDynamicParallelism : 1;  ///< Dynamic parallelism.
} hipDeviceArch_t;

typedef struct hipDeviceProp_t {
  char name[256];            ///< Device name.
  size_t totalGlobalMem;     ///< Size of global memory region (in bytes).
  size_t sharedMemPerBlock;  ///< Size of shared memory region (in bytes).
  int regsPerBlock;          ///< Registers per block.
  int warpSize;              ///< Warp size.
  int maxThreadsPerBlock;    ///< Max work items per work group or workgroup max
                             ///< size.
  int maxThreadsDim[3];  ///< Max number of threads in each dimension (XYZ) of a
                         ///< block.
  int maxGridSize[3];    ///< Max grid dimensions (XYZ).
  int clockRate;         ///< Max clock frequency of the multiProcessors in khz.
  int memoryClockRate;   ///< Max global memory clock frequency in khz.
  int memoryBusWidth;    ///< Global memory bus width in bits.
  size_t totalConstMem;  ///< Size of shared memory region (in bytes).
  int major;  ///< Major compute capability.  On HCC, this is an approximation
              ///< and features may differ from CUDA CC.  See the arch feature
              ///< flags for portable ways to query feature caps.
  int minor;  ///< Minor compute capability.  On HCC, this is an approximation
              ///< and features may differ from CUDA CC.  See the arch feature
              ///< flags for portable ways to query feature caps.
  int multiProcessorCount;  ///< Number of multi-processors (compute units).
  int l2CacheSize;          ///< L2 cache size.
  int maxThreadsPerMultiProcessor;  ///< Maximum resident threads per
                                    ///< multi-processor.
  int computeMode;                  ///< Compute mode.
  int clockInstructionRate;  ///< Frequency in khz of the timer used by the
                             ///< device-side "clock*" instructions.  New for
                             ///< HIP.
  hipDeviceArch_t arch;      ///< Architectural feature flags.  New for HIP.
  int concurrentKernels;     ///< Device can possibly execute multiple kernels
                             ///< concurrently.
  int pciDomainID;           ///< PCI Domain ID
  int pciBusID;              ///< PCI Bus ID.
  int pciDeviceID;           ///< PCI Device ID.
  size_t maxSharedMemoryPerMultiProcessor;  ///< Maximum Shared Memory Per
                                            ///< Multiprocessor.
  int isMultiGpuBoard;    ///< 1 if device is on a multi-GPU board, 0 if not.
  int canMapHostMemory;   ///< Check whether HIP can map host memory
  int gcnArch;            ///< DEPRECATED: use gcnArchName instead
  char gcnArchName[256];  ///< AMD GCN Arch Name.
  int integrated;         ///< APU vs dGPU
  int cooperativeLaunch;  ///< HIP device supports cooperative launch
  int cooperativeMultiDeviceLaunch;  ///< HIP device supports cooperative launch
                                     ///< on multiple devices
  int maxTexture1DLinear;  ///< Maximum size for 1D textures bound to linear
                           ///< memory
  int maxTexture1D;        ///< Maximum number of elements in 1D images
  int maxTexture2D[2];  ///< Maximum dimensions (width, height) of 2D images, in
                        ///< image elements
  int maxTexture3D[3];  ///< Maximum dimensions (width, height, depth) of 3D
                        ///< images, in image elements
  unsigned int*
      hdpMemFlushCntl;  ///< Addres of HDP_MEM_COHERENCY_FLUSH_CNTL register
  unsigned int*
      hdpRegFlushCntl;      ///< Addres of HDP_REG_COHERENCY_FLUSH_CNTL register
  size_t memPitch;          ///< Maximum pitch in bytes allowed by memory copies
  size_t textureAlignment;  ///< Alignment requirement for textures
  size_t texturePitchAlignment;  ///< Pitch alignment requirement for texture
                                 ///< references bound to pitched memory
  int kernelExecTimeoutEnabled;  ///< Run time limit for kernels executed on the
                                 ///< device
  int ECCEnabled;                ///< Device has ECC support enabled
  int tccDriver;  ///< 1:If device is Tesla device using TCC driver, else 0
  int cooperativeMultiDeviceUnmatchedFunc;  ///< HIP device supports cooperative
                                            ///< launch on multiple
                                            /// devices with unmatched functions
  int cooperativeMultiDeviceUnmatchedGridDim;   ///< HIP device supports
                                                ///< cooperative launch on
                                                ///< multiple
                                                /// devices with unmatched grid
                                                /// dimensions
  int cooperativeMultiDeviceUnmatchedBlockDim;  ///< HIP device supports
                                                ///< cooperative launch on
                                                ///< multiple
                                                /// devices with unmatched block
                                                /// dimensions
  int cooperativeMultiDeviceUnmatchedSharedMem;  ///< HIP device supports
                                                 ///< cooperative launch on
                                                 ///< multiple
                                                 /// devices with unmatched
                                                 /// shared memories
  int isLargeBar;     ///< 1: if it is a large PCI bar device, else 0
  int asicRevision;   ///< Revision of the GPU in this device
  int managedMemory;  ///< Device supports allocating managed memory on this
                      ///< system
  int directManagedMemAccessFromHost;  ///< Host can directly access managed
                                       ///< memory on the device without
                                       ///< migration
  int concurrentManagedAccess;  ///< Device can coherently access managed memory
                                ///< concurrently with the CPU
  int pageableMemoryAccess;  ///< Device supports coherently accessing pageable
                             ///< memory without calling hipHostRegister on it
  int pageableMemoryAccessUsesHostPageTables;  ///< Device accesses pageable
                                               ///< memory via the host's page
                                               ///< tables
} hipDeviceProp_t;

// Attribute structs declared in HIP headers.
struct hipPointerAttribute_t {
  hipMemoryType memoryType;
  int device;
  void* devicePointer;
  void* hostPointer;
  int isManaged;
  unsigned allocationFlags;
};
struct hipFuncAttributes {
  int binaryVersion;
  int cacheModeCA;        // always 0.
  size_t constSizeBytes;  // always 0.
  size_t localSizeBytes;
  int maxDynamicSharedSizeBytes;
  int maxThreadsPerBlock;
  int numRegs;
  int preferredShmemCarveout;  // always 0.
  int ptxVersion;              // always 30.
  size_t sharedSizeBytes;
};

#endif  // TFRT_GPU_WRAPPER_HIP_STUB_H_
