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

#ifndef TFRT_BACKENDS_JITRT_INCLUDE_TFRT_JITRT_JITRT_COMPILER_H_
#define TFRT_BACKENDS_JITRT_INCLUDE_TFRT_JITRT_JITRT_COMPILER_H_

namespace mlir {
class DialectRegistry;
class OpPassManager;
}  // namespace mlir

namespace tfrt {
namespace jitrt {

// Registers dialects, interfaces and dialects translations with the registry
// required by the default JitRt compilation pipeline.
void RegisterDefaultJitRtDialects(mlir::DialectRegistry& registry);

struct CompilationPipelineOptions {
  // Byte alignment for allocated memrefs. Depending on the compiler flags
  // Tensorflow requires tensors to be aligned on 16, 32 or 64 bytes.
  int alignment = 0;

  // The number of worker threads (host context concurrent work queue size) that
  // can be used for parallelizing compute intensive parts of the kernel.
  int num_worker_threads = 0;

  // Use experimental cost model for lowering scf.parallel to async dialect.
  bool cost_driven_async_parallel_for = false;

  // Enables math approximations that emit AVX2 intrinsics.
#ifdef __AVX2__
  bool math_avx2 = true;
#else
  bool math_avx2 = false;
#endif
};

// Creates the default JitRt compilation pipeline that lowers from the Linalg
// on buffers to the LLVM dialect. This is a very simple pipeline that is mostly
// intended for writing tests for the JitRt inside the TFRT project, and it is
// expected that all end users will construct their own compilation pipelines
// from the available JitRt and MLIR passes.
//
// This reference pipeline is integrated with the JitRt async runtime, and
// allows the execution of `scf.parallel` operations using the runtime-provided
// concurrent work queue.
//
// Input program requirements:
//  - only dialects available in upstream MLIR are supported
//  - program must be bufferized: inputs and results must be memrefs, no tensors
//    in the function body are allowed
void CreateDefaultJitRtCompilationPipeline(
    mlir::OpPassManager& pm, const CompilationPipelineOptions& opts);

}  // namespace jitrt
}  // namespace tfrt

#endif  // TFRT_BACKENDS_JITRT_INCLUDE_TFRT_JITRT_JITRT_COMPILER_H_
