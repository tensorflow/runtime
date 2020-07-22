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

//===- test_kernels.h - Interface to test_kernels library -------*- C++ -*-===//
//
// This declares the interfaces to register the kernels used by the
// host-executor testing tool.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_TEST_KERNELS_H_
#define TFRT_TEST_KERNELS_H_

namespace tfrt {
class KernelRegistry;
class NativeFunctionRegistry;

// Install some simple sync kernels and types for use by the test driver.
void RegisterSimpleKernels(KernelRegistry* registry);

// Install some async kernels and types for use by the test driver.
void RegisterAsyncKernels(KernelRegistry* registry);

// Install benchmark kernels.
void RegisterBenchmarkKernels(KernelRegistry* registry);
void RegisterSyncBenchmarkKernels(KernelRegistry* registry);

// Install some atomic test kernels for use by the test driver.
void RegisterAtomicTestKernels(KernelRegistry* registry);

// Install some async test kernels for use by the test driver.
void RegisterAsyncTestKernels(KernelRegistry* registry);

// Install some simple test kernels for use by the test driver.
void RegisterSimpleTestKernels(KernelRegistry* registry);

// Install some test native functions.
void RegisterTestNativeFunctions(NativeFunctionRegistry* registry);

// Install some kernels defined in tutorial.md.
void RegisterTutorialKernels(KernelRegistry* registry);

}  // namespace tfrt

#endif  // TFRT_TEST_KERNELS_H_
