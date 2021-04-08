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

// Sample kernels defined in tutorial.md, added here for testing.

#include <cstdio>

#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/kernel_registry.h"
#include "tfrt/host_context/kernel_utils.h"

namespace tfrt {
namespace {
struct Coordinate {
  int32_t x = 0;
  int32_t y = 0;
};

static Coordinate CreateCoordinate(int32_t x, int32_t y) {
  return Coordinate{x, y};
}

static Chain PrintCoordinate(Coordinate coordinate) {
  printf("(%d, %d)\n", coordinate.x, coordinate.y);
  fflush(stdout);
  return Chain();
}
}  // namespace

void RegisterTutorialKernels(KernelRegistry* registry) {
  registry->AddKernel("tfrt_tutorial.create_coordinate",
                      TFRT_KERNEL(CreateCoordinate));
  registry->AddKernel("tfrt_tutorial.print_coordinate",
                      TFRT_KERNEL(PrintCoordinate));
}

}  // namespace tfrt
