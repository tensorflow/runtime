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

//===- compiler_pass.cc - Compiler Pass ------------------*- C++ -*--------===//
//
// This file contains implementation of GetCompilerPass and
// RegisterCompilerPass.
#include "tfrt/compiler/compiler_pass.h"

namespace tfrt {

static llvm::StringMap<CompilerPass*>& GetStaticCompilerPasses() {
  static llvm::StringMap<CompilerPass*>* compiler_passes =
      new llvm::StringMap<CompilerPass*>();
  return *compiler_passes;
}

void RegisterCompilerPass(const std::string& name, CompilerPass* pass) {
  llvm::StringMap<CompilerPass*>& compiler_passes = GetStaticCompilerPasses();
  compiler_passes[name] = pass;
}
const CompilerPass* GetCompilerPass(const std::string& name) {
  llvm::StringMap<CompilerPass*>& compiler_passes = GetStaticCompilerPasses();
  auto iter = compiler_passes.find(name);
  if (iter != compiler_passes.end()) {
    return iter->second;
  }
  return nullptr;
}
}  // namespace tfrt