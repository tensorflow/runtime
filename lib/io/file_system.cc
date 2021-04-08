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

// This file implements the FileSystemRegistry class.

#include "tfrt/io/file_system.h"

namespace tfrt {
namespace io {

void FileSystemRegistry::Register(const std::string& scheme,
                                  std::unique_ptr<FileSystem> file_system) {
  assert(file_system);
  mutex_lock lock(mu_);
  auto& value = file_systems_[scheme];
  if (!value || value->GetPriority() < file_system->GetPriority()) {
    value = std::move(file_system);
  }
}

FileSystem* FileSystemRegistry::Lookup(const std::string& scheme) {
  mutex_lock lock(mu_);
  return file_systems_[scheme].get();
}

}  // namespace io
}  // namespace tfrt
