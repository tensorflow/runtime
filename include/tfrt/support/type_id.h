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

// This file defines a mapping from type to numeric id.

#ifndef TFRT_SUPPORT_TYPE_ID_H_
#define TFRT_SUPPORT_TYPE_ID_H_

#include <atomic>

namespace tfrt {

// Use this as DenseTypeId<some_type_specific_to_your_use>, that way you are
// guaranteed to get contiguous IDs starting at 0 unique to your particular
// use-case, as would be appropriate to use for indexes into a vector.
// 'some_type_specific_to_your_use' could (e.g.) be the class that contains
// that particular vector.
template <typename IdSet>
class DenseTypeId {
 public:
  template <typename T>
  static size_t get() {
    static const size_t id = next_id_.fetch_add(1, std::memory_order_relaxed);
    return id;
  }

 private:
  static std::atomic<size_t> next_id_;
};

template <typename IdSet>
std::atomic<size_t> DenseTypeId<IdSet>::next_id_;

}  // namespace tfrt

#endif  // TFRT_SUPPORT_TYPE_ID_H_
