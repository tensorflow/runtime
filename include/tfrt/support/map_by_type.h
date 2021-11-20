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

// This file defines a container type that can store arbitrary data type keyed
// by the data type.

#ifndef TFRT_SUPPORT_MAP_BY_TYPE_H_
#define TFRT_SUPPORT_MAP_BY_TYPE_H_

#include <vector>

#include "llvm_derived/Support/unique_any.h"
#include "tfrt/support/type_id.h"

namespace tfrt {

/**
 * MapByType<IdSet> stores arbitrary data type keyed by the data type.
 *
 * Example usage:
 * struct MyMapTagType {};
 *
 * MapByType<MyMapTagType> map;
 *
 * map.insert(2);
 * assert(map.contains<int>());
 * assert(map.get<int>(), 2);
 *
 * When a data type is inserted more than once, the previous value is replaced.
 *
 * Example:
 * map.insert(2);
 * map.insert(3);
 * assert(map.get<int>(), 3);
 */
template <typename IdSet>
class MapByType {
  struct StorageBase {
    virtual ~StorageBase() = default;
  };

  template <typename ConcreteT>
  struct Storage : StorageBase {
    template <typename... Args>
    explicit Storage(Args&&... args) : value(std::forward<Args>(args)...) {}

    ConcreteT value;
  };

 public:
  template <typename T, typename... Args, typename VT = std::decay_t<T>>
  VT& emplace(Args&&... args) {
    auto id = getTypeId<VT>();
    if (id >= data_.size()) data_.resize(id + 1);

    data_[id] = std::make_unique<Storage<VT>>(std::forward<Args>(args)...);

    return cast<VT>(data_[id].get());
  }

  template <typename T>
  std::decay_t<T>& insert(T&& t) {
    return emplace<T>(std::forward<T>(t));
  }

  template <typename T>
  T& get() {
    return const_cast<T&>(static_cast<const MapByType*>(this)->get<T>());
  }

  template <typename T>
  const T& get() const {
    using VT = std::decay_t<T>;
    auto id = getTypeId<VT>();
    assert(id < data_.size());
    return cast<VT>(data_[id].get());
  }

  template <typename T>
  T* getIfExists() {
    return const_cast<T*>(
        static_cast<const MapByType*>(this)->getIfExists<T>());
  }

  template <typename T>
  const T* getIfExists() const {
    using VT = std::decay_t<T>;

    auto id = getTypeId<VT>();
    if (id >= data_.size()) return nullptr;

    auto& value = data_[id];
    if (value) return &cast<VT>(value.get());

    return nullptr;
  }

  template <typename T>
  bool contains() const {
    using VT = std::decay_t<T>;
    auto id = getTypeId<VT>();
    if (id >= data_.size()) return false;
    return data_[id] != nullptr;
  }

 private:
  template <typename T>
  static size_t getTypeId() {
    return DenseTypeId<IdSet>::template get<std::decay_t<T>>();
  }

  template <typename T>
  static T& cast(StorageBase* base) {
    return static_cast<Storage<T>*>(base)->value;
  }

  std::vector<std::unique_ptr<StorageBase>> data_;
};

}  // namespace tfrt

#endif  // TFRT_SUPPORT_MAP_BY_TYPE_H_
