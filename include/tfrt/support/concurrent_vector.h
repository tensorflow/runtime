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

//===- concurrent_vector.h --------------------------------------*- C++ -*-===//
//
// A concurent sequential container optimized for read access.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_SUPPORT_CONCURRENT_VECTOR_H_
#define TFRT_SUPPORT_CONCURRENT_VECTOR_H_

#include <atomic>
#include <cassert>
#include <memory>
#include <mutex>
#include <vector>

namespace tfrt {

// A simple concurrent sequential container that allows concurrent reads and
// writes and is optimized for read access. It is designed for the usage pattern
// where objects are inserted once but are read many times. The key difference
// between this data structure and std::vector is that when we re-allocate the
// underlying buffer, we do not free the previous buffer. This allows us to
// implement read access with a single atomic load.
//
// Sample usage:
//
// ConcurrentVector<T> vec;
//
// On the writer side, concurrent writers are allowed;
//
// size_t index1 = vec.emplace_back(args);
// size_t index2 = vec.emplace_back(args);
//
// On the reader side, concurrent readers are allowed.
//
// auto& t1 = vec[index1];
// auto& t2 = vec[index1];
//
// Requirements:
//
// Type T needs to be copyable.

template <typename T>
class ConcurrentVector {
 public:
  // Initialize the vector with the given initial_capapcity
  explicit ConcurrentVector(size_t initial_capacity) {
    all_allocated_elements_.emplace_back();
    auto& v = all_allocated_elements_.back();
    v.reserve(initial_capacity);
    elements_ = v.data();
  }

  T& operator[](size_t index) {
    assert(index < size() && "invalid ConcurrentVector index");
    // This acquire fence synchronizes with the release fence in emplace_back to
    // ensure the reader sees consistent data.
    T* elements = elements_.load(std::memory_order_relaxed);

    std::atomic_thread_fence(std::memory_order_acquire);

    return elements[index];
  }

  const T& operator[](size_t index) const {
    assert(index < size() && "invalid ConcurrentVector index");
    T* elements = elements_.load(std::memory_order_relaxed);

    // This acquire fence synchronizes with the release fence in emplace_back to
    // ensure the reader sees consistent data.
    std::atomic_thread_fence(std::memory_order_acquire);

    return elements[index];
  }

  // Return the number of elements currently valid in this vector.  The vector
  // only grows, so this is conservative w.r.t. the execution of the current
  // thread.
  size_t size() const { return num_elements_; }

  // Insert a new element at the end. If the current buffer is full, we allocate
  // a new buffer with twice as much capacity and copy the items in the
  // previous buffer over.
  //
  // Returns the index of the newly inserted item.
  template <typename... Args>
  size_t emplace_back(Args&&... args) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto& last = all_allocated_elements_.back();

    if (last.size() < last.capacity()) {
      // There is still room in the current vector without reallocation. Just
      // add the new element there.
      last.emplace_back(std::forward<Args>(args)...);

      // Increment the number of elements.
      num_elements_.fetch_add(1);

      // This release fence synchronizes with the acquire fence in operator[] to
      // ensure the reader sees consistent data.
      //
      // We assume that the index returned here will be propagated to the
      // reading thread using a relaxed atomic store/load or stronger.  This
      // relaxed atomic load/store pair along with the the release/acquire fence
      // will establish a synchronizes with relationship.
      std::atomic_thread_fence(std::memory_order_release);

      return last.size() - 1;
    }

    // There is no more room in the current vector without reallocation.
    // Allocate a new vector with twice as much capacity, copy the elements
    // from the previous vector, and set elements_ to point to the data of the
    // new vector.
    all_allocated_elements_.emplace_back();
    auto& new_last = all_allocated_elements_.back();
    auto& prev = *(all_allocated_elements_.rbegin() + 1);
    new_last.reserve(prev.capacity() * 2);
    assert(prev.size() == prev.capacity());

    // Copy over the previous vector to the new vector.
    new_last.insert(new_last.begin(), prev.begin(), prev.end());

    new_last.emplace_back(std::forward<Args>(args)...);

    // This release fence synchronizes with the acquire fence in operator[] to
    // ensure the reader sees consistent data.
    //
    // The release fence should be before elements_.store() line below to ensure
    // that if the reader sees the new value for elements_, they also see the
    // store operations for the data.
    std::atomic_thread_fence(std::memory_order_release);

    elements_.store(new_last.data(), std::memory_order_relaxed);

    // Increment the number of elements.
    num_elements_.fetch_add(1);

    return new_last.size() - 1;
  }

 private:
  // pointing to the data of the last vector allocated.
  std::atomic<T*> elements_{nullptr};

  // Return the current number of valid elements.
  std::atomic<size_t> num_elements_{0};

  std::mutex mutex_;
  std::vector<std::vector<T>> all_allocated_elements_;
};

}  // namespace tfrt
#endif  // TFRT_SUPPORT_CONCURRENT_VECTOR_H_
