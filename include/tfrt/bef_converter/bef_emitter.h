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

//===- bef_emitter.h - BEFEmitter -------------------------------*- C++ -*-===//
//
// This file defines the BEFEmitter that emits bytes into an aligned buffer.
//
//===----------------------------------------------------------------------===//

#ifndef TFRT_BEF_CONVERTER_BEF_EMITTER_H_
#define TFRT_BEF_CONVERTER_BEF_EMITTER_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/MathExtras.h"
#include "tfrt/support/aligned_buffer.h"

namespace tfrt {

class BEFEmitter {
 public:
  // TODO(zhangqiaorjc): Consider making alignment a template parameter.
  static const size_t kMaxAlignment = 8;

  BEFEmitter() {}
  BEFEmitter(const BEFEmitter&) = delete;
  BEFEmitter& operator=(const BEFEmitter&) = delete;

  // Return the alignment required by this chunk of a BEF file.
  unsigned GetRequiredAlignment() const { return required_alignment_; }

  size_t size() const { return result_.size(); }

  void EmitByte(uint8_t byte) { result_.push_back(byte); }
  void EmitDummyByte() { result_.push_back(kDummyByte); }
  void EmitRepeatedByte(uint8_t byte, int repeats);
  void EmitRepeatedDummyByte(int repeats);
  void EmitBytes(llvm::ArrayRef<uint8_t> bytes);

  void OverwriteBytes(size_t offset, const void* data, size_t size);

  void EmitAlignment(unsigned alignment);

  // Emit a guaranteed 2-byte integer aligned to 2 bytes, allowing this to be
  // directly mapped into the target process in little-endian form.
  void EmitInt2(uint16_t value);

  // Emit a guaranteed 4-byte integer aligned to 4 bytes, allowing this to be
  // directly mapped into the target process in little-endian form.
  void EmitInt4(uint32_t value);

  // Emit a guaranteed 8-byte integer aligned to 8 bytes, allowing this to be
  // directly mapped into the target process in little-endian form.
  void EmitInt8(uint64_t value);

  // Emit a vbr encoded integer of arbitrary width.
  void EmitInt(size_t value) { EmitIntImpl(value, false); }

  // Many parts of the emitter logic includes forward references into stuff
  // that hasn't been emitted and has variable size.  This is handled by making
  // nested emitters.  This helper function emits the subpieces once they are
  // constructed, ensuring that alignment requirements of the nested emitter
  // are maintained correctly.
  void EmitEmitter(const BEFEmitter& emitter);

  tfrt::AlignedBuffer<kMaxAlignment> TakeResult() { return std::move(result_); }

  // Move size bytes in the result from src_offset to dst_offset.
  void MoveResult(size_t dst_offset, size_t src_offset, size_t size);

  // Set size bytes in the result from offset to value
  void SetResult(size_t offset, uint8_t value, size_t size);

 protected:
  static const uint8_t kDummyByte = 0xCC;

  void EmitIntImpl(size_t value, bool is_high_part);
  // Keep track of the alignment required for the start of this object.
  unsigned required_alignment_ = 1;
  tfrt::AlignedBuffer<kMaxAlignment> result_;
};

}  // namespace tfrt

#endif
