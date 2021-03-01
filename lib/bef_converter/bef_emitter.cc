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

//===- bef_emitter.cc -----------------------------------------------------===//
//
// This file defines the BEFEmitter that emits bytes into an aligned buffer.
//
//===----------------------------------------------------------------------===//

#include "tfrt/bef_converter/bef_emitter.h"

namespace tfrt {

const size_t BEFEmitter::kMaxAlignment;
const uint8_t BEFEmitter::kDummyByte;

void BEFEmitter::MoveResult(size_t dst_offset, size_t src_offset, size_t size) {
  memmove(result_.data() + dst_offset, result_.data() + src_offset, size);
}

void BEFEmitter::SetResult(size_t offset, uint8_t value, size_t size) {
  memset(result_.data() + offset, value, size);
}

// Our fundamental unit is a bytestream, but we want to be able to emit large
// values as well.  We use a VBR encoding, where the high bit set indicates
// that this is only a portion of the value.
void BEFEmitter::EmitIntImpl(size_t value, bool is_high_part) {
  if ((value >> 7) != 0) EmitIntImpl(value >> 7, /*is_high_part=*/true);

  result_.push_back(
      static_cast<uint8_t>((value & 127) | (is_high_part ? 128 : 0)));
}

void BEFEmitter::EmitRepeatedByte(uint8_t byte, int repeats) {
  for (int i = 0; i < repeats; ++i) result_.push_back(byte);
}

void BEFEmitter::EmitRepeatedDummyByte(int repeats) {
  EmitRepeatedByte(kDummyByte, repeats);
}

void BEFEmitter::EmitBytes(llvm::ArrayRef<uint8_t> bytes) {
  result_.insert(result_.end(), bytes.begin(), bytes.end());
}

// Emit a guaranteed 2-byte integer aligned to 2 bytes, allowing this to be
// directly mapped into the target process in little-endian form.
void BEFEmitter::EmitInt2(uint16_t value) {
  EmitAlignment(2);
  uint8_t data[] = {uint8_t(value & 0xFF), uint8_t((value >> 8) & 0xFF)};
  EmitBytes(data);
}

// Emit a guaranteed 4-byte integer aligned to 4 bytes, allowing this to be
// directly mapped into the target process in little-endian form.
void BEFEmitter::EmitInt4(uint32_t value) {
  EmitAlignment(4);
  uint8_t data[] = {uint8_t(value & 0xFF), uint8_t((value >> 8) & 0xFF),
                    uint8_t((value >> 16) & 0xFF),
                    uint8_t((value >> 24) & 0xFF)};
  EmitBytes(data);
}

// Emit a guaranteed 8-byte integer aligned to 8 bytes, allowing this to be
// directly mapped into the target process in little-endian form.
void BEFEmitter::EmitInt8(uint64_t value) {
  EmitAlignment(8);
  uint8_t data[] = {
      uint8_t(value & 0xFF),         uint8_t((value >> 8) & 0xFF),
      uint8_t((value >> 16) & 0xFF), uint8_t((value >> 24) & 0xFF),
      uint8_t((value >> 32) & 0xFF), uint8_t((value >> 40) & 0xFF),
      uint8_t((value >> 48) & 0xFF), uint8_t((value >> 56) & 0xFF)};
  EmitBytes(data);
}

void BEFEmitter::OverwriteBytes(size_t offset, const void* data, size_t size) {
  assert(offset + size <= result_.size());
  std::memcpy(&result_[offset], data, size);
}

void BEFEmitter::EmitEmitter(const BEFEmitter& emitter) {
  EmitAlignment(emitter.GetRequiredAlignment());
  EmitBytes(emitter.result_);
}

void BEFEmitter::EmitAlignment(unsigned alignment) {
  // Alignment of 0 and 1 is a noop.
  if (alignment < 2) return;

  assert(llvm::isPowerOf2_32(alignment));

  // We need attributes to have proper alignment in the file, so figure out
  // whether we need padding before this to make sure it ends up at the right
  // address.
  size_t cur_offset = size();
  size_t needed_padding = llvm::alignTo(cur_offset, alignment) - cur_offset;

  // Emit dummy padding bytes to get up to the right offset.
  while (needed_padding--) EmitByte(kDummyByte);

  // Keep track of the maximum required alignment.
  required_alignment_ = std::max(required_alignment_, alignment);
}

}  // namespace tfrt
