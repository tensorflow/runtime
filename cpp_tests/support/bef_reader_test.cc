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

// Unit test for BEFReader

#include "tfrt/bef/bef_reader.h"

#include <numeric>

#include "gtest/gtest.h"
#include "tfrt/bef/bef_buffer.h"

namespace tfrt {
namespace {

constexpr size_t kBufferSize = 10;

class BefReaderTest : public ::testing::Test {
 protected:
  BefReaderTest() {
    // Prepare a 8-byte aligned buffer with 0 to 9 (size 10).
    aligned_buffer_.resize(kBufferSize);
    std::iota(aligned_buffer_.begin(), aligned_buffer_.end(), 0);
    array_ref_ = ArrayRef<uint8_t>(aligned_buffer_);
  }

  BefBuffer aligned_buffer_;
  ArrayRef<uint8_t> array_ref_;
};

TEST_F(BefReaderTest, Empty) {
  ArrayRef<uint8_t> ar = {};
  BEFReader reader(ar);
  EXPECT_TRUE(reader.Empty());
}

TEST_F(BefReaderTest, NotEmpty) {
  BEFReader reader(array_ref_);
  EXPECT_FALSE(reader.Empty());
}

TEST_F(BefReaderTest, GetFile) {
  BEFReader reader(array_ref_);
  const ArrayRef<uint8_t> reader_file = reader.file();
  EXPECT_EQ(array_ref_, reader_file);
}

TEST_F(BefReaderTest, SkipPast) {
  const ArrayRef<uint8_t> skip_section = array_ref_.slice(0, 7);
  BEFReader reader(array_ref_);
  reader.SkipPast(skip_section);
  const ArrayRef<uint8_t> reader_file = reader.file();
  EXPECT_EQ(ArrayRef<uint8_t>({7, 8, 9}), reader_file);
}

TEST_F(BefReaderTest, SkipOffset) {
  BEFReader reader(array_ref_);
  reader.SkipOffset(8);
  const ArrayRef<uint8_t> reader_file = reader.file();
  EXPECT_EQ(ArrayRef<uint8_t>({8, 9}), reader_file);
}

TEST_F(BefReaderTest, ReadByte) {
  BEFReader reader(array_ref_);
  uint8_t value;
  for (uint8_t i = 0; i < kBufferSize; ++i) {
    EXPECT_TRUE(reader.ReadByte(&value));
    EXPECT_EQ(i, value);
  }
  EXPECT_FALSE(reader.ReadByte(&value));
}

TEST_F(BefReaderTest, ReadIntFromEmptyFile) {
  ArrayRef<uint8_t> ar = {};
  BEFReader reader(ar);
  size_t value = 0;
  EXPECT_FALSE(reader.ReadVbrInt(&value));
}

// ReadVbrInt() uses Variable Byte Rate (VBR) encoding to read an integer.
// Only 7 LSB bits are used as interger payload
// while the MSB bit is used to indicate there is more byte.
// e.g.,
//   0x01           --> 1B integer, value = 0x01
//   0x81 0x02      --> 2B integer, value = (1 << 7) + 2 = 130
//   0x81 0x82 0x03 --> 3B integer, value = (1 << 14) + (2 << 7) + 3 = 16643
TEST_F(BefReaderTest, ReadIntOneByte) {
  // A sucessful case for an 1 byte integer.
  uint8_t file_content[] = {0x01};
  BEFReader reader{file_content};
  size_t value = 0;
  EXPECT_TRUE(reader.ReadVbrInt(&value));
  EXPECT_EQ(0x01, value);
  EXPECT_TRUE(reader.Empty());
}

TEST_F(BefReaderTest, ReadIntTwoBytes) {
  // A sucessful case for a 2 byte integer.
  uint8_t file_content[] = {0x81, 0x02};
  BEFReader reader{file_content};
  size_t value = 0;
  EXPECT_TRUE(reader.ReadVbrInt(&value));
  EXPECT_EQ(130, value);
  EXPECT_TRUE(reader.Empty());
}

TEST_F(BefReaderTest, ReadIntTwoBytesFailure) {
  // Should fail: two bytes are expected but only one byte exists.
  uint8_t file_content[] = {0x81};
  BEFReader reader{file_content};
  size_t value = 0;
  EXPECT_FALSE(reader.ReadVbrInt(&value));
  EXPECT_TRUE(reader.Empty());
}

TEST_F(BefReaderTest, ReadIntThreeBytes) {
  // A sucessful case for a 3 byte integer.
  uint8_t file_content[] = {0x81, 0x82, 0x03};
  BEFReader reader{file_content};
  size_t value = 0;
  EXPECT_TRUE(reader.ReadVbrInt(&value));
  EXPECT_EQ(16643, value);
}

TEST_F(BefReaderTest, ReadAlignmentAlreadyAligned) {
  BEFReader reader(array_ref_);

  // The buffer is already aligned.
  EXPECT_TRUE(reader.ReadAlignment(8));
  const ArrayRef<uint8_t> reader_file = reader.file();
  EXPECT_EQ(array_ref_, reader_file);
}

TEST_F(BefReaderTest, ReadAlignment) {
  BEFReader reader(array_ref_);

  // Skip one byte to break alignment(8).
  reader.SkipOffset(1);

  // Should skip 7 bytes for 8 byte alignment.
  EXPECT_TRUE(reader.ReadAlignment(8));

  const ArrayRef<uint8_t> reader_file = reader.file();
  EXPECT_EQ(ArrayRef<uint8_t>({8, 9}), reader_file);
}

TEST_F(BefReaderTest, ReadAlignmentNotEnoughBytes) {
  BEFReader reader(array_ref_);
  reader.SkipOffset(9);
  EXPECT_FALSE(reader.ReadAlignment(8));
}

TEST_F(BefReaderTest, ReadSectionEmptyFile) {
  ArrayRef<uint8_t> ar = {};
  BEFReader reader(ar);

  uint8_t section_id;
  ArrayRef<uint8_t> section;
  EXPECT_FALSE(reader.ReadSection(&section_id, &section));
}

TEST_F(BefReaderTest, ReadSectionNoLength) {
  // Should fail: only section_id exists.
  BEFReader reader(array_ref_.slice(0, 1));

  uint8_t section_id;
  ArrayRef<uint8_t> section;
  EXPECT_FALSE(reader.ReadSection(&section_id, &section));
}

TEST_F(BefReaderTest, ReadSectionNoAlignmentNotEnough) {
  // Should fail: length is set to 3, but remaining data are not enough.
  aligned_buffer_[1] = 3 << 1;
  BEFReader reader(array_ref_.slice(0, 4));

  uint8_t section_id;
  ArrayRef<uint8_t> section;
  EXPECT_FALSE(reader.ReadSection(&section_id, &section));
}

TEST_F(BefReaderTest, ReadSectionNoAlignment) {
  // Should succeed: no alignment, length = 3.
  aligned_buffer_[1] = 3 << 1;
  BEFReader reader(array_ref_);

  uint8_t section_id;
  ArrayRef<uint8_t> section;
  EXPECT_TRUE(reader.ReadSection(&section_id, &section));
  EXPECT_EQ(0, section_id);
  EXPECT_TRUE(section.equals({2, 3, 4}));
}

TEST_F(BefReaderTest, ReadSectionAlignmentNotExist) {
  // Should fail: alignment byte does not exist.
  aligned_buffer_[1] = (3 << 1) | 1;
  BEFReader reader(array_ref_.slice(0, 2));

  uint8_t section_id;
  ArrayRef<uint8_t> section;
  EXPECT_FALSE(reader.ReadSection(&section_id, &section));
}

TEST_F(BefReaderTest, ReadSectionWithInvalidAlignment) {
  // Should fail: invalid alignment value (3).
  aligned_buffer_[1] = (3 << 1) | 1;
  aligned_buffer_[2] = 3;
  BEFReader reader(array_ref_);

  uint8_t section_id;
  ArrayRef<uint8_t> section;
  EXPECT_FALSE(reader.ReadSection(&section_id, &section));
}

TEST_F(BefReaderTest, ReadSectionWithAlignmentNotEnough) {
  // Should fail: remaining data are not enough after the alignment.
  aligned_buffer_[1] = ((10 << 1) | 1);
  aligned_buffer_[2] = 8;
  BEFReader reader(array_ref_);
  uint8_t section_id;
  ArrayRef<uint8_t> section;
  EXPECT_FALSE(reader.ReadSection(&section_id, &section));
}

TEST_F(BefReaderTest, ReadSectionWithAlignment) {
  // Should succeed: length = 2, alignment = 8
  aligned_buffer_[1] = ((2 << 1) | 1);
  aligned_buffer_[2] = 8;
  BEFReader reader(array_ref_);

  uint8_t section_id;
  ArrayRef<uint8_t> section;

  EXPECT_TRUE(reader.ReadSection(&section_id, &section));
  EXPECT_EQ(0, section_id);
  EXPECT_TRUE(section.equals({8, 9}));
}

TEST_F(BefReaderTest, ReadSectionWithAlignmentSmallerSize) {
  // Should succeed: length = 1, alignment = 8
  aligned_buffer_[1] = ((1 << 1) | 1);
  aligned_buffer_[2] = 8;
  BEFReader reader(array_ref_);

  uint8_t section_id;
  ArrayRef<uint8_t> section;

  EXPECT_TRUE(reader.ReadSection(&section_id, &section));
  EXPECT_EQ(0, section_id);
  EXPECT_TRUE(section.equals({8}));
}

}  // namespace
}  // namespace tfrt
