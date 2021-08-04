// Copyright 2021 The TensorFlow Runtime Authors
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

#include "tfrt/support/string_util.h"

#include <ostream>

#include "gtest/gtest.h"

namespace tfrt {
namespace {

TEST(StringUtilTest, HumanReadableNum) {
  EXPECT_EQ(HumanReadableNum(0), "0");
  EXPECT_EQ(HumanReadableNum(123), "123");
  EXPECT_EQ(HumanReadableNum(12345), "12.35k");
  EXPECT_EQ(HumanReadableNum(12345678), "12.35M");
  EXPECT_EQ(HumanReadableNum(1234567890), "1.23B");
  EXPECT_EQ(HumanReadableNum(123456789012), "123.46B");
  EXPECT_EQ(HumanReadableNum(1234567890123), "1.23T");
  EXPECT_EQ(HumanReadableNum(1234E15), "1.23E+18");
}

TEST(StringUtilTest, NegativeHumanReadableNum) {
  EXPECT_EQ(HumanReadableNum(-123), "-123");
  EXPECT_EQ(HumanReadableNum(-12345), "-12.35k");
  EXPECT_EQ(HumanReadableNum(-12345678), "-12.35M");
  EXPECT_EQ(HumanReadableNum(-1234567890), "-1.23B");
  EXPECT_EQ(HumanReadableNum(-123456789012), "-123.46B");
  EXPECT_EQ(HumanReadableNum(-1234567890123), "-1.23T");
  EXPECT_EQ(HumanReadableNum(-1234E15), "-1.23E+18");
}

TEST(StringUtilTest, HumanReadableNumBytes) {
  EXPECT_EQ(HumanReadableNumBytes(0), "0");
  EXPECT_EQ(HumanReadableNumBytes(123), "123");
  EXPECT_EQ(HumanReadableNumBytes(12345), "12.1KiB");
  EXPECT_EQ(HumanReadableNumBytes(12345678), "11.77MiB");
  EXPECT_EQ(HumanReadableNumBytes(1234567890), "1.15GiB");
  EXPECT_EQ(HumanReadableNumBytes(123456789012), "114.98GiB");
  EXPECT_EQ(HumanReadableNumBytes(1234567890123), "1.12TiB");
  EXPECT_EQ(HumanReadableNumBytes(12345678901234567), "10.96PiB");
  EXPECT_EQ(HumanReadableNumBytes(1234567890123456789), "1.07EiB");
}

TEST(StringUtilTest, NegativeHumanReadableNumBytes) {
  EXPECT_EQ(HumanReadableNumBytes(-123), "-123");
  EXPECT_EQ(HumanReadableNumBytes(-12345), "-12.1KiB");
  EXPECT_EQ(HumanReadableNumBytes(-12345678), "-11.77MiB");
  EXPECT_EQ(HumanReadableNumBytes(-1234567890), "-1.15GiB");
  EXPECT_EQ(HumanReadableNumBytes(-123456789012), "-114.98GiB");
  EXPECT_EQ(HumanReadableNumBytes(-1234567890123), "-1.12TiB");
  EXPECT_EQ(HumanReadableNumBytes(-12345678901234567), "-10.96PiB");
  EXPECT_EQ(HumanReadableNumBytes(-1234567890123456789), "-1.07EiB");
}

TEST(StringUtilTest, HumanReadableElapsedTime) {
  EXPECT_EQ(HumanReadableElapsedTime(0.000456), "456 us");
  EXPECT_EQ(HumanReadableElapsedTime(0.123456), "123 ms");
  EXPECT_EQ(HumanReadableElapsedTime(0), "0 s");
  EXPECT_EQ(HumanReadableElapsedTime(123), "2.05 min");
  EXPECT_EQ(HumanReadableElapsedTime(12345), "3.43 h");
  EXPECT_EQ(HumanReadableElapsedTime(12345678), "4.69 months");
  EXPECT_EQ(HumanReadableElapsedTime(1234567890), "39.1 years");
}

TEST(StringUtilTest, NegativeHumanReadableElapsedTime) {
  EXPECT_EQ(HumanReadableElapsedTime(-0.000456), "-456 us");
  EXPECT_EQ(HumanReadableElapsedTime(-0.123456), "-123 ms");
  EXPECT_EQ(HumanReadableElapsedTime(-123), "-2.05 min");
  EXPECT_EQ(HumanReadableElapsedTime(-12345), "-3.43 h");
  EXPECT_EQ(HumanReadableElapsedTime(-12345678), "-4.69 months");
  EXPECT_EQ(HumanReadableElapsedTime(-1234567890), "-39.1 years");
}

struct Foo {
  friend std::ostream& operator<<(std::ostream& os, const Foo&) {
    return os << "foo";
  }
};

TEST(StringUtilTest, OStreamStrCat) {
  Foo foo;
  EXPECT_EQ(OstreamStrCat(foo, foo), "foofoo");
}

}  // namespace
}  // namespace tfrt
