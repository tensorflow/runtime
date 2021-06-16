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

// Unit tests for TFRT Location and LocationHandler classes.

#include "tfrt/host_context/location.h"

#include "gtest/gtest.h"

namespace tfrt {
namespace {

constexpr intptr_t kMockData = 1234;
constexpr char kTestFileName[] = "foo.txt";
constexpr int kTestLine = 10;
constexpr int kTestColumn = 20;

class MockLocationHandler : public LocationHandler {
 public:
  MockLocationHandler() = default;
  DecodedLocation DecodeLocation(Location loc) const override {
    DecodedLocation decoded_location;
    if (loc.data == kMockData) {
      decoded_location =
          FileLineColLocation{kTestFileName, kTestLine, kTestColumn};
    }
    return decoded_location;
  }

  virtual ~MockLocationHandler() {}
};

TEST(LocationTest, EmptyLocation) {
  Location empty_location{};

  EXPECT_FALSE(static_cast<bool>(empty_location));

  DecodedLocation decoded_location = empty_location.Decode();
  FileLineColLocation loc = decoded_location.get<FileLineColLocation>();
  EXPECT_TRUE(loc.filename.empty());
  EXPECT_EQ(loc.line, -1);
  EXPECT_EQ(loc.column, -1);

  EXPECT_EQ(0, empty_location.data);
}

TEST(LocationTest, Data) {
  MockLocationHandler location_handler;
  Location location(&location_handler, kMockData);

  EXPECT_TRUE(static_cast<bool>(location));
  EXPECT_EQ(kMockData, location.data);
}

TEST(LocationTest, DecodeForKnownLocation) {
  MockLocationHandler location_handler;
  Location location(&location_handler, kMockData);

  DecodedLocation decoded_location = location.Decode();
  FileLineColLocation loc = decoded_location.get<FileLineColLocation>();
  EXPECT_EQ(loc.filename, kTestFileName);
  EXPECT_EQ(loc.line, kTestLine);
  EXPECT_EQ(loc.column, kTestColumn);
}

TEST(LocationTest, DecodeForUnknownLocation) {
  MockLocationHandler location_handler;
  Location location(&location_handler, 0);

  DecodedLocation decoded_location = location.Decode();
  FileLineColLocation loc = decoded_location.get<FileLineColLocation>();
  EXPECT_TRUE(loc.filename.empty());
  EXPECT_EQ(loc.line, -1);
  EXPECT_EQ(loc.column, -1);
}

}  // namespace
}  // namespace tfrt
