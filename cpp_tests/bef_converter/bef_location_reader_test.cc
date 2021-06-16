/*
 * Copyright 2021 The TensorFlow Runtime Authors
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

// Unit tests for BefLocationReader class.

#include "../../lib/bef_converter/bef_to_mlir/bef_location_reader.h"

#include <memory>

#include "../../lib/bef_converter/mlir_to_bef/bef_location_emitter.h"
#include "gtest/gtest.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"

namespace tfrt {
namespace {

class BefLocationReaderTest : public ::testing::Test {
 protected:
  BefLocationReaderTest() {}

  mlir::MLIRContext context_;
};

TEST_F(BefLocationReaderTest, FileLineColLoc) {
  auto loc = mlir::FileLineColLoc::get(&context_, "file", 123, 456);

  BefLocationEmitter emitter;
  const size_t offset = emitter.EmitLocation(loc);

  auto reader = BefLocationReader(emitter.GetStringsSectionEmitter().result(),
                                  emitter.result(), &context_);
  auto read_loc = reader.ReadLocation(offset).dyn_cast<mlir::FileLineColLoc>();
  EXPECT_EQ(read_loc, loc);
}

TEST_F(BefLocationReaderTest, NameLoc) {
  auto child = mlir::FileLineColLoc::get(&context_, "file", 123, 456);
  auto loc =
      mlir::NameLoc::get(mlir::Identifier::get("testloc", &context_), child);

  BefLocationEmitter emitter;
  const size_t offset = emitter.EmitLocation(loc);

  auto reader = BefLocationReader(emitter.GetStringsSectionEmitter().result(),
                                  emitter.result(), &context_);
  auto read_loc = reader.ReadLocation(offset).dyn_cast<mlir::NameLoc>();
  EXPECT_EQ(read_loc, loc);
}

TEST_F(BefLocationReaderTest, CallSiteLoc) {
  auto callee = mlir::FileLineColLoc::get(&context_, "file1", 11, 22);
  auto caller = mlir::FileLineColLoc::get(&context_, "file2", 33, 44);
  auto loc = mlir::CallSiteLoc::get(callee, caller);

  BefLocationEmitter emitter;
  const size_t offset = emitter.EmitLocation(loc);

  auto reader = BefLocationReader(emitter.GetStringsSectionEmitter().result(),
                                  emitter.result(), &context_);
  auto read_loc = reader.ReadLocation(offset).dyn_cast<mlir::CallSiteLoc>();
  EXPECT_EQ(read_loc, loc);
}

TEST_F(BefLocationReaderTest, FusedLoc) {
  llvm::SmallVector<mlir::Location, 2> locations;
  locations.push_back(
      mlir::NameLoc::get(mlir::Identifier::get("testloc", &context_)));
  locations.push_back(mlir::FileLineColLoc::get(&context_, "file", 111, 222));
  auto loc = mlir::FusedLoc::get(&context_, locations);

  BefLocationEmitter emitter;
  const size_t offset = emitter.EmitLocation(loc);

  auto reader = BefLocationReader(emitter.GetStringsSectionEmitter().result(),
                                  emitter.result(), &context_);
  auto read_loc = reader.ReadLocation(offset).dyn_cast<mlir::FusedLoc>();
  EXPECT_EQ(read_loc, loc);
}

}  // namespace
}  // namespace tfrt
