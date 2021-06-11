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

#include "../../lib/bef_converter/mlir_to_bef/bef_location_emitter.h"

#include "gtest/gtest.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "tfrt/bef/bef_location.h"

namespace tfrt {
namespace {

class BefLocationEmitterTest : public ::testing::Test {
 protected:
  BefLocationEmitterTest() {}
  mlir::MLIRContext context_;
};

TEST_F(BefLocationEmitterTest, IsSupportedLocationFileLineColumnLoc) {
  auto loc = mlir::FileLineColLoc::get(&context_, "file", 1, 2);
  EXPECT_TRUE(BefLocationEmitter::IsSupportedLocation(loc));
}

TEST_F(BefLocationEmitterTest, IsSupportedLocationNamedLoc) {
  auto loc = mlir::NameLoc::get(mlir::Identifier::get("testloc", &context_));
  EXPECT_TRUE(BefLocationEmitter::IsSupportedLocation(loc));

  auto child = loc.getChildLoc();
  EXPECT_TRUE(child.isa<mlir::UnknownLoc>());
}

TEST_F(BefLocationEmitterTest, IsSupportedLocationCallSiteLoc) {
  mlir::Location callee_loc =
      mlir::FileLineColLoc::get(&context_, "callee", 1, 2);
  mlir::Location caller_loc =
      mlir::FileLineColLoc::get(&context_, "caller", 3, 4);
  auto loc = mlir::CallSiteLoc::get(callee_loc, caller_loc);
  EXPECT_TRUE(BefLocationEmitter::IsSupportedLocation(loc));
}

TEST_F(BefLocationEmitterTest, IsSupportedLocationOpaqueLoc) {
  auto loc = mlir::OpaqueLoc::get<uintptr_t>(9, &context_);
  EXPECT_FALSE(BefLocationEmitter::IsSupportedLocation(loc));
}

TEST_F(BefLocationEmitterTest, IsSupportedLocationUnknownLoc) {
  auto loc = mlir::UnknownLoc::get(&context_);
  EXPECT_TRUE(BefLocationEmitter::IsSupportedLocation(loc));
}

TEST_F(BefLocationEmitterTest, IsSupportedLocationBasicFusedLoc) {
  llvm::SmallVector<mlir::Location, 3> locations;
  locations.push_back(mlir::UnknownLoc::get(&context_));
  locations.push_back(mlir::OpaqueLoc::get<uintptr_t>(9, &context_));
  locations.push_back(mlir::FileLineColLoc::get(&context_, "file", 1, 2));
  auto loc = mlir::FusedLoc::get(&context_, locations);
  EXPECT_TRUE(BefLocationEmitter::IsSupportedLocation(loc));
}

TEST_F(BefLocationEmitterTest, NestedFusedLoc) {
  llvm::SmallVector<mlir::Location, 2> locations;
  locations.push_back(mlir::UnknownLoc::get(&context_));
  locations.push_back(
      mlir::NameLoc::get(mlir::Identifier::get("testloc", &context_)));

  llvm::SmallVector<mlir::Location, 2> nested_locations;
  nested_locations.push_back(mlir::OpaqueLoc::get<uintptr_t>(9, &context_));
  nested_locations.push_back(mlir::FusedLoc::get(&context_, locations));

  auto loc = mlir::FusedLoc::get(&context_, nested_locations);
  EXPECT_TRUE(BefLocationEmitter::IsSupportedLocation(loc));
}

TEST_F(BefLocationEmitterTest, EmitLocationFileLineColumnLoc) {
  auto loc = mlir::FileLineColLoc::get(&context_, "file", 1, 2);

  BefLocationEmitter emitter;
  const size_t offset = emitter.EmitLocation(loc);
  auto buffer = emitter.TakeResult();
  BefFileLineColLocation location(buffer.data() + offset);

  EXPECT_EQ(location.filename(emitter.GetStringsSection()), "file");
  EXPECT_EQ(location.line(), 1);
  EXPECT_EQ(location.column(), 2);
}

TEST_F(BefLocationEmitterTest, EmitLocationNameLoc) {
  auto loc = mlir::NameLoc::get(mlir::Identifier::get("testloc", &context_));

  BefLocationEmitter emitter;
  const size_t offset = emitter.EmitLocation(loc);

  auto buffer = emitter.TakeResult();
  BefNameLocation location(buffer.data() + offset);

  EXPECT_EQ(location.name(emitter.GetStringsSection()), "testloc");
}

TEST_F(BefLocationEmitterTest, EmitLocationCallSiteLoc) {
  mlir::Location callee_loc =
      mlir::NameLoc::get(mlir::Identifier::get("callee", &context_));
  mlir::Location caller_loc =
      mlir::NameLoc::get(mlir::Identifier::get("caller", &context_));
  auto loc = mlir::CallSiteLoc::get(callee_loc, caller_loc);

  BefLocationEmitter emitter;
  const size_t offset = emitter.EmitLocation(loc);

  auto buffer = emitter.TakeResult();
  BefCallSiteLocation location(buffer.data() + offset);

  auto callee = location.callee().dyn_cast<BefNameLocation>();
  auto caller = location.caller().dyn_cast<BefNameLocation>();

  EXPECT_EQ(callee.name(emitter.GetStringsSection()), "callee");
  EXPECT_EQ(caller.name(emitter.GetStringsSection()), "caller");
}

TEST_F(BefLocationEmitterTest, EmitLocationFusedLoc) {
  llvm::SmallVector<mlir::Location, 2> locations;
  locations.push_back(mlir::UnknownLoc::get(&context_));
  locations.push_back(
      mlir::NameLoc::get(mlir::Identifier::get("testloc", &context_)));
  locations.push_back(mlir::OpaqueLoc::get<uintptr_t>(9, &context_));
  locations.push_back(mlir::FileLineColLoc::get(&context_, "file", 111, 222));
  auto loc = mlir::FusedLoc::get(&context_, locations);

  BefLocationEmitter emitter;
  const size_t offset = emitter.EmitLocation(loc);
  auto buffer = emitter.TakeResult();
  BefFusedLocation location(buffer.data() + offset);
  EXPECT_EQ(location.size(), 2);

  auto first = location.GetLocation(0).dyn_cast<BefNameLocation>();
  EXPECT_EQ(first.name(emitter.GetStringsSection()), "testloc");

  auto second = location.GetLocation(1).dyn_cast<BefFileLineColLocation>();
  EXPECT_EQ(second.filename(emitter.GetStringsSection()), "file");
  EXPECT_EQ(second.line(), 111);
  EXPECT_EQ(second.column(), 222);
}

}  // namespace
}  // namespace tfrt
