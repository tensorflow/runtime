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

// This file declares types for the 'data' dialect.

#ifndef TFRT_DATA_OPDEFS_TYPES_H_
#define TFRT_DATA_OPDEFS_TYPES_H_

#include "mlir/IR/Types.h"

namespace tfrt {
namespace data {

class DatasetType
    : public mlir::Type::TypeBase<DatasetType, mlir::Type, mlir::TypeStorage> {
 public:
  using Base::Base;
};

class IteratorType
    : public mlir::Type::TypeBase<IteratorType, mlir::Type, mlir::TypeStorage> {
 public:
  using Base::Base;
};

}  // namespace data

}  // namespace tfrt

#endif  // TFRT_DATA_OPDEFS_TYPES_H_
