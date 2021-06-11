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

// Read a location from a BEF locations section.

#ifndef TFRT_LIB_BEF_CONVERTER_BEF_TO_MLIR_BEF_LOCATION_READER_H_
#define TFRT_LIB_BEF_CONVERTER_BEF_TO_MLIR_BEF_LOCATION_READER_H_

#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "tfrt/bef/bef_location.h"

namespace tfrt {

// This class converts a BEF location to an mlir::Location.
class BefLocationReader {
 public:
  BefLocationReader(ArrayRef<uint8_t> location_strings,
                    ArrayRef<uint8_t> locations, mlir::MLIRContext* context)
      : location_strings_(location_strings),
        locations_(locations),
        context_(*context) {}

  // Read a location from the given offset.
  mlir::Location ReadLocation(size_t offset);

 private:
  mlir::Location ReadLocation(BefLocation loc);

  ArrayRef<uint8_t> location_strings_;
  ArrayRef<uint8_t> locations_;
  mlir::MLIRContext& context_;
};

}  // namespace tfrt

#endif  // TFRT_LIB_BEF_CONVERTER_BEF_TO_MLIR_BEF_LOCATION_READER_H_
