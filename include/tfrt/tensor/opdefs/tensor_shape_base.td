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

//===- tensor_shape_base.td -----------------------------------------------===//
//
// Base type definitions for tensor shape ops.
//
//===----------------------------------------------------------------------===//

#ifdef TS_OPS_BASE
#else
#define TS_OPS_BASE

// Tensor shape dialect.
def TensorShapeDialect : Dialect {
  let name = "ts";

  let description = [{
    The Tensor Shape dialect.

    This dialect contains operations for working with tensor shapes.
  }];

  let cppNamespace = "::tfrt::ts";

  let useDefaultTypePrinterParser = 1;
}

// Type definition.
def TS_Shape : DialectType<TensorShapeDialect,
    CPred<"$_self.isa<::tfrt::ts::ShapeType>()">, "!ts.shape type">,
    BuildableType<"$_builder.getType<::tfrt::ts::ShapeType>()"> {
  let description = [{
    `!ts.shape type` represents a static tensor shape.
  }];
}

def TS_PartialShape : DialectType<TensorShapeDialect,
    CPred<"$_self.isa<::tfrt::ts::PartialShapeType>()">, "!ts.partial_shape type">,
    BuildableType<"$_builder.getType<::tfrt::ts::PartialShapeType>()"> {
  let description = [{
    `!ts.partial_shape type` represents either a static tensor shape, unranked
    tensor shape or a ranked tensor shape with unknown dimension sizes.
  }];
}

#endif  // TS_OPS_BASE
