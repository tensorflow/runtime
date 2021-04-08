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

// This file declares types for the 'tfrt_dist' dialect.

#ifndef TFRT_DISTRIBUTED_RUNTIME_OPDEFS_TYPES_H_
#define TFRT_DISTRIBUTED_RUNTIME_OPDEFS_TYPES_H_

#include "mlir/IR/Types.h"

namespace tfrt {
namespace dist {

class DistributedContextType
    : public mlir::Type::TypeBase<DistributedContextType, mlir::Type,
                                  mlir::TypeStorage> {
 public:
  using Base::Base;
};

class DistributedContextConfigurationType
    : public mlir::Type::TypeBase<DistributedContextConfigurationType,
                                  mlir::Type, mlir::TypeStorage> {
 public:
  using Base::Base;
};

class TaskHandleType : public mlir::Type::TypeBase<TaskHandleType, mlir::Type,
                                                   mlir::TypeStorage> {
 public:
  using Base::Base;
};

class RemoteObjectIdType
    : public mlir::Type::TypeBase<RemoteObjectIdType, mlir::Type,
                                  mlir::TypeStorage> {
 public:
  using Base::Base;
};

class RemoteExecuteSpecType
    : public mlir::Type::TypeBase<RemoteExecuteSpecType, mlir::Type,
                                  mlir::TypeStorage> {
 public:
  using Base::Base;
};

class RemoteChainManagerType
    : public mlir::Type::TypeBase<RemoteChainManagerType, mlir::Type,
                                  mlir::TypeStorage> {
 public:
  using Base::Base;
};

class PayloadType
    : public mlir::Type::TypeBase<PayloadType, mlir::Type, mlir::TypeStorage> {
 public:
  using Base::Base;
};

}  // namespace dist
}  // namespace tfrt

#endif  // TFRT_DISTRIBUTED_RUNTIME_OPDEFS_TYPES_H_
