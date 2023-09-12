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

#ifdef CORERT_BASE
#else
#define CORERT_BASE

def CoreRT_Dialect : Dialect {
  let name = "corert";

  let description = [{
    The Core Runtime Dialect.

    The Core Runtime Dialect is a hook to use Core Runtime op-by-op execution
    in graph mode. It provides corresponding Core Runtime types in MLIR and a
    corert.executeop kernel to invoke op-by-op execution.

    For more details, please refer to documents/tfrt_op_by_op_execution_design.md.
  }];

  let cppNamespace = "::tfrt::corert";
}

// Deivce is an aggregate of Core Runtime ops. It is passed to corert.executeop
// so that the required Core Runtime can be looked up in this device.
def CoreRT_OpHandlerType :
    Type<CPred<"$_self.isa<tfrt::corert::OpHandlerType>()">, "!corert.ophandler type">,
    BuildableType<"$_builder.getType<tfrt::corert::OpHandlerType>()">;

// TensorHandle is the currency type for Core Runtime execution.
def CoreRT_TensorHandleType:
    Type<CPred<"$_self.isa<tfrt::corert::TensorHandleType>()">, "!corert.tensorhandle type">,
    BuildableType<"$_builder.getType<tfrt::corert::TensorHandleType>()">;

#endif
