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

//===- kernels.td ---------------------------------------------------------===//
//
// Operation definitions for distributed ops.
//
//===----------------------------------------------------------------------===//

#ifdef DISTRIBUTED_OPS
#else
#define DISTRIBUTED_OPS

include "mlir/IR/OpBase.td"
include "tfrt/tfrt_op_base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Distributed dialect.
def DistDialect : Dialect {
  let name = "dist";

  let description = [{The Distributed dialect.
                     This dialect contains distributed execution operations.}];

  let cppNamespace = "dist";
}

def DistributedContextType
    : OpaqueType<"dist", "dist_context", "!dist.dist_context type">;

def DistributedContextConfigurationType
    : OpaqueType<"dist", "dist_context_configuration",
                 "!dist.dist_context_configuration type">;

def CollectiveGroupType
    : OpaqueType<"dist", "collective_group", "!dist.collective_group type">;

def RemoteObjectIdType
    : OpaqueType<"dist", "remote_object_id", "!dist.remote_object_id type">;

// Base class for the operation in this dialect
class DistOp<string mnemonic, list<OpTrait> traits = []>
    : Op<DistDialect, mnemonic, !listconcat(traits, [IsolatedFromAbove])> {
}

// TODO(ayushd, xldrx): consider changing reduction_fn from attribute to a
// part of the kernel name.
class AllReduceOp<string dtype> : DistOp<"cpu.allreduce." # dtype> {
  let summary = "dist.cpu.allreduce operation";

  let description = [{
    An operation to perform an allreduce primitive on members of a communicator.
    When done,the out_chain is set. Collective on different devices share an
    identical instance_id. A collective takes a communicator, instance id as
    string, a tensor as the input, a tensor to hold the result, and a chain and
    outputs a chain. It also takes the reduction functions as an attribute,
    "reduction_fn". Currently the following functions are supported: "sum".

    Example:
      %out_chain = dist.cpu.allreduce.f32 %context, %collective_group, %instance_id, %tensor_in, %tensor_out, %in_chain {reduction_fn="sum"};
  }];

  let arguments = (ins
    DistributedContextType,
    CollectiveGroupType,
    StringType,
    TensorType,
    TensorType,
    TFRT_ChainType);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = "operands attr-dict";
}

foreach dtype = ["i32", "f32"] in {
  def Dist_AllReduceOp_#dtype : AllReduceOp<dtype>;
}

class BroadcastOp<string dtype> : DistOp<"cpu.broadcast." # dtype> {
  let summary = "dist.cpu.broadcast operation";

  let description = [{
    An operation to perform a broadcast primitive from one member to all other
    members of a communicator. This collective takes a communicator, an instance
    id (identical on all devices), an input, and a sender id, and outputs a
    chain. The sender places data to be broadcasted in the input. The receivers
    prepare an empty buffer in the input.

    Example:
      %out_chain = dist.cpu.broadcast.i32 %context, %instance_id, %input, %sender_id, %in_chain
  }];

  let arguments = (ins
    DistributedContextType,
    CollectiveGroupType,
    StringType,
    TensorType,
    I32,
    TFRT_ChainType);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = "operands attr-dict";
}

foreach dtype = ["i32", "f32"] in {
  def Dist_BroadcastOp_#dtype : BroadcastOp<dtype>;
}

// Test kernels
// TODO(ayushd): Implement set_up_from_string, create_collective_group kernels
def CreateConfigurations : DistOp<"test_create_configurations"> {
  let summary = "dist.test_create_configurations operation";

  let description = [{
    A test kernel that creates a distributed context configuration consisting of
    N members running on localhost, grpc_communicator as a fabric communicator,
    and one collective group of all members.

    Example:
      // Creating 4 DistributedContextConfigurations
      %configs:4 = dist.test_create_configurations : 4
  }];

  let results = (outs Variadic<DistributedContextConfigurationType> : $result);
  let parser = [{ return tfrt::dist::parse$cppClass(parser, result); }];
}

def CreateDistributedContext : DistOp<"test_create_distributed_context"> {
  let summary = "dist.test_create_distributed_context config";

  let description = [{
    A test kernel that creates a distributed context from the given
    configuration.

    Example:
      // Creating 2 DistributedContextConfigurations
      %configs:2 = dist.test_create_configurations : 2
      // Creating first DistributedContext based on the first configuration
      %context0 = dist.test_create_distributed_context %configs#0
      // Creating first DistributedContext based on the second configuration
      %context0 = dist.test_create_distributed_context %configs#1
  }];

  let arguments = (ins DistributedContextConfigurationType);
  let results = (outs DistributedContextType : $result);
  let assemblyFormat = "operands attr-dict";
}

def CreateCollectiveGroupOp : DistOp<"create_collective_group"> {
  let summary = "dist.create_collective_group operation";

  let description = [{
    An operation that fetches a collective group by name from a distributed
    context.
  }];

  let arguments = (ins
    DistributedContextType:$context,
    StringType:$name
  );
  let results = (outs CollectiveGroupType);
  let assemblyFormat = "operands attr-dict";
}

def RemoteRegisterOp :  DistOp<"remote_register"> {
  let summary = "dist.remote_register operation";
  let description = [{
    Register a program on a remote location.

    "program" has to be a string representing MLIR in TFRT dialect

    Example:
      dist.remote_register (%context, %id) "program" "program_name"
  }];

  let arguments = (ins
    TFRT_ChainType:$in_op_chain,
    DistributedContextType:$context,
    I32:$host_id,
    StrAttr:$program,
    StrAttr:$program_name
  );
  let results = (outs
    TFRT_ChainType:$out_op_chain
  );

  let printer = [{ return tfrt::dist::print(p, *this); }];
  let parser = [{ return tfrt::dist::parse$cppClass(parser, result); }];
}

def RemoteExecuteOp :  DistOp<"remote_execute"> {
  let summary = "dist.remote_execute operation";
  let description = [{
    Executes a program on a remote location.
    // TODO(bramandia): Add supports for inputs and outputs.

    Example:
      dist.remote_execute (%context, %id) "program_name"
  }];

  let arguments = (ins
    TFRT_ChainType:$in_op_chain,
    DistributedContextType:$context,
    I32:$host_id,
    StrAttr:$program_name,
    Variadic<RemoteObjectIdType>:$arguments
  );
  let results = (outs
    TFRT_ChainType:$out_op_chain,
    Variadic<RemoteObjectIdType>:$results
  );

  let printer = [{ return tfrt::dist::print(p, *this); }];
  let parser = [{ return tfrt::dist::parse$cppClass(parser, result); }];
}

#endif  // DISTRIBUTED_OPS
