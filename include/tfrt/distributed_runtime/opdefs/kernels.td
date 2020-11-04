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
include "tfrt/core_runtime/opdefs/corert_base.td"
include "tfrt/tfrt_op_base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Distributed dialect.
def DistDialect : Dialect {
  let name = "tfrt_dist";

  let description = [{The Distributed dialect.
                     This dialect contains distributed execution operations.}];

  let cppNamespace = "::tfrt::dist";
}

def DistributedContextType
    : OpaqueType<"tfrt_dist", "dist_context", "!tfrt_dist.dist_context type">;

def DistributedContextConfigurationType
    : OpaqueType<"tfrt_dist", "dist_context_configuration",
                 "!tfrt_dist.dist_context_configuration type">;

def CollectiveGroupType
    : OpaqueType<"tfrt_dist", "collective_group", "!tfrt_dist.collective_group type">;

def RemoteObjectIdType
    : OpaqueType<"tfrt_dist", "remote_object_id", "!tfrt_dist.remote_object_id type">;

def RemoteExecuteSpecType
    : OpaqueType<"tfrt_dist", "remote_execute_spec", "!tfrt_dist.remote_execute_spec type">;

// Base class for the operation in this dialect
class DistOp<string mnemonic, list<OpTrait> traits = []>
    : Op<DistDialect, mnemonic, !listconcat(traits, [IsolatedFromAbove])> {
}

// TODO(ayushd, xldrx): consider changing reduction_fn from attribute to a
// part of the kernel name.
class AllReduceOp<string dtype> : DistOp<"cpu.allreduce." # dtype> {
  let summary = "tfrt_dist.cpu.allreduce operation";

  let description = [{
    An operation to perform an allreduce primitive on members of a communicator.
    When done,the out_chain is set. Collective on different devices share an
    identical instance_id. A collective takes a communicator, instance id as
    string, a tensor as the input, a tensor to hold the result, and a chain and
    outputs a chain. It also takes the reduction functions as an attribute,
    "reduction_fn". Currently the following functions are supported: "sum".

    Example:
      %out_chain = tfrt_dist.cpu.allreduce.f32 %context, %collective_group, %instance_id, %tensor_in, %tensor_out, %in_chain {reduction_fn="sum"};
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
  let summary = "tfrt_dist.cpu.broadcast operation";

  let description = [{
    An operation to perform a broadcast primitive from one member to all other
    members of a communicator. This collective takes a communicator, an instance
    id (identical on all devices), an input, and a sender id, and outputs a
    chain. The sender places data to be broadcasted in the input. The receivers
    prepare an empty buffer in the input.

    Example:
      %out_chain = tfrt_dist.cpu.broadcast.i32 %context, %instance_id, %input, %sender_id, %in_chain
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
  let summary = "tfrt_dist.test_create_configurations operation";

  let description = [{
    A test kernel that creates a distributed context configuration consisting of
    N members running on localhost, grpc_communicator as a fabric communicator,
    and one collective group of all members.

    Example:
      // Creating 4 DistributedContextConfigurations
      %configs:4 = tfrt_dist.test_create_configurations : 4
  }];

  let results = (outs Variadic<DistributedContextConfigurationType> : $result);
  let parser = [{ return tfrt::dist::parse$cppClass(parser, result); }];
}

def RegisterFakeCompilerPass : DistOp<"test_register_fake_compiler_pass"> {
  let summary = "tfrt_dist.test_register_fake_compiler_pass operation";

  let description = [{
    A test kernel that registers a fake compiler pass with hard-coded output.

    Example:
      tfrt_dist.test_register_fake_compiler_pass "pass_name" "lowered_output"
      (%output_device1, %output_device2)
  }];

  let arguments = (ins
    StrAttr:$compiler_pass_name,
    StrAttr:$compiled_program,
    Variadic<StringType>:$output_devices
  );

  let assemblyFormat = [{
      $compiler_pass_name $compiled_program `(` $output_devices `)` attr-dict
  }];}

def CreateDistributedContext : DistOp<"test_create_distributed_context"> {
  let summary = "tfrt_dist.test_create_distributed_context config";

  let description = [{
    A test kernel that creates a distributed context from the given
    configuration. At the end of executing ops with this context, use
    test_close_distributed_context to close it.

    Example:
      // Creating 2 DistributedContextConfigurations
      %configs:2 = tfrt_dist.test_create_configurations : 2
      // Creating first DistributedContext based on the first configuration
      %context0 = tfrt_dist.test_create_distributed_context %configs#0
      // Creating second DistributedContext based on the second configuration
      %context1 = tfrt_dist.test_create_distributed_context %configs#1

      // Executing ops that use the contexts
      %out_chain = ...

      // Closing first DistributedContext
      %close0 = tfrt_dist.test_close_distributed_context %context0, %out_chain
      // Closing second DistributedContext
      %close1 = tfrt_dist.test_close_distributed_context %context1, %close0
  }];

  let arguments = (ins DistributedContextConfigurationType);
  let results = (outs DistributedContextType : $result);
  let assemblyFormat = "operands attr-dict";
}

def CloseDistributedContext : DistOp<"test_close_distributed_context"> {
  let summary = "tfrt_dist.test_close_distributed_context config chain";

  let description = [{
    A test kernel that closes the given distributed context.

    See CreateDistributedContext for examples.
  }];

  let arguments = (ins
    DistributedContextType : $context,
    TFRT_ChainType : $in_op_chain);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = "operands attr-dict";
}

def CreateCollectiveGroupOp : DistOp<"create_collective_group"> {
  let summary = "tfrt_dist.create_collective_group operation";

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

def RegisterTFRTFunctionOp :  DistOp<"register_tfrt_function"> {
  let summary = "tfrt_dist.register_tfrt_function operation";
  let description = [{
    Register a TFRT program on a remote location.

    "program" has to be a string representing MLIR in TFRT dialect

    Example:
      %out_chain = tfrt_dist.register_tfrt_function (%in_chain, %context, %id) "program" "program_name"
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

  let assemblyFormat = [{
      `(` $in_op_chain`,` $context`,` $host_id `)` $program_name $program attr-dict
  }];
}

def RegisterTFFunctionOp :  DistOp<"register_tf_function"> {
  let summary = "dist.register_tf_function operation";
  let description = [{
    Register a program in TF dialect on a remote location.
    Returns the execution spec.

    "program" has to be a string representing MLIR in TF dialect

    Example:
      %spec = dist.register_tf_function (%in_chain, %context, %id)
          "program" "program_name"
  }];

  let arguments = (ins
    TFRT_ChainType:$in_op_chain,
    DistributedContextType:$context,
    I32:$host_id,
    StrAttr:$program,
    StrAttr:$program_name
  );
  let results = (outs
    RemoteExecuteSpecType:$out_spec
  );

  let assemblyFormat = [{
      `(` $in_op_chain`,` $context`,` $host_id `)` $program_name $program attr-dict
  }];
}

def CreateRemoteExecuteSpecOp :  DistOp<"create_remote_execute_spec"> {
  let summary = "tfrt_dist.create_remote_execute_spec operation";
  let description = [{
    Create a specification for remote execution.
    A specification contains a list of output devices.

    Example:
      %spec = tfrt_dist.create_remote_execute_spec %device_name1 %device_name2
  }];

  let arguments = (ins
    Variadic<StringType>:$output_devices
  );
  let results = (outs
    RemoteExecuteSpecType:$out_spec
  );

  let printer = [{ return tfrt::dist::print(p, *this); }];
  let parser = [{ return tfrt::dist::parse$cppClass(parser, result); }];
}


def RemoteExecuteFuncOp :  DistOp<"remote_execute_func"> {
  let summary = "tfrt_dist.remote_execute_func operation";
  let description = [{
    Executes a program on a remote location.

    Example:
      %chain, %results:2 = tfrt_dist.remote_execute_func[%chain, %context, %worker_id] @program_name(%arg1, %arg2, ...) : (...) -> (...)
  }];

  let arguments = (ins
    TFRT_ChainType:$in_op_chain,
    DistributedContextType:$context,
    I32:$host_id,
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$callee_args
  );
  let results = (outs
    TFRT_ChainType:$out_op_chain,
    Variadic<AnyType>:$results
  );
  let assemblyFormat = [{
      `[` $in_op_chain`,` $context`,` $host_id `]`  $callee `(` $callee_args `)` attr-dict `:` functional-type ( $callee_args , $results )
  }];

}

def RemoteExecuteOp :  DistOp<"remote_execute"> {
  let summary = "tfrt_dist.remote_execute operation";
  let description = [{
    Executes a program on a remote location.

    Example:
      %chain, %result = tfrt_dist.remote_execute [%context, %id, %spec] "program_name" (%in) : (!tfrt_dist.remote_object_id) -> !tfrt_dist.remote_object_id
  }];

  let arguments = (ins
    TFRT_ChainType:$in_op_chain,
    DistributedContextType:$context,
    I32:$host_id,
    RemoteExecuteSpecType:$spec,
    StrAttr:$program_name,
    Variadic<RemoteObjectIdType>:$arguments
  );
  let results = (outs
    TFRT_ChainType:$out_op_chain,
    Variadic<RemoteObjectIdType>:$results
  );

  let assemblyFormat = [{
      `[` $in_op_chain`,` $context`,` $host_id`,` $spec `]`  $program_name `(` $arguments `)` attr-dict `:` functional-type ( $arguments , $results )
  }];
}

def RemoteExecuteTHOp :  DistOp<"remote_execute_th"> {
  let summary = "tfrt_dist.remote_execute_th operation";
  let description = [{
    Executes a program on a remote location.
    This is similar to RemoteExecuteOp with a difference that we also returned
    the TensorHandle representing the remote object, in addition to the
    RemoteObjectId.

    The user specify an input N which indicates how many outputs in which the
    TensorHandle are to be returned.

    If the remote function returns K outputs, both the RemoteObjectId and the
    TensorHandle of the last N outputs will be returned. Whereas for the
    first K-N outputs, only the RemoteObjectId will be returned.
    It is required that the last N outputs are of type Tensor.
21
    Example:
      // "1" after spec indicates N above.
      %out_id1, %out_id2, %out_th2 =
          tfrt_dist.remote_execute_th (%context, %id, %spec, 1) "program_name" %in
  }];

  let arguments = (ins
    TFRT_ChainType:$in_op_chain,
    DistributedContextType:$context,
    I32:$host_id,
    RemoteExecuteSpecType:$spec,
    I32Attr:$num_tensorhandle_output,
    StrAttr:$program_name,
    Variadic<RemoteObjectIdType>:$arguments
  );
  let results = (outs
    TFRT_ChainType:$out_op_chain,
    // Return RemoteObjectId outputs followed by $num_tensorhandle_outputs.
    Variadic<AnyType>:$results
  );

  let assemblyFormat = [{
      `[` $in_op_chain`,` $context`,` $host_id`,` $spec`,` $num_tensorhandle_output `]`  $program_name `(` $arguments `)` attr-dict `:` functional-type ( $arguments , $results )
  }];
}

def RemoteExecuteTHPreallocatedOp :  DistOp<"remote_execute_th_preallocated"> {
  let summary = "tfrt_dist.remote_execute_th_preallocated operation";
  let description = [{
    Executes a program on a remote location.
    This is similar to remote_execute_th with a difference that the
    RemoteObjectIds of ALL of the output have been assigned previously and we
    should reuse them.

    Note: This kernel does not support taking preallocated RemoteIds for some
    (but not all) of the outputs. If there is a need, such support can be added
    in the future.

    Since the RemoteObjectIds of the outputs are given as inputs to this kernel,
    this kernel only returns TensorHandle as outputs.

    Example:
      // "1" after spec indicates that there is 1 RemoteObjectIds input in the
      // $arguments. The remaining arguments are the outputs RemoteObjectIds.
      // Subsequent "1" indicates that there is 1 TensorHandle output.
      %out_th2 =
          tfrt_dist.remote_execute_th_preallocated (%context, %id, %spec, 1, 1) "program_name" %in %out1 %out2
  }];

  let arguments = (ins
    TFRT_ChainType:$in_op_chain,
    DistributedContextType:$context,
    I32:$host_id,
    RemoteExecuteSpecType:$spec,
    I32Attr:$num_input,
    I32Attr:$num_tensorhandle_output,
    StrAttr:$program_name,
    Variadic<RemoteObjectIdType>:$arguments
  );
  let results = (outs
    TFRT_ChainType:$out_op_chain,
    // Return $num_tensorhandle_outputs.
    Variadic<CoreRT_TensorHandleType>:$results
  );

  let assemblyFormat = [{
      `[` $in_op_chain`,` $context`,` $host_id`,` $spec`,` $num_input`,` $num_tensorhandle_output `]`  $program_name `(` $arguments `)` attr-dict `:` functional-type ( $arguments , $results )
  }];
}

#endif  // DISTRIBUTED_OPS
