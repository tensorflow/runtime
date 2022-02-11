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
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Distributed dialect.
def DistDialect : Dialect {
  let name = "tfrt_dist";

  let description = [{The Distributed dialect.
                     This dialect contains distributed execution operations.}];

  let cppNamespace = "::tfrt::dist";

  let emitAccessorPrefix = kEmitAccessorPrefix_Raw;
}

def DistributedContextType :
  Type<CPred<"$_self.isa<tfrt::dist::DistributedContextType>()">, "!tfrt_dist.dist_context type">,
  BuildableType<"$_builder.getType<tfrt::dist::DistributedContextType>()">;

def DistributedContextConfigurationType :
  Type<CPred<"$_self.isa<tfrt::dist::DistributedContextConfigurationType>()">, "!tfrt_dist.dist_context_configuration type">,
  BuildableType<"$_builder.getType<tfrt::dist::DistributedContextConfigurationType>()">;

def TaskHandleType :
  Type<CPred<"$_self.isa<tfrt::dist::TaskHandleType>()">, "!tfrt_dist.task_handle type">,
  BuildableType<"$_builder.getType<tfrt::dist::TaskHandleType>()">;

def RemoteObjectIdType :
  Type<CPred<"$_self.isa<tfrt::dist::RemoteObjectIdType>()">, "!tfrt_dist.remote_object_id type">,
  BuildableType<"$_builder.getType<tfrt::dist::RemoteObjectIdType>()">;

def RemoteExecuteSpecType :
  Type<CPred<"$_self.isa<tfrt::dist::RemoteExecuteSpecType>()">, "!tfrt_dist.remote_execute_spec type">,
  BuildableType<"$_builder.getType<tfrt::dist::RemoteExecuteSpecType>()">;

def RemoteChainManagerType :
  Type<CPred<"$_self.isa<tfrt::dist::RemoteChainManagerType>()">, "!tfrt_dist.remote_chain_manager type">,
  BuildableType<"$_builder.getType<tfrt::dist::RemoteChainManagerType>()">;

def PayloadType :
  Type<CPred<"$_self.isa<tfrt::dist::PayloadType>()">, "!tfrt_dist.payload type">,
  BuildableType<"$_builder.getType<tfrt::dist::PayloadType>()">;

// Base class for the operation in this dialect
class DistOp<string mnemonic, list<Trait> traits = []>
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
      %out_chain = tfrt_dist.cpu.allreduce.f32 %context, %collective_group_name, %instance_id, %tensor_in, %tensor_out, %in_chain {reduction_fn="sum"};
  }];

  let arguments = (ins
    DistributedContextType,
    TFRT_StringType,
    TFRT_StringType,
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
    id (identical on all devices), an input, and a sender task, and outputs a
    chain. The sender places data to be broadcasted in the input. The receivers
    prepare an empty buffer in the input.

    Example:
      %out_chain = tfrt_dist.cpu.broadcast.i32 %context, %collective_group_name, %instance_id, %input, %sender_task, %in_chain
  }];

  let arguments = (ins
    DistributedContextType,
    TFRT_StringType,
    TFRT_StringType,
    TensorType,
    TaskHandleType,
    TFRT_ChainType);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = "operands attr-dict";
}

foreach dtype = ["i32", "f32"] in {
  def Dist_BroadcastOp_#dtype : BroadcastOp<dtype>;
}

class AllGatherFixedShapeOp<string dtype> : DistOp<"cpu.allgather_fixed_shape." # dtype> {
  let summary = "dist.cpu.allgather_fixed_shape operation";

  let description = [{
    An operation to perform an allgather primitive on tensors of the same shape
    on a given axis.

    Example:
      %out_chain = dist.cpu.allgather_fixed_shape.i32 %context, %collective_group_name, %instance_id, %input, %output, %axis, %in_chain

    Examples on how axis is used:
    Suppose worker0 with t0 and worker1 with t1 perform an AllGather.
      t0 = [[0,1],[2,3]] dimension = 2x2
      t1 = [[40,50],[60,70]] dimension = 2x2
     * case axis = 0 means gathering on the 0th axis
        output dimension = 4x2
        result = [[0,1],[2,3],[40,50],[60,70]]
     * case axis = 1 means gathering on the 1st axis
        output dimension = 2x4
        result = [[0,1,40,50],[2,3,60,70]]
  }];

  let arguments = (ins
    DistributedContextType,
    TFRT_StringType,
    TFRT_StringType,
    TensorType,
    TensorType,
    UI64,
    TFRT_ChainType);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = "operands attr-dict";
}

foreach dtype = ["i32", "f32"] in {
  def Dist_AllGatherFixedShapeOp_#dtype : AllGatherFixedShapeOp<dtype>;
}

class AllGatherAnyShapeOp<string dtype> : DistOp<"cpu.allgather_any_shape." # dtype> {
  let summary = "dist.cpu.allgather_any_shape operation";

  let description = [{
    An operation to perform an allgather primitive on tensors of variable shapes
    on a given axis. All dimensions except the axis must be equal.

    Example: out_chain, %output = tfrt_dist.cpu.allgather_any_shape.i32 %context, %collective_group_name, %instance_id, %input, %axis, %in_chain

    Examples on how axis is used:
    Suppose worker0 with t0 and worker1 with t1 perform an AllGather.
      t0 = [[0,1],[2,3]] dimension = 2x2
      t1 = [[40],[60]] dimension = 2x1
    Only AllGather on axis = 1 is allowed.
      output dimension = 2x3
      result = [[0,1,40],[2,3,60]]

    Suppose worker0 with t0 and worker1 with t1 perform an AllGather.
      t0 = [[0,1],[2,3]] dimension = 2x2
      t1 = [[40,50]] dimension = 1x2
    Only AllGather on axis = 0 is allowed.
      output dimension = 3x2
      result = [[0,1],[2,3],[40,50]]
  }];

  let arguments = (ins
    DistributedContextType,
    TFRT_StringType,
    TFRT_StringType,
    TensorType,
    UI64,
    TFRT_ChainType);
  let results = (outs TFRT_ChainType, TensorType);
  let assemblyFormat = "operands attr-dict";
}

foreach dtype = ["i32", "f32"] in {
  def Dist_AllGatherAnyShapeOp_#dtype : AllGatherAnyShapeOp<dtype>;
}

// Test kernels
// TODO(ayushd): Implement set_up_from_string, create_collective_group kernels
def CreateConfigurations : DistOp<"test_create_configurations",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "tfrt_dist.test_create_configurations operation";

  let description = [{
    A test kernel that creates a distributed context configuration consisting of
    N members running on localhost, grpc_communicator as a fabric communicator,
    and one collective group of all members.

    Example:
      // Creating 4 DistributedContextConfigurations
      %configs:4 = tfrt_dist.test_create_configurations : 4
  }];

  let arguments = (ins I64Attr:$n);

  let results = (outs Variadic<DistributedContextConfigurationType> : $result);

  let assemblyFormat = [{
      `:` $n  attr-dict
  }];
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
    Variadic<TFRT_StringType>:$output_devices
  );

  let results = (outs TFRT_ChainType : $chain);

  let assemblyFormat = [{
      $compiler_pass_name $compiled_program `(` $output_devices `)` attr-dict
  }];
}

def CreateDistributedContext : DistOp<"test_create_distributed_context"> {
  let summary = "tfrt_dist.test_create_distributed_context config";

  let description = [{
    A test kernel that creates a set of distributed contexts from the given
    set of configurations. At the end of executing ops with this context, use
    test_close_distributed_context to close it.

    Example:
      // Multi client setup
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

      // Single client setup
      // Create 2 Servers on 2 different ports. Ask first server to create
      // DistributedContext and create a remote context on the second server
      // Returns both the distributed contexts.
      %client_context, %worker_context =
          tfrt_dist.test_create_distributed_context %configs#0, %configs#1 :
          (!tfrt_dist.dist_context_configuration,
          !tfrt_dist.dist_context_configuration) ->
          (!tfrt_dist.dist_context, !tfrt_dist.dist_context)
  }];

  let arguments = (ins Variadic<DistributedContextConfigurationType> : $configs);
  let results = (outs Variadic<DistributedContextType> : $results);
  let assemblyFormat = [{
      $configs attr-dict `:` functional-type ( $configs , $results )
  }];
}

def CreateRemoteChainManager : DistOp<"test_create_remote_chain_manager"> {
  let summary = "tfrt_dist.test_create_remote_chain_manager config";

  let description = [{
    A test kernel that creates a RemoteChainManager from the given
    DistributedContext.
  }];

  let arguments = (ins DistributedContextType);
  let results = (outs RemoteChainManagerType : $result);
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

def GetTaskHandleOp : DistOp<"get_task_handle"> {
  let summary = "tfrt_dist.get_task_handle operation";

  let description = [{
    A test kernel to fetch task handle by name from the distributed context.
  }];

  let arguments = (ins
    DistributedContextType:$context,
    StrAttr:$task_name
  );
  let results = (outs TaskHandleType);
  let assemblyFormat = "operands attr-dict";
}

def RegisterTFRTFunctionOp :  DistOp<"register_tfrt_function"> {
  let summary = "tfrt_dist.register_tfrt_function operation";
  let description = [{
    Register a TFRT program on a remote location.

    "program" has to be a string representing MLIR in TFRT dialect

    Example:
      %out_chain = tfrt_dist.register_tfrt_function (%in_chain, %context, %remote_task) "program" "program_name"
  }];

  let arguments = (ins
    TFRT_ChainType:$in_op_chain,
    DistributedContextType:$context,
    TaskHandleType:$remote_task,
    StrAttr:$program,
    StrAttr:$program_name
  );
  let results = (outs
    TFRT_ChainType:$out_op_chain
  );

  let assemblyFormat = [{
      `(` $in_op_chain`,` $context`,` $remote_task `)` $program_name $program attr-dict
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
    TaskHandleType:$remote_task,
    StrAttr:$program,
    StrAttr:$program_name
  );
  let results = (outs
    RemoteExecuteSpecType:$out_spec
  );

  let assemblyFormat = [{
      `(` $in_op_chain`,` $context`,` $remote_task `)` $program_name $program attr-dict
  }];
}

def CreateRemoteExecuteSpecOp :  DistOp<"create_remote_execute_spec"> {
  let summary = "tfrt_dist.create_remote_execute_spec operation";
  let description = [{
    Create a specification for remote execution.
    A specification contains a list of output devices.

    Example:
      %spec = tfrt_dist.create_remote_execute_spec %context {output_devices = ["device_name1", "device_name2"]}
  }];

  let arguments = (ins
    DistributedContextType:$context,
    StrArrayAttr:$output_devices
  );
  let results = (outs
    RemoteExecuteSpecType:$out_spec
  );

  let assemblyFormat = "operands attr-dict";
}


def RemoteExecuteFuncOp :  DistOp<"remote_execute_func"> {
  let summary = "tfrt_dist.remote_execute_func operation";
  let description = [{
    Executes a program on a remote location.

    Example:
      %chain, %results:2 = tfrt_dist.remote_execute_func[%chain, %context, %remote_task] @program_name(%arg1, %arg2, ...) : (...) -> (...)
  }];

  let arguments = (ins
    TFRT_ChainType:$in_op_chain,
    DistributedContextType:$context,
    TaskHandleType:$remote_task,
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$callee_args
  );
  let results = (outs
    TFRT_ChainType:$out_op_chain,
    Variadic<AnyType>:$results
  );
  let assemblyFormat = [{
      `[` $in_op_chain`,` $context`,` $remote_task `]`  $callee `(` $callee_args `)` attr-dict `:` functional-type ( $callee_args , $results )
  }];

}

def RemoteExecuteOp :  DistOp<"remote_execute"> {
  let summary = "tfrt_dist.remote_execute operation";
  let description = [{
    Executes a program on a remote location.

    Example:
      %chain, %result = tfrt_dist.remote_execute [%context, %remote_task, %spec] "program_name" (%in) : (!tfrt_dist.remote_object_id) -> !tfrt_dist.remote_object_id
  }];

  let arguments = (ins
    TFRT_ChainType:$in_op_chain,
    DistributedContextType:$context,
    TaskHandleType:$remote_task,
    RemoteExecuteSpecType:$spec,
    StrAttr:$program_name,
    Variadic<RemoteObjectIdType>:$arguments
  );
  let results = (outs
    TFRT_ChainType:$out_op_chain,
    Variadic<RemoteObjectIdType>:$results
  );

  let assemblyFormat = [{
      `[` $in_op_chain`,` $context`,` $remote_task`,` $spec `]`  $program_name `(` $arguments `)` attr-dict `:` functional-type ( $arguments , $results )
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

    Example:
      // "1" after spec indicates N above.
      %out_id1, %out_id2, %out_th2 =
          tfrt_dist.remote_execute_th (%context, %remote_task, %spec, 1) "program_name" %in
  }];

  let arguments = (ins
    TFRT_ChainType:$in_op_chain,
    DistributedContextType:$context,
    TaskHandleType:$remote_task,
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
      `[` $in_op_chain`,` $context`,` $remote_task`,` $spec`,` $num_tensorhandle_output `]`  $program_name `(` $arguments `)` attr-dict `:` functional-type ( $arguments , $results )
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
          tfrt_dist.remote_execute_th_preallocated (%context, %remote_task, %spec, 1, 1) "program_name" %in %out1 %out2
  }];

  let arguments = (ins
    TFRT_ChainType:$in_op_chain,
    DistributedContextType:$context,
    TaskHandleType:$remote_task,
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
      `[` $in_op_chain`,` $context`,` $remote_task`,` $spec`,` $num_input`,` $num_tensorhandle_output `]`  $program_name `(` $arguments `)` attr-dict `:` functional-type ( $arguments , $results )
  }];
}

def GetChainForTaskHandleOp : DistOp<"get_chain_for_task_handle"> {
  let summary =
      "tfrt_dist.get_chain_for_task_handle chain_mgr remote_task_handle";

  let description = [{
    Get the remote chain of the given task on the given RemoteChainManager.
  }];

  let arguments = (ins
    TFRT_ChainType:$in_op_chain,
    RemoteChainManagerType:$manager,
    TaskHandleType:$remote_task);
  let results = (outs
    RemoteObjectIdType : $remote_chain);
  let assemblyFormat = "operands attr-dict";
}


def SetChainForTaskHandleOp : DistOp<"set_chain_for_task_handle"> {
  let summary =
      "tfrt_dist.set_chain_for_task_handle chain_mgr remote_task_handle";

  let description = [{
    Set the remote chain of the given task on the given RemoteChainManager.
  }];

  let arguments = (ins
    TFRT_ChainType:$in_op_chain,
    RemoteChainManagerType:$manager,
    TaskHandleType:$remote_task,
    RemoteObjectIdType : $remote_chain);
  let results = (outs
    TFRT_ChainType:$out_op_chain);
  let assemblyFormat = "operands attr-dict";
}

def GetDistributedContextOp :  DistOp<"get_distributed_context"> {
  let summary = "tfrt_dist.get_distributed_context";
  let description = [{
    Get a DistributedContext from the ExecutionContext.

    Example:
      %dist_context = tfrt_dist.get_distributed_context
  }];

  let arguments = (ins);
  let results = (outs DistributedContextType : $result);
  let assemblyFormat = "attr-dict";
}

// TODO(donglin): Remove this op once we can use tfrt_dist.get_chain_for_host
// to get a remote chain for a given host.
def GetDummyRemoteChainOp : DistOp<"get_dummy_remote_chain"> {
  let summary = "tfrt_test.get_dummy_remote_chain";

  let description = [{
    Get a dummy remote chain that can be used in tests.
  }];

  let arguments = (ins);
  let results = (outs RemoteObjectIdType : $remote_chain);
  let assemblyFormat = "attr-dict";
}

def GetRemoteObjectIdFromTHOp :  DistOp<"get_remote_object_id_from_th"> {
  let summary = "tfrt_dist.get_remote_object_id_from_th";
  let description = [{
    Get the remote object id from a TensorHandle. It should throw error if the
    TensorHandle does not wrap around a RemoteTensor.

    Example:
      %result = tfrt_dist.get_remote_object_id_from_th %tensor_handle
  }];

  let arguments = (ins CoreRT_TensorHandleType);
  let results = (outs RemoteObjectIdType : $result);
  let assemblyFormat = "operands attr-dict";
}

def SendBytesOp :  DistOp<"send_bytes"> {
  let summary = "tfrt_dist.send_bytes operation";
  let description = [{
    Sends serialized data.

    Example:
      %chain = tfrt_dist.send_bytes %context %receiver_task %key %serialized_payload
  }];

  let arguments = (ins
    DistributedContextType:$context,
    TaskHandleType:$host_task,
    TFRT_StringType:$instance_key,
    PayloadType:$serialized_payload
  );
  let results = (outs
    TFRT_ChainType:$out_op_chain
  );
  let assemblyFormat = "operands attr-dict";
}

def ReceiveBytesOp :  DistOp<"recv_bytes"> {
  let summary = "tfrt_dist.recv_bytes operation";
  let description = [{
    Receives serialized data.

    Example:
      %serialized_payload = tfrt_dist.recv_bytes %context %sender_task %key
  }];

  let arguments = (ins
    DistributedContextType:$context,
    TaskHandleType:$host_task,
    TFRT_StringType:$instance_key
  );
  let results = (outs
    PayloadType:$serialized
  );
  let assemblyFormat = "operands attr-dict";
}


#endif  // DISTRIBUTED_OPS
