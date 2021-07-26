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

//===- tpu_ops.td ---------------------------------------------------------===//
//
// Operation definitions for tpu_ops.
//
//===----------------------------------------------------------------------===//

#ifdef TPU_OPS
#else
#define TPU_OPS

include "tfrt/tfrt_op_base.td"
include "tfrt/core_runtime/opdefs/corert_base.td"

// "tpurt" dialect
def TFRT_TPU_Dialect : Dialect {
  let name = "tpurt";

  let description = [{
    The TPU runtime dialect.

    This dialect contains common TPU runtime operations.
  }];

  let cppNamespace = "::tfrt::tpu";
}

// Type definition.
def ChipLocationType : OpaqueType<"tpurt", "chip_location", "!tpurt.chip_location type">;
def CoreLocationType : OpaqueType<"tpurt", "core_location", "!tpurt.core_location type">;
def CoreType : OpaqueType<"tpurt", "core_type", "!tpurt.core_type type">;
def DimensionsType : OpaqueType<"tpurt", "dimensions", "!tpurt.dimensions type">;
def EventType : OpaqueType<"tpurt", "event", "!tpurt.event type">;
def HostLocationType : OpaqueType<"tpurt", "host_location", "!tpurt.host_location type">;
def SharedMemoryLocationType : OpaqueType<"tpurt", "shared_memory_location",
  "!tpurt.shared_memory_location type">;
def SharedMemoryType : OpaqueType<"tpurt", "shared_memory_type",
  "!tpurt.shared_memory_type type">;
def SystemType : OpaqueType<"tpurt", "system", "!tpurt.system type">;
def TopologyType : OpaqueType<"tpurt", "topology", "!tpurt.topology type">;
def VersionType : OpaqueType<"tpurt", "version", "!tpurt.version type">;
def WrapType : OpaqueType<"tpurt", "wrap", "!tpurt.wrap type">;
def ResourceManagerType : OpaqueType<"tpurt", "resource_manager", "!tpurt.resource_manager type">;
def BufferHandleType : OpaqueType<"tpurt", "buffer_handle", "!tpurt.buffer_handle type">;
def DenseTensorType : OpaqueType<"tpurt", "dense_tensor", "!tpurt.dense_tensor type">;
def CompilationResultType : OpaqueType<"tpurt", "compilation_result", "!tpurt.compilation_result type">;
def LoadedProgram : OpaqueType<"tpurt", "loaded_program", "!tpurt.loaded_program type">;
def XlaShapeType : OpaqueType<"tpurt", "xla_shape", "!tpurt.xla_shape type">;

// Base class for TFRT TPU dialect ops.
class TPU_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<TFRT_TPU_Dialect, mnemonic, traits> {

  // Each registered op in the tpu namespace needs to provide all of a printer,
  // parser and verifier.
  let printer = [{ return tfrt::tpu::print(p, *this); }];
  let parser = [{ return tfrt::tpu::parse$cppClass(parser, result); }];
}

def TopologyDetectOp : TPU_Op<"topology.detect"> {
  let summary = "tpu topology.detect operation";
  let description = [{
    tpurt.topology.detect automatically detects a Topology object and returns it.

    Example:
      %topology = tpurt.topology.detect
  }];
  let results = (outs TopologyType);
  let assemblyFormat = "attr-dict";
}

def SystemInitializeOp : TPU_Op<"system.initialize"> {
  let summary = "tpu system.initialize";
  let description = [{
    tpurt.system.initialize initializes the TPU system and returns a handle to it.

    Example:
      %system0 = tpurt.system.initialize %topology0, %chain
  }];
  let arguments = (ins TopologyType, TFRT_ChainType);
  let results = (outs SystemType);
  let assemblyFormat = "operands attr-dict";
}

def SystemShutdownOp : TPU_Op<"system.shutdown"> {
  let summary = "tpu system.shutdown";
  let description = [{
    tpurt.system.shutdown shuts down the TPU system.

    Example:
      %chain1 = tpurt.system.shutdown %system0, %chain0
  }];
  let arguments = (ins SystemType, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = "operands attr-dict";
}

def ConfigureAndInitializeGlobalTpuOp : TPU_Op<"system.configure_and_initialize_global_tpu"> {
  let summary = "Initialize the TPU system";
  let description = [{
    tpurt.system.configure_distributed_tpu initializes the TPU system, device,
    and ophandler, and returns a handle to the TPU topology serialized as a
    string.  It returns a TensorHandle because it is directly consumed in upper
    layer's code.

    Example:
      %topology = tpurt.system.configure_distributed_tpu
  }];
  let arguments = (ins
    TFRT_ChainType:$in_ch,
    StrAttr:$device
  );
  let results = (outs CoreRT_TensorHandleType);
  let assemblyFormat = "operands attr-dict";
}

def ShutdownTpuSystemOp : TPU_Op<"system.shutdown_tpu_system"> {
  let summary = "Shuts down the TPU system";

  let description = [{
    Shuts down the TPU system and clear the TPU states. Returns a TensorHandle that contains
    a boolean tensor to indicate if the shutdown process is successful.

    Example:
      %success = tpurt.system.shutdown_tpu_system
  }];

  let arguments = (ins
    TFRT_ChainType:$in_ch
  );
  let results = (outs
    CoreRT_TensorHandleType:$success
  );
  let assemblyFormat = "operands attr-dict";
}

def ConfigureDistributedTpuOp : TPU_Op<"system.configure_distributed_tpu"> {
  let summary = "Initialize the TPU system";
  let description = [{
    tpurt.system.configure_distributed_tpu initializes the TPU system, device,
    and ophandler, and returns a handle to the TPU topology serialized as a
    string.  It returns a TensorHandle because it is directly consumed in upper
    layer's code.

    Example:
      %topology = tpurt.system.configure_distributed_tpu
  }];
  let arguments = (ins TFRT_ChainType);
  let results = (outs CoreRT_TensorHandleType);
  let assemblyFormat = "operands attr-dict";
}

def CompileOp : TPU_Op<"compile"> {
  let summary = "tpu compile operation";
  let description = [{
    tpurt.compile compiles a mlir tpu program and returns a TpuCompilationCacheEntry.

    Example:
      %compilation_result = tpurt.compile (%dynamic_shapes) metadata = "metadata..." mlir_module = "mlir_module..."

    %dynamic_shapes is a list of TensorHandle with type being either
    tfrt::DenseHostTensor or tensorflow::tfd::RuntimeFallbackTensor.
  }];
  let arguments = (ins
    TFRT_ChainType:$in_ch,
    Variadic<CoreRT_TensorHandleType>:$dynamic_shapes,
    StrAttr:$metadata,
    StrAttr:$mlir_module
  );
  let results = (outs CompilationResultType);
}

def CompileV2Op : TPU_Op<"compile_v2"> {
  let summary = "tpu compile operation";
  let description = [{
    tpurt.compile_v2 compiles a mlir tpu program and returns a TpuCompilationCacheEntry populated in given resource manager.

    Example:
      %compilation_result = tpurt.compile (%resource_manager, %dynamic_shapes) metadata = "metadata..." mlir_module = "mlir_module..."

    %resource_manager is the resource manager that holds the compilation cache.
    %dynamic_shapes is a list of TensorHandle with type being either
    tfrt::DenseHostTensor or tensorflow::tfd::RuntimeFallbackTensor.
  }];
  let arguments = (ins
    ResourceManagerType:$resource_manager,
    Variadic<CoreRT_TensorHandleType>:$dynamic_shapes,
    StrAttr:$metadata,
    StrAttr:$mlir_module
  );
  let results = (outs CompilationResultType);
}

def ExecuteOp : TPU_Op<"execute"> {
  let summary = "tpu execute operation";
  let description = [{
    tpurt.execute executes the compiled tpu program.

    Example:
      %result = tpurt.execute %loaded_program, %core_location (%operands) : 1
  }];
  let arguments = (ins
    LoadedProgram:$loaded_program,
    CoreLocationType:$core_location,
    Variadic<DenseTensorType>:$operands,
    I1Attr:$use_core_selector
  );

  let results = (outs
    Variadic<DenseTensorType>:$results
  );

  let assemblyFormat = [{
    $loaded_program`,` $core_location` ` `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

def ExecuteV2Op : TPU_Op<"execute_v2"> {
  let summary = "tpu execute operation";
  let description = [{
    tpurt.execute_v2 executes the compiled tpu program with given system and resource manager.

    Example:
      %result = tpurt.execute %system, %resource_manager, %loaded_program, %core_location (%operands) : 1
  }];
  let arguments = (ins
    SystemType:$system,
    ResourceManagerType:$resource_manager,
    LoadedProgram:$loaded_program,
    CoreLocationType:$core_location,
    Variadic<DenseTensorType>:$operands
  );

  let results = (outs
    Variadic<DenseTensorType>:$results
  );
}

def ExecuteAndUpdateVariablesOp : TPU_Op<"execute_and_update_variables"> {
  let summary = "tpu execute and update variables operation";

  let description = [{
    tpurt.execute_and_update_variables executes the compiled tpu program and updates the variables in ResourceManager.
    var_reads are the indices of the inputs that represent variables read by this kernel.
    var_updates are the indices of the outputs that represent updated variable values.
    (var_reads[i], var_updates[i]) signifies that the var_reads[i]th input to this kernel is a resource variable read by this kernel,
    and the var_updates[i]th output of the TPU binary execution is the updated value of this variable.

    Example:
      %out_chain = tpurt.execute_and_update_variables %loaded_program, %core_location (%operands) {var_reads = [1, 2], var_updates = [0, 1]}
  }];

  let arguments = (ins
    LoadedProgram:$loaded_program,
    CoreLocationType:$core_location,
    Variadic<AnyType>:$operands,
    I64ArrayAttr:$var_reads,
    I64ArrayAttr:$var_updates
  );

  let results = (outs
    TFRT_ChainType:$out_chain,
    Variadic<AnyType>:$results
  );

  let assemblyFormat = [{
    $loaded_program $core_location `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

def TensorHandleToDenseTpuTensorOp : TPU_Op<"tensorhandle_to_dense_tpu_tensor"> {
  let summary = "Obtain DenseTpuTensor from tensor handle.";
  let description = [{
    Obtain DenseTpuTensor from tensor handle.

    Example:
      %dtt = tpurt.tensorhandle_to_dense_tpu_tensor %tensor_handle
  }];
  let arguments = (ins CoreRT_TensorHandleType);
  let results = (outs DenseTensorType);
  let assemblyFormat = "operands attr-dict";
}

def TransferFromTpuOp : TPU_Op<"transfer_from_tpu"> {
  let summary = "transfer tpu tensor to cpu tensor handle";
  let description = [{
    transfer tpu tensor to cpu tensor handle.

    Example:
      %result = tpurt.transfer_from_tpu %tensor
  }];
  let arguments = (ins DenseTensorType);
  let results = (outs CoreRT_TensorHandleType);
  let assemblyFormat = "operands attr-dict";
}

def DenseTpuTensorToTensorHandleOp : TPU_Op<"dense_tpu_tensor_to_tensorhandle"> {
  let summary = "Pack DenseTpuTensor to tensor handle.";
  let arguments = (ins DenseTensorType:$tpu_tensor, StrAttr:$device);
  let results = (outs CoreRT_TensorHandleType);
  let assemblyFormat = "operands attr-dict";
}

def TransferToTpuOp : TPU_Op<"transfer_to_tpu"> {
  let summary = "transfer cpu tensor handle to specified tpu core";
  let description = [{
    transfer cpu tensor handle to specified tpu core.

    Example:
      %result = tpurt.transfer_to_tpu %handle, %core_location, %preferred_layout
  }];
  let arguments = (ins CoreRT_TensorHandleType, CoreLocationType, XlaShapeType);
  let results = (outs DenseTensorType);
  let assemblyFormat = "operands attr-dict";
}

def TransferTensorFromTpuOp : TPU_Op<"transfer_tensor_from_tpu"> {
  let summary = "transfer tpu tensor to cpu tensor";
  let description = [{
    transfer tpu tensor to cpu tensor.

    Example:
      %host_tensor = tpurt.transfer_tensor_from_tpu %system, %tpu_tensor
  }];
  let arguments = (ins SystemType, DenseTensorType);
  let results = (outs TensorType);
  let assemblyFormat = "operands attr-dict";
}

def TransferTensorToTpuOp : TPU_Op<"transfer_tensor_to_tpu"> {
  let summary = "transfer cpu tensor to specified tpu core";
  let description = [{
    transfer cpu tensor to specified tpu core.

    Example:
      %tpu_tensor = tpurt.transfer_tensor_to_tpu %system, %host_tensor, %core_location, %preferred_layout
  }];
  let arguments = (ins SystemType, TensorType, CoreLocationType, XlaShapeType);
  let results = (outs DenseTensorType);
  let assemblyFormat = "operands attr-dict";
}

def BundledTransferToTpuOp : TPU_Op<"bundled_transfer_to_tpu", [SameVariadicOperandSize]> {
  let summary = "Bundle CPU variables and input tensors, and transfer them to specified TPU core in one single transfer dispatch routine";
  let description = [{
    Bundle CPU variables and input tensors, and transfer them to specified TPU core in one single transfer dispatch routine.
    var_indices are the indices of the inputs that are variables, the rest of the inputs are input tensors.
    var_indices is sorted in ascending order.

    Example:
      %result = tpurt.bundled_transfer %core_location, (%operands) (%preferred_layouts) { var_indices = [1, 3] }
  }];

  let arguments = (ins
    CoreLocationType:$core_location,
    Variadic<CoreRT_TensorHandleType>:$operands,
    Variadic<XlaShapeType>:$preferred_layouts,
    I64ArrayAttr:$var_indices
  );

  let results = (outs
    Variadic<DenseTensorType>:$results
  );

  let assemblyFormat = [{
    $core_location` ` `(` $operands `,` $preferred_layouts `)` attr-dict `:` type($results)
  }];
}

def MaybeTransferVariableOp : TPU_Op<"maybe_transfer_variable"> {
  let summary = "transfer cpu variable tensor handle to specified tpu core";
  let description = [{
    transfer cpu variable tensor handle to specified tpu core during warmup.

    Example:
      %result = tpurt.maybe_transfer_variable %handle, %core_location, %preferred_layout
  }];
  let arguments = (ins CoreRT_TensorHandleType, CoreLocationType, XlaShapeType);
  let results = (outs CoreRT_TensorHandleType);
  let assemblyFormat = "operands attr-dict";
}

def LookupVariableOp : TPU_Op<"lookup_variable"> {
  let summary = "lookup variable tensor handle on tpu";
  let description = [{
    lookup variable tensor handle on tpu given cpu variable tensor handle
    and ordinal.

    Example:
      %result = tpurt.lookup_variable %handle, %core_location
  }];
  let arguments = (ins CoreRT_TensorHandleType, CoreLocationType);
  let results = (outs DenseTensorType);
  let assemblyFormat = "operands attr-dict";
}

def TpuCoreSelectorOp : TPU_Op<"core_selector"> {
  let summary = "select tpu core";
  let description = [{
    select tpu core given compilation cache key.

    Example:
      %core_location = tpurt.core_selector %compilation_result
  }];
  let arguments = (ins CompilationResultType);
  let results = (outs CoreLocationType);
  let assemblyFormat = "operands attr-dict";
}

def TpuCoreSelectorV2Op : TPU_Op<"core_selector_v2"> {
  let summary = "select tpu core with specified resource manager.";
  let description = [{
    select tpu core given compilation cache key with specified resource manager.

    Example:
      %core_location = tpurt.core_selector_v2 %resource_manager, %compilation_result
  }];
  let arguments = (ins ResourceManagerType, CompilationResultType);
  let results = (outs CoreLocationType);
  let assemblyFormat = "operands attr-dict";
}

def LoadProgramOp : TPU_Op<"load_program"> {
  let summary = "Load program to specified tpu core";
  let description = [{
    load program to specified tpu core.

    Example:
      %result = tpurt.load_program %compilation_result, %core_location
  }];
  let arguments = (ins CompilationResultType, CoreLocationType);
  let results = (outs LoadedProgram);
  let assemblyFormat = "operands attr-dict";
}

def LoadProgramV2Op : TPU_Op<"load_program_v2"> {
  let summary = "Load program to specified tpu core with explicit system and resource manager.";
  let description = [{
    load program to specified tpu core with explicit system and resource manager.

    Example:
      %result = tpurt.load_program_v2 %system, %resource_manager, %compilation_result, %core_location
  }];
  let arguments = (ins SystemType, ResourceManagerType, CompilationResultType, CoreLocationType);
  let results = (outs LoadedProgram);
  let assemblyFormat = "operands attr-dict";
}

def GetInputLayoutOp : TPU_Op<"get_input_layout"> {
  let summary = "get input layout";
  let description = [{
    get input layout.

    Example:
      %device_shape = tpurt.get_input_layout %compilation_result : 1
  }];
  let arguments = (ins
    CompilationResultType:$compilation_result
  );

  let results = (outs
    Variadic<XlaShapeType>:$device_shape
  );
}

// Experimental/test kernels.
def CoreTypeOp : TPU_Op<"core_type"> {
  let summary = "tpu core_type operation";
  let description = [{
    tpurt.core_type casts an integer to a CoreType enum.

    Example:
      %core_type0 = tpurt.core_type %v0
  }];
  let arguments = (ins I32);
  let results = (outs CoreType);
  let assemblyFormat = "operands attr-dict";
}

def DimensionsOp : TPU_Op<"dimensions"> {
  let summary = "tpu dimension operation";
  let description = [{
    tpurt.dimensions creates a Dimensions object, which represents a 3D integral
    size or position.

    Example:
      %dimensions0 = tpurt.dimensions %v0, %v1, %v2
  }];
  let arguments = (ins I32, I32, I32);
  let results = (outs DimensionsType);
  let assemblyFormat = "operands attr-dict";
}

def EventToChainOp : TPU_Op<"event.to_chain"> {
  let summary = "tpu event.to_chain";
  let description = [{
    tpurt.event.to_chain produces a hex.chain after the Event has completed.

    Example:
      %chain0 = tpurt.event.to_chain %event0
  }];
  let arguments = (ins EventType);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = "operands attr-dict";
}

def PrintChipLocationOp : TPU_Op<"print.chip_location"> {
  let summary = "tpu print.chip_location operation";
  let description = [{
    tpurt.print.chip_location prints a human-readable representation of the
    ChipLocation enum to stdout.

    Example:
      %chain1 = tpurt.print.chip_location %chip_location0, %chain0
  }];
  let arguments = (ins ChipLocationType, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = "operands attr-dict";
}

def PrintCoreLocationOp : TPU_Op<"print.core_location"> {
  let summary = "tpu print.core_location operation";
  let description = [{
    tpurt.print.core_location prints a human-readable representation of the
    CoreLocation enum to stdout.

    Example:
      %chain1 = tpurt.print.core_location %core_location0, %chain0
  }];
  let arguments = (ins CoreLocationType, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = "operands attr-dict";
}

def PrintCoreTypeOp : TPU_Op<"print.core_type"> {
  let summary = "tpu print.core_type operation";
  let description = [{
    tpurt.print.core_type prints a human-readable representation of the given
    CoreType enum to stdout.

    Example:
      %chain1 = tpurt.print.core_type %core_type0, %chain0
  }];
  let arguments = (ins CoreType, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = "operands attr-dict";
}

def PrintHostLocationOp : TPU_Op<"print.host_location"> {
  let summary = "tpu print.host_location operation";
  let description = [{
    tpurt.print.host_location prints a human-readable representation of the
    HostLocation enum to stdout.

    Example:
      %chain1 = tpurt.print.host_location %host_location0, %chain0
  }];
  let arguments = (ins HostLocationType, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = "operands attr-dict";
}

def PrintSharedMemoryLocationOp : TPU_Op<"print.shared_memory_location"> {
  let summary = "tpu print.shared_memory_location operation";
  let description = [{
    tpurt.print.shared_memory_location prints a human-readable representation of
    the SharedMemoryLocatiEventToChainOpon enum to stdout.

    Example:
      %chain1 = tpurt.print.shared_memory_location %shared_memory_location0,
                %chain0
  }];
  let arguments = (ins SharedMemoryLocationType, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = "operands attr-dict";
}

def PrintSharedMemoryTypeOp : TPU_Op<"print.shared_memory_type"> {
  let summary = "tpu print.shared_memory_type operation";
  let description = [{
    tpurt.print.shared_memory_type prints a human-readable representation of the
    SharedMemoryType enum to stdout.

    Example:
      %chain1 = tpurt.print.shared_memory_type %shared_memory_type0, %chain0
  }];
  let arguments = (ins SharedMemoryType, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = "operands attr-dict";
}

def PrintVersionOp : TPU_Op<"print.version"> {
  let summary = "tpu print.version operation";
  let description = [{
    tpurt.print.version prints a human-readable representation of the given
    Version enum to stdout.

    Example:
      %chain1 = tpurt.print.version %version0, %chain0
  }];
  let arguments = (ins VersionType, TFRT_ChainType);
  let results = (outs TFRT_ChainType);
  let assemblyFormat = "operands attr-dict";
}

def SharedMemoryTypeOp : TPU_Op<"shared_memory_type"> {
  let summary = "tpu shared_memory_type operation";
  let description = [{
    tpurt.shared_memory_type casts an integer to a SharedMemoryType enum.

    Example:
      %shared_memory_type0 = tpurt.shared_memory_type %v0
  }];
  let arguments = (ins I32);
  let results = (outs SharedMemoryType);
  let assemblyFormat = "operands attr-dict";
}

def TopologyChipOp : TPU_Op<"topology.chip"> {
  let summary = "tpu topology.chip operation";
  let description = [{
    tpurt.topology.chip returns the ChipLocation for the chip_coordinates in the
    given Topology.

    Example:
      %chip_location0 = tpurt.topology.chip %topology0, %dimensions0
  }];
  let arguments = (ins TopologyType, DimensionsType);
  let results = (outs ChipLocationType);
  let assemblyFormat = "operands attr-dict";
}

def TopologyCoreOp : TPU_Op<"topology.core"> {
  let summary = "tpu topology.core operation";
  let description = [{
    tpurt.topology.core returns the CoreLocation for the chip_coordinates,
    core_type and core_index in the given Topology.

    Example:
      %core_location0 = tpurt.topology.core %topology0, %dimensions0,
                        %core_type0, %v0
  }];
  let arguments = (ins TopologyType, DimensionsType, CoreType, I32);
  let results = (outs CoreLocationType);
  let assemblyFormat = "operands attr-dict";
}

def TopologyCreateOp : TPU_Op<"topology.create"> {
  let summary = "tpu topology.create operation";
  let description = [{
    tpurt.topology.create creates a Topology object and returns it.

    Example:
      %topology0 = tpurt.topology.create %dimensions0, %dimensions1, %wrap0
  }];
  let arguments = (ins DimensionsType, DimensionsType, WrapType);
  let results = (outs TopologyType);
  let assemblyFormat = "operands attr-dict";
}

def TopologyHostOp : TPU_Op<"topology.host"> {
  let summary = "tpu topology.host operation";
  let description = [{
    tpurt.topology.host returns the HostLocation for the host_coordinates in the
    given Topology.

    Example:
      %host_location0 = tpurt.topology.host %topology0, %dimensions0
  }];
  let arguments = (ins TopologyType, DimensionsType);
  let results = (outs HostLocationType);
  let assemblyFormat = "operands attr-dict";
}

def TopologySharedMemoryOp : TPU_Op<"topology.shared_memory"> {
  let summary = "tpu topology.shared_memory operation";
  let description = [{
    tpurt.topology.shared_memory returns the SharedMemoryLocation for the
    chip_coordinates, shared_memory_type and shared_memory_index in the given
    Topology.

    Example:
      %shared_memory_location0 = tpurt.topology.shared_memory %topology0,
                                 %dimensions0, %shared_memory_type0, %v0
  }];
  let arguments = (ins TopologyType, DimensionsType, SharedMemoryType, I32);
  let results = (outs SharedMemoryLocationType);
  let assemblyFormat = "operands attr-dict";
}

def TopologyVersionOp : TPU_Op<"topology.version"> {
  let summary = "tpu topology.version operation";
  let description = [{
    tpurt.topology.version returns the Version for the chips in the given
    Topology.

    Example:
      %version0 = tpurt.topology.version %topology0
  }];
  let arguments = (ins TopologyType);
  let results = (outs VersionType);
  let assemblyFormat = "operands attr-dict";
}

def VersionOp : TPU_Op<"version"> {
  let summary = "tpu version operation";
  let description = [{
    tpurt.version casts an integer to a Version enum.

    Example:
      %version0 = tpurt.version %v0
  }];
  let arguments = (ins I32);
  let results = (outs VersionType);
  let assemblyFormat = "operands attr-dict";
}

def WrapOp : TPU_Op<"wrap"> {
  let summary = "tpu wrap operation";
  let description = [{
    tpurt.wrap creates a Wrap object, which indicates whether mesh wrapping is
    enabled in each dimension.

    Example:
      %wrap0 = tpurt.wrap %v0, %v1, %v2
  }];
  let arguments = (ins I1, I1, I1);
  let results = (outs WrapType);
  let assemblyFormat = "operands attr-dict";
}

def DeviceToCoreLocationOp : TPU_Op<"device_to_core_location"> {
  let summary = "convert device string to core location";
  let description = [{
    tpurt.device_to_core_location converts a device string to TPU core location.

    Example
      %core_location = tpurt.device_to_core_location {device = "/TPU:0"}
  }];
  let arguments = (ins
    TFRT_ChainType:$in_ch,
    StrAttr:$device
  );
  let results = (outs CoreLocationType);
  let assemblyFormat = "operands attr-dict";
}

#endif // TPU_OPS
