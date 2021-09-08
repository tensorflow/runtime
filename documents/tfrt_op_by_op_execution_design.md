# TFRT Op-by-op Execution Design

<!--* freshness: {
  owner: 'fishx'
  reviewed: '2020-04-14'
} *-->

<!-- TOC -->

## Objective

The new TensorFlow runtime (codename: TFRT) project aims at delivering
significant improvements in accelerator support, performance, production
deployment, convergence between mobile and data center stacks, as well as
architecture modularity for improved eng velocity. See
[TFRT Host Runtime Design](tfrt_host_runtime_design.md) for more context.

This document focuses on the Core Runtime, the
[eager](https://www.tensorflow.org/guide/eager) interface of the new TensorFlow
runtime. Here is the overall TFRT architecture, and how the Core Runtime fits
in:

![TFRT Architecture](img/tfrt-arch.svg "TFRT Architecture")

The Core Runtime mainly contains two parts: an Op notion which is equivalent to
the current TensorFlow op and an interface layer that higher level runtime (e.g.
TF Python Runtime, [Jax](https://github.com/google/jax),
[Swift for TensorFlow](https://github.com/tensorflow/swift)) can call into. It
can execute both native ops and composite ops on the local host, accelerator
devices or remotely.

**Goal:** This document only covers the Core Runtime for local op-by-op
execution.

**Non-goals:** Other important topics such as lowering TF graphs, Device
Runtimes, [Host Runtime](tfrt_host_runtime_design.md), Python Runtime, Remote
Execution and Composite Op Execution are out of scope for this document.

## Background: Host Runtime, graph lowering

This subsection provides some conceptual background for readers to review the
Core Runtime design.

We refer to the end user facing computation as an **Op**. An op can take
multiple tensors as inputs and produce multiple tensors as output. It is a high
level representation that typically captures more of the programmers’ intent
than lower level representation. The programmer can decide how they are going to
execute the ops: they can either execute one at a time (op-by-op execution) or
execute multiple ops as an **op graph**. We have many representations of op
graphs, including GraphDef, HLO and MLIR TF Dialect. An op graph is usually, but
not always, portable (aka device neutral). `Tensor`s are immutable values in op
computations.

**Kernel** is a lower level concept designed for efficient execution. A kernel
can be an arbitrary C++ function that takes arbitrary types of inputs and
produces arbitrary types of outputs. Normally we compose multiple kernels as a
**kernel graph** and execute the graph as a whole. A kernel graph can be
represented in a low level dialect of MLIR, [BEF](binary_executable_format.md)
or other forms. Kernel graphs are platform specific, and typically encode the
specific accelerators that will run the computation. Kernels can do tensor
computation as well. In a kernel, a tensor can either be an immutable value or a
mutable buffer.

Core Runtime provides the notion of op and a C++ API for the programmers to
execute the ops. This op execution API is equivalent to the
[`TFE_Execute`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/eager/c_api.h)
API in the current TF eager runtime.

Host Runtime provides the notion of kernel and mechanisms for executing kernel
graphs. Host Runtime also provides a few basic building blocks like
[Threading Model](tfrt_host_runtime_design.md#threading-model) and
[Memory Allocation](tfrt_host_runtime_design.md#memory-allocation).

## Requirements

-   Efficient op-by-op execution

    -   Current TensorFlow stack is optimized for graph execution, and incurs
        non-trivial overhead when dispatching a single op. A high performance
        low level runtime is a key toward high end-to-end eager performance.
    -   Minimize heap allocation in critical code path. Op should reuse the
        input buffer if no other op needs to use it.

-   Asynchronous execution

    -   We can return from runtime ASAP and allow clients to schedule subsequent
        op execution, thus maximizing the concurrency across the client.
    -   It opens up opportunities for future optimization (e.g. aggregate
        multiple pending ops and execute them as a graph).

-   Good error reporting

    -   Accurate error location tracking: Error should report with accurate
        location information (e.g. Python stack trace).
    -   Synchronously report metadata (shape) errors: allows for immediate error
        reports for a large class of errors to the user.

-   Modular and “pluggable” support for new accelerator devices

    -   Supporting new accelerator hardware should not involve hacking the
        entire system.

-   Unified execution path with Host Runtime

    -   We should use the same set of building blocks for op-by-op execution,
        graph lowering and graph execution.

## Op Execution API

[Core Runtime](https://github.com/tensorflow/runtime/blob/master/include/tfrt/core_runtime/core_runtime.h)
provides follow API for op-by-op execution:

```c++
void Execute(string_view op_name,
             OpHandler* op_handler,
             Location loc,
             MutableArrayRef<TensorHandle> arguments,
             const OpAttrsRef& attrs,
             MutableArrayRef<TensorHandle> results,
             AsyncValueRef<Chain>* chain);
```

After calling this API, Core Runtime will execute the op by the given `op_name`
on the specified `op_handler`. This API is thread-safe.

This section discusses the abstractions used in this API.

### `TensorHandle`

[`TensorHandle`](https://github.com/tensorflow/runtime/blob/master/include/tfrt/core_runtime/tensor_handle.h)
is a future semantic data abstraction used in Core Runtime. It is a type erased
type which can represent tensors that live in any `op_handler` (e.g. locally or
remotely; on GPU or CPU), has different dtype, layout (dense format or sparse
format) and shape. It contains two parts:

*   A `Tensor` [`AsyncValue`](async_value.md), which holds the underlying data.
*   A most-of-the time synchronous `TensorMetadata` which includes the shape and
    dtype of this tensor. Sometimes it can be `AsyncValue` as well because some
    ops cannot propagage metadata synchronously. `TensorMetadata` must be ready
    if `Tensor` is ready.

Type erasing in this API is important for dynamic dispatch: Basically we can use
the same `Execute()` API to execute any arbitrary op. And it slightly improves
usability, since then one op can handle different types of input tensor.
However, the downside is extra dynamic dispatch overhead (some extra switch
statements).

Unlike current TensorFlow which has a single `Tensor` type, TFRT uses different
`Tensor` types to represent tensors on different devices or in different
layouts.

`Execute()` is an asynchronous execution API. It can return the result
`TensorHandle`s before actually running the computation. In this case, the
client will receive unavailable `TensorHandle`s. The client can use these
not-ready `TensorHandle` as inputs to schedule more ops. Once the computation is
finished, the result `TensorHandle`s will be set to ready state and the
downstream ops will be executed as well.

`TensorHandle` is a **value semantic** type: its underlying data is usually
immutable after it is set to ready since it can be consumed by multiple
downstream ops at the same time. However, if the op knows that it has the last
reference to the input `TensorHandle` and no other ops are going to use this
`TensorHandle` anymore, then it is fine for the op to “steal” the buffer from
the input `TensorHandle` and reuse it in output `TensorHandle`.

#### Error `TensorHandle`

Core runtime supports fine-grained error propagation: when an op encounters an
error, only the affected downstream ops are skipped (a callback registered by
the client will be notified as well). Core runtime also supports cancellation.
Both error propagation and cancellation are supported by Error `TensorHandle`.

An Error `TensorHandle` is basically a `TensorHandle` whose tensor is an
`ErrorAsyncValue`. When the core runtime encounters an Error in any op's
argument, including the `Chain` argument, the core runtime skips execution of
the op and propagates the error as the op's output.

Since all ops are executed asynchronously, there could be multiple pending ops
enqueued at any moment. TFRT provides a cancellation API
[`HostContext::CancelExecution()`](https://github.com/tensorflow/runtime/blob/master/include/tfrt/host_context/host_context.h)
to cancel pending ops and upcoming ops. After calling cancellation API, the
output `TensorHandle`s of all pending ops will be set to Error `TensorHandle`
with cancellation error and `Execute()` API will return immediately with Error
`TensorHandle`. The user can exit cancellation state by calling
`HostContext::Restart()`.

### `OpHandler`

[**`OpHandler`**](https://github.com/tensorflow/runtime/blob/master/include/tfrt/core_runtime/op_handler.h)
is a flexible abstraction which determines how core runtime handles the op.
After calling `Execute()` API in core runtime, it will basically invoke
`OpHandler::Execute()` for the given `op_handler`:

```c++
// It is to pack the args of the CoreRuntime::Execute(). This is preferred
// over having a flattened list because we can write cleaner code if we want
// to delegate execution from one op_handler to another op_handler.
struct OpInvocation {
  // This is the name of the op to invoke, e.g. "tf.Add".
  string_view op_name;

  // Source location of the op invocation.
  Location loc;

  // This is the input arguments to an op invocation.
  MutableArrayRef<TensorHandle> arguments;

  // The attributes for the op invocation.
  const OpAttrsRef& attrs;

  // Result TensorHandles to be provided by the op invocation.
  MutableArrayRef<TensorHandle> results;

  AsyncValueRef<Chain>* chain;
};

class OpHandler {
  // Execute the op specified by op invocation on this op_handler.
  virtual bool Execute(const OpInvocation &invocation) = 0;
}
```

So based on different `op_handler`s the client provides, core runtime will have
different behavior.

This doc only focuses on **Device `OpHandler`** which allows us to execute an op
eagerly on a specific CPU/accelerator device. For example,
[`CpuOpHandler`](https://github.com/tensorflow/runtime/blob/master/backends/cpu/lib/core_runtime/cpu_op_handler.cc)
and
[`GpuOpHandler`](https://github.com/tensorflow/runtime/blob/master/backends/gpu/lib/core_runtime/gpu_op_handler.cc).

There are also many other pseudo `op_handler`s for different purposes:

*   [**Sync Logging `OpHandler`**](https://github.com/tensorflow/runtime/blob/master/include/tfrt/core_runtime/logging_op_handler.h):
    This `op_handler` can print the inputs and outputs of every ops. We used it
    for debugging.

These pseudo `op_handler`s are out of scope for this document.

### `OpAttrs`

The client can send attributes to an op via
[**`OpAttrs`**](https://github.com/tensorflow/runtime/blob/master/include/tfrt/core_runtime/op_attrs.h).
An `OpAttrs` maps from the name of an attribute to its value. It supports
multiple data types including trivial types like `int32`, array types of any
trivial types, string and so on.

Attributes need to live on the heap when ops are executed asynchronously. On one
hand, we want to minimize heap allocations since it is expensive, but on the
other hand, we also want to be flexible enough to support an arbitrary number of
attributes. As such, we introduce following three types:

*   `OpAttrs`, a stack allocated data structure which has two inline buffers:
    one to store up to 6 `OpAttrsRawEntry` and the other is a buffer with the
    size of 128 bytes for both attribute keys and values. Users use this type to
    construct the attributes map.
    *   If the buffers still have space, adding a new small attribute does not
        cause any heap allocation.
    *   If either of these buffers is full, we will move the attributes to a
        `StringMap` allocated on the heap. In this case, adding a new attribute
        causes at most **two** heap allocations: the first one is to allocate an
        entry for `StringMap`, the second one is to allocate a buffer for
        attribute value.
*   `ImmutableOpAttrs`, a heap allocated and ref counted copy of `OpAttrs`.
*   `OpAttrsRef` is an either reference to `OpAttrs` or `ImmutableOpAttrs`. Core
    Runtime, including op implementations, uses this type to get attributes. It
    has a method called `freeze()` which can extend the attributes' lifetime.
    `freeze()` has two different behavior:
    *   If it is a reference to `OpAttrs`, `freeze()` will heap allocate an
        `ImmutableOpAttrs`, copy all attributes to it and return a `OpAttrsRef`
        of the `ImmutableOpAttrs`.
    *   If it is a reference to `ImmutableOpAttrs`, `freeze()` will return a
        newly constructed `OpAttrsRef` which holds a reference to underlying
        `ImmutableOpAttrs`.

To execute an op with attributes:

```c++
// No heap allocation in building the OpAttrs.
OpAttrs attrs;
attrs.Set("f32_attr", 2.0);
attrs.SetArray("array_attr", {2, 3, 4});
attrs.SetString("str_attr", "str");

// Construct an OpAttrsRef referencing to OpAttrs. Does not cause heap
// allocation.
core_runtime->Execute(..., OpAttrsRef(attrs), ...);
```

An op implementation can use the `Get()` methods in `OpAttrsRef` to retrieve the
attributes. Example usage:

```c++
// Enqueue an asynchronous task. If calling freeze() when the underlying class
// is OpAttrs, a ImmutableOpAttrs will be allocated on the heap.
host_context->Enqueue([attrs_ref = attrs_ref.freeze()] {
  float f32_attr = attrs_ref.GetAsserting<float>("f32_attr");
  ArrayRef<int32_t> array_attr = attrs_ref.GetArrayAsserting<int32_t>(
      "array_attr");
  string str_attr = attrs_ref.GetStringAsserting("str_attr");
});
```

With this design, asynchronous execution typically requires one heap allocation
for attributes. Also, this design is also flexible enough to handle ops that
contain many attributes.

### `Location`

A
[**`Location`**](https://github.com/tensorflow/runtime/blob/master/include/tfrt/host_context/location.h)
is an opaque token representing location information, e.g. python source code
file name and line number, provided by the client. The client needs to provide a
diagnostic handler to construct core runtime.

```c++
// TODO: Link to a more general error handling doc.
CoreRuntime(std::function<void(const DecodedDiagnostic&)> handler, ...);
```

When encountering an error in op execution, core runtime will call the error
handler with the encoded `Location`. The error handler can decide what to do
with the error. A typical implementation is to decode the `Location` and log an
error to end-users.

Right now we don’t yet have an end-to-end implementation on TFRT based error
reporting yet. This is highly dependent on the client integration (e.g.
TensorFlow Python Runtime integration) which is out of scope for this document.

### `Chain`

The `Chain` argument is optional for non-side-effecting op but required for
side-effecting op (See
[Non-side-effecting v.s. Side-effecting op](#non-side-effecting-vs-side-effecting-op)).
When it is present, TFRT will only execute the op when the `Chain` and all its
arguments are ready.

Other synchronization mechanisms (e.g. mutex) are allowed in TFRT op
implementation in addition to `Chain`s.

### Ownership of arguments

After calling `Execute()`, the ownership of arguments will transfer to core
runtime. That is why it takes `MutableArrayRef<TensorHandle> arguments`, all the
values in the array will be set to null when `Execute()` returns.

This design allows an op to track the reference to the buffer in an input
`TensorHandle` and it can reuse the buffer if the op holds the last reference to
the buffer.

The client needs to make a copy of `TensorHandle` to keep it alive. We manage
the size of `TensorHandle` carefully to ensure copying a `TensorHandle` is
cheap. As of 04/13/2020, a `TensorHandle` is 28 bytes in size including a
pointer to `Tensor` and an inline metadata.

## Native Op and Device `OpHandler`

To calculate metadata and tensor in `TensorHandle` respectively, a TFRT native
**op** is implemented by two C++ functions:

*   An optional **metadata function**, which aspires to propagate metadata
    synchronously. It can report metadata errors synchronously.
*   A **dispatch function**, which dispatches the real tensor computation
    asynchronously.

Ops are registered on a **Device `OpHandler`**. We have different device
`op_handler`s for different physical devices: CPU
([`CpuOpHandler`](https://github.com/tensorflow/runtime/blob/master/backends/cpu/lib/core_runtime/cpu_op_handler.cc))
and GPU
([`GpuOpHandler`](https://github.com/tensorflow/runtime/blob/master/backends/gpu/lib/core_runtime/gpu_op_handler.cc)).
Using `CpuOpHandler` as example, the registration API it provide is like:

```c++
class CpuOpRegistry {
  // Set a metadata function for the specified op_name.
  void AddMetadataFn(string_view op_name, OpMetadataFn metadata_fn);

  // Add an op with the specified dispatch function.
  void AddOp(string_view op_name, CpuDispatchFn dispatch_fn);
}
```

When executing an op with device `op_handler`, it will look up the metadata
function and dispatch function for the given `op_name` and run them on the
device respectively.

### Metadata Function

Metadata function computes the metadata of the result values of an op, and emit
any errors about invalid shapes, dtype or attribute.

Here is the signature implemented by all metadata functions:

```c++
using OpMetadataFn = RCReference<AsyncValue> (*)(
    ArrayRef<TensorMetadata> inputs, const OpAttrsRef& attrs,
    MutableArrayRef<TensorMetadata> results, Location loc);
```

As the signature shows, a metadata function is basically a synchronous C++
function which takes multiple metadata and attributes as input and produces
multiple metadata. Core runtime provides `TFRT_METADATA` macro to make defining
metadata functions more straightforward. An example implementation of metadata
function:

```c++
// Elementwise add operation.
// result = test.add(lhs, rhs)

// Metadata function
static Expected<TensorMetadata> TestAddMD(const TensorMetadata& lhs,
                                          const TensorMetadata& rhs) {
  // Error out if metadata mismatch
  if (lhs.dtype != rhs.dtype)
    return MakeStringError("incompatible dtypes for test.add");
  if (lhs.shape != rhs.shape)
    return MakeStringError("arguments of test.add must have same shape");
  return lhs;
}

// TFRT_METADATA macro converts TestAddMD into OpMetadataFn.
op_registry.AddMetadataFn("test.add", TFRT_METADATA(TestAddMD));
```

A metadata function can also use attributes set by the client:

```c++
// Metadata function for CreateDenseHostTensor
static Expected<TensorMetadata> CreateDHTMd(const OpAttrsRef& op_attrs) {
  ArrayRef<tfrt::Index> shape;
  if (!attrs.GetArray("shape", &shape))
    return MakeStringError("missing 'shape' attribute");
  …
  return TensorMetadata(..., shape);
}
```

Metadata functions are device independent. So both CPU, GPU or other accelerator
devices can share the same metadata function. No matter which device
`op_handler` is involved, core runtime will run the metadata function
synchronously on CPU.

### Dispatch Function

Dispatch function takes tensors as inputs, allocates and computes tensors for
its results.

Here is the signature implemented by all CPU dispatch functions:

```c++
using CpuDispatchFn = void (*)(ArrayRef<HostTensor*> inputs,
                               const OpAttrsRef& attrs,
                               ArrayRef<TensorMetadata> result_mds,
                               MutableArrayRef<RCReference<AsyncValue>> results,
                               AsyncValueRef<Chain>* chain, Location loc);
```

CPU Dispatch functions are strict C++ functions which can only take `HostTensor`
as inputs and produce asynchronous tensor results. If the op has a metadata
function, the `result_mds` arguments will contain the metadata of the results.
Otherwise it will be empty.

Similar to metadata functions, core runtime also provides macro `TFRT_CPU_OP`to
make defining CPU dispatch function more straightforward. An example
implementation of CPU dispatch function:

```c++
// Dispatch function
static Expected<DenseHostTensor> CpuAddOp(const DenseHostTensor& lhs,
                                          const DenseHostTensor& rhs,
                                          const TensorMetadata& result_md,
                                          HostContext* host) {
  // Allocate buffer for result
  auto dht = DenseHostTensor::CreateUninitialized(result_md.shape, host);
  if (!dht.hasValue())
    return MakeStringError("cannot allocate result tensor");

  DHTArrayView lhs_view(&lhs);
  DHTArrayView rhs_view(&rhs);
  MutableDHTArrayView dest_view(dht.getPointer());

  for (size_t i = 0; i != lhs.NumElements(); ++i)
    dest_view[i] = lhs_view[i] + rhs_view[i];

  return std::move(*dht);
}

// TFRT_CPU_OP macro converts CpuAddOp into CpuDispatchFn.
cpu_op_registry.AddOp("test.add", TFRT_CPU_OP(CpuAddOp));
```

A dispatch function can perform asynchronous computation as well by using
asynchronous computation library like Eigen or use thread pool from TFRT Host
runtime.

```c++
static AsyncValueRef<DenseHostTensor> AsyncAddOp(
    const DenseHostTensor& lhs, const DenseHostTensor& rhs,
    const TensorMetadata& result_md, HostContext* host) {
  // Allocate buffer for result, but the buffer has been marked as ready yet.
  auto dht =
      DenseHostTensor::MakeConstructedAsyncValueRef(result_md, host);

  // Perform `add` asynchronously.
  host->EnqueueWork([lhs = lhs.CopyRef(), rhs = rhs.CopyRef(),
                     dht = dht.CopyRef()] {
    DHTArrayView lhs_view(&lhs);
    DHTArrayView rhs_view(&rhs);
    MutableDHTArrayView dest_view(&dht);

    for (size_t i = 0; i != lhs.NumElements(); ++i)
      dest_view[i] = lhs_view[i] + rhs_view[i];

    // IMPORTANT: Only mark the buffer as ready after completing the
    // computation. Otherwise a downstream op may read the buffer before it is
    // set.
    dht.SetStateConcrete();
  });
  return dht;
}
```

Dispatch functions can use attributes set by the client similar to metadata
functions.

#### Data-dependent result metadata

For ops like [Reshape](https://www.tensorflow.org/api_docs/python/tf/reshape)
and [Unique](https://www.tensorflow.org/api_docs/python/tf/unique), their result
metadata is data dependent. So they don’t have metadata function and their
dispatch function is slightly different with normal ops:

```c++
// Dispatch function of reshape op
static Expected<DenseHostTensor> ReshapeOp(
    const DenseHostTensor& tensor,
    const DenseHostTensor& shape,
    // Reshape op does not have metadata function that produces result_md.
    // const TensorMetadata& result_md,
    HostContext* host) {
  ...
}
```

The only difference here is that the dispatch function of `ReshapeOp` doesn’t
take result metadata as one of its input arguments.

When executing this op, the result metadata will not be propagated
synchronously. Instead, it will be set asynchronously when the dispatch function
is finished. In addition, all the downstream ops cannot run their metadata
synchronously as well. As a result, after executing an op without metadata
function, core runtime loses the ability to report metadata error synchronously
for all downstream ops. To recover from that, users can wait until the metadata
is ready (Using `HostContext::Await()`) before calling the next op.

#### Dispatch function on `GpuOpHandler`

GPU Dispatch functions have slightly different signature because it has some
unique hardware features:

```c++
using GpuDispatchFn = void (*)(GpuDispatchContext* dctx,  // <-- Extra argument
                               ArrayRef<Tensor*> inputs,
                               const OpAttrsRef& attrs,
                               ArrayRef<TensorMetadata> result_mds,
                               MutableArrayRef<RCReference<AsyncValue>> results,
                               AsyncValueRef<Chain>* chain, Location loc);
```

The `GpuDispatchContext` contains GPU unique properties, e.g. GPU stream, GPU
allocator, etc. Core runtime also provides macro `TFRT_GPU_OP`to make defining
GPU dispatch function more straightforward.

Details of GPU support is out of scope for this doc. See this
[separate doc](cuda-proposal.md) for more details.

#### Type Conversion in dispatch function

Dispatch functions for different ops support different types of tensors. A
dispatch function can even support multiple types of tensors. For example, the
dispatch function of CPU Add op can support all following combinations:
`DenseHostTensor` + `DenseHostTensor`, `DenseHostTensor` + `ScalarHostTensor`,
`ScalarHostTensor` + `DenseHostTensor`, `ScalarHostTensor` + `ScalarHostTensor`.

There are multiple ops to allow users to convert tensor from one type to another
**explicitly**. For example,
[`gpu_tensor_to_host_tensor`](https://github.com/tensorflow/runtime/blob/master/backends/gpu/lib/ops/test/test_ops.cc)
op can convert an tensor from `GpuTensor` to `DenseHostTensor` by transferring
the data from GPU to CPU. These tensor conversion ops are registered on the
device `op_handler` that supports input type. For example, the input type
`gpu_tensor_to_host_tensor` is `GpuTensor`, so it will be registered on
`GpuOpHandler`.

Beforing calling an op, the input tensors must be explicitly converted to the
type that is supported by the op.

### Non-side-effecting v.s. Side-effecting op

Since `TensorHandle` is a value semantic type, most of the ops are
**non-side-effecting** ops, which means that its dispatch function should never
mutate the inputs’ buffers.

Non-side-effecting op does not need to handle `Chain`. Compared to
side-effecting op, it is more friendly to the end users and easier for compilers
to analyze.

On the other hand, we still need to support a very limited number of
**side-effecting** ops. This is a concrete example:

```c++
// Print op does not produce TensorHandle as output. So we need a chain to
// indicate it has finished.
static AsyncValueRef<Chain> PrintOp(const DenseHostTenosr& dht,
                                    const Chain& chain);
```

For side-effecting ops, we preserve the execution order of these ops by
providing them an `in_chain`, and these ops can produce an `out_chain` to
indicate its operation has finished.

## `corert` Kernels

In TFRT Host Runtime, we introduced BEFExecutor which is capable of executing a
kernel graph. In order to allow BEFExecutor to execute an “op” graph, we
introduce following kernel:

```c++
corert.executeop(<op_handler>) "op_name"(<tensorhandle inputs>) {attributes...}
      -> <list of tensorhandle results>
```

It is basically a wrapper kernel which can call into Core Runtime `Execute()`
API. There are also a few other kernels:

*   `corert.get_op_handler`, … : get the corresponding `op_handler` in core
    runtime
*   `corert.print_tensorhandle`: prints the shape and values of a
    `TensorHandle`.
*   …

This is an example op graph using `corert` kernels:

```c++
func @example() -> !tfrt.chain {
  %cpu = corert.get_op_handler("cpu")

  // Create TensorHandles
  %lhs = corert.executeop(%cpu)
    "test.create_dense_tensor"() { shape = [1, 1], values = [-1.0 : f32] }
  %rhs = corert.executeop(%cpu)
    "test.create_dense_tensor"() { shape = [1, 1], values = [-2.0 : f32] }

  %result = corert.executeop(%cpu) "test.add" (%lhs, %rhs)
  %ch0 = tfrt.new.chain
  %ch1 = corert.print_tensorhandle(%result, %ch0)
  tfrt.return %ch1 : !tfrt.chain
}
```
