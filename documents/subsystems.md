# Library and Subsystem Overview

<!--* freshness: {
  owner: 'hongm'
  reviewed: '2020-03-27'
} *-->

<!-- TOC -->

This document is a reference that describes the various libraries in the TFRT
project, as well as any important design notes and rationale for the
functionality defined within them.

## `host_context` Library

### `HostContext`

The `HostContext` provides the core abstractions that the host executor and
kernels interact with, including a concurrent work queue, an allocator for
tensor data, and a way to create asynchronous values, error reporting,
cancellation, and access to kernel registry. HostContext is the megaobject that
*everything* unavoidably depends on, even in the smallest profile.

For more context, please refer to
[TFRT Host Runtime Design](tfrt_host_runtime_design.md).

One not-yet-implemented design idea is that multiple CPU contexts may be active
and communicating with each other, e.g. in a multi-socket NUMA system. These
contexts could communicate to each other through send/receive operations
(implemented with `memcpy` for example) just as they communicate with any other
device.

### `ConcurrentWorkQueue`

Intended to allow multiple different threading model implementations with
different tradeoffs. The host executor shouldn't depend on the details of these
thread pools.

-   We want to support single threaded contexts, which have no concurrency and
    start no threads.
-   We want to support [Abseil](https://abseil.io/) thread pool, fibers and
    other lighter-weight abstractions.
-   Platform specific abstractions, e.g. pthreads, Windows threads, other exotic
    things.
-   We want to support exotic cases such as external processes, task schedulers
    that dynamically change the number of cores available to the process, mobile
    devices that power gate their cores, etc.
-   We don't want the core runtime to hardlink to any of these options, we want
    our dependencies kept clean.

Allowing the threading model to be pluggable, and it being a reference type also
allows other clients to put their work into the thread pool as well, e.g.
[Swift](https://developer.apple.com/swift/) programs that want to use the same
abstractions for an actor abstraction.

Users should subclass `ConcurrentWorkQueue` to meet their own needs.
`SingleThreadedWorkQueue` provides a single-threaded work queue,
`MultiThreadedWorkQueue` provides a simple multi-threaded work queue based on
`std::thread`, while `NonBlockingWorkQueue` provides a production-quality
high-performance multi-threaded work queue with work-stealing and other
features.

### `HostAllocator`

This is an abstraction over `malloc`/`free` that provides a standard
implementation, but also allows runtime clients to customize their allocation
policy.

NUMA systems are one reason for this to exist. A HostContext may represent a set
of cores, and its HostAllocator manages memory attached to those cores.

The deallocate hook takes a size and alignment in addition to a pointer, because
this enables some `malloc` implementations (those with free lists for small
sized objects) to be implemented more efficiently. This can be as much as a 30%
performance win for free.

The idea of the `HostAllocator` is that the majority of the large allocations
and other easy allocations specific to a `HostContext` should be funnelled
through it. However, there is no guarantee that all allocations within a
`HostContext` will go through `HostAllocator` - it simply is not important
enough or practical to do this: you do not want every `std::string` to have to
use a custom allocator.

Users should subclass `HostAllocator` to meet their own needs. `MallocAllocator`
provides a simple allocator based on malloc. There are two decorator classes,
`ProfiledAllocator` for profiling, and `LeakCheckAllocator` for checking memory
leaks.

## `bef_executor` Library

The BEF executor executes a program made up of an open set of operations and
types, which are provided through a registry mechanism - this allows us to
support an open set of runtimes and abstractions, by modeling them as individual
dialects in MLIR. The program is represented by a BEF function (see
[Binary Executable Format](binary_executable_format.md)).

The BEF executor dispatches operations as soon as their dataflow inputs are
available (we also support "non-strict kernels", which can be dispatched without
all inputs being available), in order to efficiently support concurrent op
computation which does not have a statically determined schedule.

Each BEF kernel is implemented by a C++ function that takes inputs and returns
results through the `AsyncValue` interface, and can do so after the operation
starts from an arbitrary worker thread. This means that a kernel can split its
computation into concurrently executing tasks (including materializing the final
results), and submit them to the `ConcurrentWorkQueue` for scheduling. This
style of execution is also useful for talking to device runtimes.

The execute() method on the executor touches each instruction (BEF kernel) in a
BEF function in a top-down order. The execution ordering always respects data
dependencies. Thread hops are minimized for sequences of instructions with data
dependencies.

## `Tensor` Library

`Tensor` is the base class. It contains a `TensorMetadata` as a member, and uses
[LLVM style RTTI](http://llvm.org/docs/ProgrammersManual.html#the-isa-cast-and-dyn-cast-templates)
via the `subclass()` API. `TensorMetadata` is a convenience concept that
contains both the shape and `DType` of a tensor.

`HostTensor` represents tensors whose data are stored in the host’s memory. The
`HostTensor` class is the base class for all host tensor types. It inherits from
Tensor but is itself an empty class. `DenseHostTensor`, `CooHostTensor` and
`ScalarHostTensor` derive from `HostTensor`.

`DenseHostTensor` only stores a `HostBuffer` that contains a memory blob and
does not have the static `DType` and rank information. To interpret the data in
a `DenseHostTensor`, the user needs to use one of the tensor view types,
`DHTArrayView<DType>` or `DHTIndexableView<DType, Rank>`. The view classes only
contain a pointer to the underlying `DenseHostTensor`.

To implement device tensors, the device runtime code inherits from the base
`Tensor` class to implement its own tensors. One example is `DenseGpuTensor`,
which represents GPU tensors with a dense memory layout. Internally,
`DenseGpuTensor` contains a `GpuBuffer` that contains a pointer for the GPU
memory.

`Tensor::ConvertToHostTensor` is a pure virtual function for converting all
tensor types to the host tensor abstraction as a common exchange "currency
type."

## `support` Library

This library is intended to be the bottom of the dependency stack, generally
built with header-only library functionality.

Notably we use many LLVM support library utilities, instead of
[Abseil](https://abseil.io/) for code size reasons. Further, the LLVM support
library provides important utilities such as `PointerUnion` that Abseil lacks,
and provides better implementations for utilities like `llvm::SmallVector` (has
a type-erased `SmallVectorImpl` subclass, compared to `absl::InlinedVector`).

In general, we want to adopt LLVM utilities that are aligned with future C++
standard libraries, and we intend to migrate to the latter when available.
