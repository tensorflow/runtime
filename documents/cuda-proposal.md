# CUDA in TensorFlow Runtime

<!--*
# Document freshness: For more information, see go/fresh-source.
freshness: { owner: 'xldrx' reviewed: '2021-04-26' }
*-->

## CUDA in TensorFlow Runtime

This document describes the proposed design for supporting CUDA devices in
TensorFlow Runtime (TFRT). Please note this preliminary proposal was written in
the early stages of the TFRT project and the overall design has changed
considerably. We are working on an updated document.

<!-- TOC -->

## Introduction

TFRT supports CUDA by defining a set of ops which are collectively called the
CUDA RunTime dialect or CRT dialect for short[^1]. This document discusses key
design choices and makes a proposal.

TFRT refers to functions launched by
[BEFExecutor](https://github.com/tensorflow/runtime/tree/master/tools/bef_executor)
as “kernels”. For example, the MLIR op `tfrt.add.i32` is implemented using the
TFRTAddI32 kernel. At the same time, CUDA uses “kernel” to mean the program
running on GPU. To limit the confusion we qualify the word kernel with either
CUDA or CRT. For example, the CRT `cuda.launch` kernel can be used to launch a
CUDA kernel.

## Larger GPU Ecosystem

In this design we are trying to expose a low level API into NVIDIA GPUs through
CUDA.

Other APIs for GPUs, e.g. Vulkan, SYCL, will be exposed through a different set
of low-level ops because the APIs are very different. In some cases, e.g. CUDA
&lt;-> Vulkan, we should be able to provide limited efficient interoperability.
For example, if a user allocates GPU memory using CUDA, does some CUDA
computation on it and wants to pass it to Vulkan, we should be able to provide a
kernel that will “transfer” the memory ownership from CUDA to Vulkan without
copying the buffer. Starting with version 10, CUDA exposes
[external semaphore and external memory abstractions](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXTRES__INTEROP.html)
precisely for such use cases. Details of such interoperation are beyond the
scope of this document.

An interesting special case is
[CUDA Graph](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html).
It is part of CUDA but is fundamentally different from other “immediate” CUDA
APIs. We are not planning to support construction of CUDA graphs natively in CRT
dialect, but as we will discuss later, kernels are free to construct and execute
CUDA graphs internally. From TFRT’s perspective there is no difference between a
CRT kernel that launches a single CUDA kernel and a CRT kernel that constructs
and launches a CUDA graph.

## Core Design Principles and Decisions

The CRT dialect aims to be a thin and predictable wrapper over core CUDA APIs.

CRT will support multiple CUDA devices and compute streams within each. In
particular, unlike TF today, which has a single dedicated compute stream and a
few transfer streams, CRT itself will not assign special purposes to streams.
Users can assign meanings to streams if they want. For example, a viable
alternative specially for eager execution is to use a single stream for
everything, including transfers, is a viable strategy for eager execution.
Another example, when prefetching data for the next step,
[`tf.data`](https://www.tensorflow.org/guide/data) can create and use its own
compute and transfer stream(s)[^3]. Finally, if we expose streams through eager
Python APIs, a sophisticated user should be able to design her own “stream usage
strategy”.

At least initially, CRT will not pick streams automatically. Eager execution
should perform well with a single stream and compilers should be able to assign
streams to functions they are lowering. If we discover a need, we can consider
making CRT pick streams automatically.

Since CRT is the lowest CUDA dialect in TFRT, we would like to avoid or at least
minimize features that diminish predictability. Such features include automatic
reordering of kernels, automatic stream synchronization (e.g. to ensure memory
safety), automatic reordering of memory allocations and deallocations. As we
will see later, it does not seem possible to avoid these completely.

## Illustrative Core Kernels

This section describes a few core kernels in CRT. This should give an idea of
how CRT makes CUDA usable within the BEFExecutor async execution semantics.

```
%buffer, %chain = cuda.mem.allocate %allocator, %stream, %size, %chain
```

As expected,
[`cuda.mem.allocate`](https://github.com/tensorflow/runtime/blob/80b7e677e2e341a44b21ae3c8c98f6f8e8b875cd/backends/gpu/include/tfrt/gpu/kernels/cuda_opdefs/cuda_ops.td#L194)
takes size and outputs a memory buffer. Since this kernel has a side effect,
following the TFRT convention, it also takes and returns a chain. Finally, the
kernel takes an allocator and a stream to associate each allocation with a
stream. We will discuss memory management in detail below. When exactly the
allocation takes place is not specified, but users are encouraged to follow the
generally good practice of allocating memory right before using it. By “right
before”; “right before” is defined in the dependency chain sense. The actual
timing difference can be large or small depending on how long the “allocate”
call takes to complete the buffer async value.

```
%chain = cuda.mem.copy_device_to_host %context, %host_buffer, %device_buffer, %chain
```

[`cuda.mem.copy_device_to_host`](https://github.com/tensorflow/runtime/blob/80b7e677e2e341a44b21ae3c8c98f6f8e8b875cd/backends/gpu/include/tfrt/gpu/kernels/cuda_opdefs/cuda_ops.td#L304)
schedules the transfer of device memory in `device_buffer` to host memory in
`host_buffer`. The `chain` input and output are regular chains to sequence this
kernel in BEFExecutor. This kernel basically just calls `cuMemcpyDtoHAsync`. To
know when the `host_buffer` can be used on CPU, i.e. when the operation
completes, the user would write something like this:

```
%ch1 = cuda.mem.copy_device_to_host %context %host_buffer %device_buffer %ch0
%ev %ch2 = cuda.event.create %flags %ch1
%ch3 = cuda.event.record %stream %ev %ch2
%ch4 = cuda.event.poll %ev %ch3
%ch5 = tfrt_dht.print_tensor %host_buffer %ch4
```

`cuda.event.create` simply calls `cuEventCreate`. `cuda.event.record` simply
calls `cuEventRecord`. `cuda.event.poll` starts polling for event completion.
When the event is reached, i.e. `cuEventQuery` returns `CUDA_SUCCESS`, `ch4`
completes. This chain can then be used as a dependency for accessing
`host_buffer`.

```
%chain = cuda.launch %stream %launchable <arguments> %chain
```

This kernel is an informed guess of what launching can look like. `cuda.launch`
launches a [`launchable`](#what-is-inside-the-launchable) with `arguments` on
`stream`. The output `chain` completes immediately. The `launchable` can contain
a few different types of programs. In the simplest case, it can contain a cuda
kernel symbol. Then, `cuda.launch` will simply call `cudaLaunchKernel`. At the
other extreme, it can be a host function that builds and executes a CUDA Graph.

## Proposal Summary

In most cases, the proposal is “let's do the simplest thing that should be good
enough and add complexity as we go”. If some term is not clear, please see the
relevant section(s) below. We don’t define the terms here to keep it short.

### CRT and New Kernels

Here are proposed design principles of CRT and new kernels:

*   No predefined special purpose streams
*   Allocate CUDA memory on-demand and cache
*   Don’t try to hide raw CUDA addresses
*   Satisfy allocations in the order allocation requests arrived
*   `cuda.mem.allocate` is an async kernel
*   `cuda.launch` will be a sync kernel
*   The `launchable` argument to `cuda.launch` can be quite complex and call a
    subset of CUDA APIs directly.
*   Don’t track launched kernels or try to limit the number of launched and
    unfinished kernels
*   Require the user to add necessary cross stream synchronizations for memory
    safety.
*   Require the user to add unexpected synchronizations if they are doing a few
    weird things
*   Automatically add necessary synchronizations to make buffer reuse safe.

These principles are explained in more details in the rest of this document.

## Streams

TensorFlow has a very specific design for using CUDA streams. There is a single
“compute” stream, a pair of `host_to_device` and `device_to_host` streams, as
well as a vector of `device_to_device` streams. All the streams except the
compute stream are used for corresponding transfers. This design mimics the
older GPU hardware capabilities. Older GPUs could run one kernel at a time and
do concurrent transfers to and from host at the same time. Thus, this design
allows TF to express as much parallelism to the GPU as it could handle. Most of
the code in TF GPU runtime assumes this design. For example, here is
[how GPU->CPU copy is performed](https://github.com/tensorflow/tensorflow/blob/0c6be605296f386b84925b5bc35392ec98fb6d6e/tensorflow/core/common_runtime/gpu/gpu_util.cc#L254).

An alternative design it to just use one stream for both transfers and
computation. This simple design generally works fine for eager execution because
of how imperative programs are generally written. For example, users generally
copy a tensor to CPU right before reaching a branch in the program. The program
can’t launch any kernel to be executed in parallel with the transfer because it
needs to look at the value that is being transferred first. Thus, it does not
hurt that no computation is happening concurrently with the transfer[^4].

As mentioned above, in CRT, we don’t dictate a specific “stream architecture”.
The eager runtime can default to using a single stream for everything. When
running a [tf.function](https://www.tensorflow.org/api_docs/python/tf/function),
a compiler pass can decide on the architecture. If the function is sequential,
i.e. kernels and transfers need the outputs of preceding kernels, a single
stream would be most efficient. If on the other hand, the function has parallel
tracks of computation and transfers, several streams can be useful. “Several
streams” is not limited to the TF’s architecture. For example, scheduling the
transfer right after the computation on the same stream is efficient and
perfectly fine.

We currently don’t plan to support automatic stream selection in CRT. Unlike CPU
threads, CUDA streams are just a mechanism to express dependencies. In other
words, a single kernel on a single stream is perfectly capable of utilizing the
full GPU. In fact, most cuDNN kernels seem to be written assuming that they are
the only thing running on the GPU and scheduling multiple of them on different
streams actually slows things down. Multiple streams are really needed only to
express parallelism between compute and transfers as well as to run multiple
small kernels in parallel.

## Memory Management

### Case Studies

**TF Memory Management:** One important challenge in TF CUDA memory management
for BEFExecutor’s asynchronous execution semantics is that allocation requests
in TF can sometimes block. When a TF CUDA kernel attempts to allocate device
memory, enough contiguous memory might not be available. This is usually because
the TF executor has launched a lot of CUDA kernels and memory transfers that
have not finished executing. Knowing this, instead of immediately raising an OOM
error, the `Allocator.allocate` call will block waiting for kernels/transfers to
finish and release some memory. If no memory has been freed after 10 seconds,
the OOM error is raised and execution terminates.

Another important detail is that memory lifetime semantics are tied to special
purpose streams. Memory lifetime on the single compute stream is
just-until-queued (JUQ). The common usage pattern is (1) allocate a buffer; (2)
launch a CUDA kernel using this buffer on the compute stream; (3) deallocate the
buffer without waiting for the kernel to complete. Formally, the contract for
the compute stream is that the buffer must stay allocated just until the kernel
using it is queued.

Memory lifetime of buffers used for transfers (i.e. on transfer streams) is
valid-until-termination (VUT). To perform a CPU->GPU copy TF (1) allocates a GPU
buffer; (2) launches a memcpy on the `host-to-device` stream; (3) deallocates
the GPU buffer after the copy has actually finished. Formally, the contract for
the transfer streams is that the buffer must stay allocated as long as there is
any operation on any transfer stream that touches this buffer.

In aggregate, the special purpose streams combined with their associated memory
lifetime contracts ensure the following property. A buffer returned by
`Allocator.allocate` might be in use only on the compute stream and no other
stream.

**PyTorch Memory Management:** PyTorch’s allocator is implemented in this
[file](https://github.com/pytorch/pytorch/blob/master/c10/cuda/CUDACachingAllocator.cpp).
Unlike TF[^5], PyTorch does not allocate a large chunk of memory in the
beginning. Instead, as memory requests come in, it just calls `cudaMalloc`.
Calling `cudaMalloc` frequently is very bad for performance because it blocks
until all outstanding GPU work completes. When a block is freed, it is not given
back to CUDA immediately and cached instead. Subsequent allocation requests can
be satisfied either by returning a free cached block, splitting a free cached
block, calling `cudaMalloc`, or giving all cached blocks back to cuda and
calling `cudaMalloc` again. Note that before calling `cudaFree` on a cached
block, one has to (and PyTorch does) block for all launched and unfinished
kernels to complete. This requirement is not documented but seems to be true.
This synchronization is usually needed for subsequent calls to `cudaMalloc`
anyway.

Each allocation is associated with a primary stream. Allocations and
deallocations are done in JUQ[^6] fashion with respect to the primary stream. A
cached allocation can only be re-allocated with the same primary stream. When
there are no free cached blocks associated with a desired stream, a new one is
allocated using `cudaMalloc`. The association with a primary stream should work
well as long as in steady state, the memory can be statically partitioned
between streams. Otherwise, each iteration will result in `cudaMalloc` calls.

When an allocated block is used on a non-primary stream, that stream is recorded
in the block structure. PyTorch’s allocator provides a public function
`recordStream` which users must call before using a block on a non-primary
stream. When the user calls `free()`, the block must not be used in kernels
launched afterwards, but it can’t be reused immediately (even on the primary
stream) because previously launched kernels on a non-primary stream might not be
done yet. PyTorch inserts CUDA events on all the non-primary streams that use
this buffer. These events are remembered in the block as well as in a shared
(across all blocks) queue of events to be processed. PyTorch does not track the
timestamp at which the block is used, the events are added to the top of the
streams. This can delay making the block available.

At the top of the malloc function, PyTorch calls `processEvents()`, which goes
over the queue checking which events have been reached. Events that have been
reached are removed from the queue and associated blocks. When all events are
removed from a block, the block can be reused and it is put on the free list.
Notably, the queue processing stops as soon as it encounters an event that has
not been reached. This saves CPU time but can delay realizing that some events
are complete and returning the memory to the cache.

If PyTorch is not able to find a block of memory (after looking at cached
blocks, returning them to CUDA, and calling `cudaMalloc`) it will call a list of
registered `FreeMemoryCallback`s. It seems the only use case for these callbacks
now is to free buffers used for sharing tensors between processes in
`CudaIPCTypes.h`. If these callbacks say that they released some memory, PyTorch
will call cudaMalloc again. If that fails, it returns an OOM error. In
particular, it does not attempt to wait until pending events are processed. This
is probably because the vast majority of PyTorch use cases use a single stream
for everything.

### One Big vs On-Demand Allocation and Fragmentation

By default, TF famously allocates most of available CUDA memory for itself in
the very beginning, while PyTorch allocates on-demand and caches. Obvious
trade-offs include:

*   TF does not incur cudaMalloc overheads in the beginning.
*   PyTorch users get easier visibility into their model’s memory usage using
    regular nvidia tools.

A more subtle trade-off[^7] is fragmentation[^8]. Consider a case where all the
GPU memory is used except for 10MB and these 10MB are split into two 5MB blocks.
In PyTorch’s case, `malloc` will see that there is no available 10MB block, it
will release these two blocks and invoke `cudaMalloc` again. Because
`cudaMalloc` returns virtual addresses from 49bit space, it will almost
certainly find 10MB of contiguous virtual addresses and will be able to return
the requested 10MBs. In TF’s case, unless these two blocks happened to be
adjacent, it would need to raise an OOM error.

Note that on-demand allocation does not solve all fragmentation issues and in
some cases, it can actually be worse than one big allocation. For example, a
single cached block can be split when a smaller allocation is requested and no
closely matching block is available. As long as the smaller block is alive, its
parent block cannot be freed. Because on-demand allocation manages many
discontinuous blocks, more blocks could be split than in one big allocation. For
example, consider the following allocation pattern: allocate 10 blocks of 10MB
each; free them; allocate 10 blocks of 6MB each. With one big allocation, this
pattern will result in zero fragmentation. With on-demand caching allocation,
this pattern will result in 10 split blocks, each with 6MB used and 4MB free.

Fragmentation has not been a huge issue in TF 1.* because the computation and
tensor lifetimes were generally regular[^9] and the JUQ allocation scheme tends
not to fragment memory too much. However, fragmentation will likely be a more
serious concern in the future. TF 2.0 eager execution and `tf.function`s enable
more “irregular” models. More complex allocation schemes (e.g. TF’s timestamped
allocator, stream-aware allocation) are more prone to fragmentation.

We are planning to adopt the on-demand allocation paradigm in CRT. If some usage
pattern is a bad fit for on-demand allocation, users can always do a dummy huge
allocation in the beginning to simulate the one big strategy.

### Compaction

We have considered whether to support compaction, moving allocated objects
together and to form larger contiguous empty space, and believe that the
drawbacks outweigh the benefits for GPU. The benefits are pretty obvious. If we
want to support compaction in the future, it will be possible. Here are the
drawbacks:

*   The GPU buffer object would need to be pretty large and ref-counted. Users
    like XLA GPU generally do one allocation for all intermediary buffers and
    statically subdivide them for different kernels. To represent these
    sub-buffers we would need to make them point to the parent buffer and
    remember the offset.

*   Raw addresses are obviously needed by GPU kernels. We would need to make
    sure all kernel developers know that they must not accidentally store the
    address somewhere. There should not be valid reasons for storing a raw
    pointer, but if some kernel writer accidentally does it, debugging will be
    pretty hard[^10].

*   Relocating buffers assumes that buffers don’t store pointers to themselves
    or other buffers internally. In other words, transparent relocation makes it
    impossible to implement data structures on GPU. RaggedTensors are already in
    TF today and it is likely that more will come in the future. Of course, it
    is always possible to make each data structure implement some `RelocateSelf`
    method, but it is extra complexity.

If we don’t plan to support compaction, how would we battle fragmentation when
it inevitably occurs?

*   As mentioned above, on-demand allocation gives us a way of exploiting CUDA
    virtual addresses to exchange a few small cached free blocks for one large
    contiguous block.
*   When on-demand allocation suffers from “can’t `cudaFree` this block because
    it has been split and part of it is used” problem, we can investigate
    various heuristics to choose between splitting a cached block and
    proactively `cudaFree`ing it and `cudaMalloc`ing just the right size.
*   Users and/or compilers can provide allocation lifetime hints. Satisfying
    allocations with similar lifetimes from the same block will reduce
    fragmentation.
*   Users and/or compilers can coalesce allocations with similar lifetimes to
    minimize allocation overhead and fragmentation.
*   We can expose some control over the block cache to users. The simplest
    control can be a ClearCache() method. More complex controls can include
    exposing something like an arena and allowing users to say “allocate from
    this arena”.
*   Users can also try to make larger parts of their programs statically shaped
    and let compilers figure out the best allocation strategy.

### Allocation Order

There are different alternatives for satisfying allocation requests[^11]. For
example, allocations can take place immediately in `cuda.mem.allocate` call or
later (see discussion
[below](#cuda-mem-allocate-returns-asyncvalue<bufferfuture>)). Allocations can
be satisfied in the order of `cuda.mem.allocate` calls arrival or in a different
order. For example, we can perform the allocation at a later stage, e.g. right
before executing a kernel that needs this buffer. When performing the allocation
right before the kernel execution, we can allocate all buffers needed by this
kernel and not worry about the order of `cuda.mem.allocate` calls. In this
section, we describe some of the inherent pros and cons of preserving the
allocation order and propose a strategy.

#### Allocate in-order within `cuda.mem.allocate` call

This is the simplest approach. Unfortunately, CRT cannot always perform the
allocation without blocking because it might have launched a large number of
CUDA kernels that collectively use all the available GPU memory. Satisfying the
allocation in this scenario requires waiting for kernel completion. This would
block the calling thread and violate the BEFExecutor requirement that kernel
functions return quickly and don’t block the calling thread.

#### Allocate in-order, potentially outside of `cuda.mem.allocate` call

Allocating in-order greatly increases system predictability, which is an
important feature of a low level API. Unfortunately, there are few potential
issues. First, as we mentioned earlier, TF today can block multiple GPU kernels
in `ctx.allocate_*` calls. The kernel whose allocations are satisfied gets
unblocked and can continue running. This mechanism effectively prioritizes
kernels with satisfiable allocation requests. If CRT promises to satisfy
allocations in order, some models (particularly those with many legacy kernels),
will likely fail with an OOM. This can be considered a regression from the
current TF stack and can hinder the adoption of TFRT.

Second, consider a toy scenario of running two independent identical models
consisting of three allocations, 3GB each, and a single kernel taking all three
buffers. Assume that the GPU has 10GB of memory. If CRT promises to satisfy
allocations in the order they arrive, and the allocation requests from these two
models interleave, at least one model will fail with OOM. Whether one or both
models fail, and if one, which one, is unpredictable. If allocations can be
satisfied out-of-order, we can satisfy all allocations of one kernel
“atomically”. This can guarantee that both toy models run without OOMing.
However, simple approaches like the “atomicity” stop working very quickly for
models containing more than 1 kernel. To sum up, satisfying allocations in-order
can decrease predictability in some, likely not important, corner cases.

This issue with multiple independent models running in parallel and interfering
is likely not very important. "Training" workloads generally occupy the whole
GPU and "Inference" workloads are generally uses cloud resources. In all these
cases, isolating the models by statically partitioning GPU memory is probably
needed anyway - to reduce latency in Waymo/Robotic use cases and to isolate
tenants in the Cloud use case. Such static partitioning would rule out the toy
scenario.

Finally, allocating in-order puts a burden on the compiler to sequence
allocations appropriately, in the very least compiler should make sure that
`cuda.mem.allocate` is not called much earlier than when it is actually needed.
In the graph below, `allocB` can happen before `kernelA` is launched.

```
allocA
  |
  |     allocB
  v      |
kernelA  |
    |    |
    v    v
    kernelB
```

This is fine to do (or even combine both allocations into one) when allocations
are known to be small, but in general the compiler should sequence allocations
and produce the following graph.

```
 allocA
  |
  v
 kernelA
    |  \\
    |   \\
    |   allocB
    |    |
    v    v
    kernelB
```

where the double-dashed line is a `chain`, i.e. a control dependency. Without
appropriate allocation scheduling by the compiler most models will OOM in the
very beginning. This should be fairly easy for the compiler to do using standard
techniques.

#### Allocate out-of-order

In both in-order and out-of-order scenarios, we respect the fundamental
dependencies in the computation graph. In particular, all the allocations needed
by the kernel will be satisfied before the kernel is launched. However, the
fundamental dependencies don’t completely define the allocation satisfaction
order. They leave some wiggle room. The in-order allocation strategy is saying
that we will satisfy the allocation in whatever order they happen to come to us.
The out-of-order strategy is saying that I will hold onto the allocation
requests and satisfy them in a more intelligent order.

Out-of-order strategy is effectively what TF does today when operating at the
brink of OOM (i.e. usually) and it gives CRT the most scheduling power. The
scheduling power allows optimizations like satisfying all allocations of a given
kernel “atomically” instead of satisfying half of allocations of a bunch of
kernels and hitting an OOM. The “atomic” allocation satisfaction is likely not
very important as neither TF nor PyTorch does it today.

Finally, it is worth realizing that any “intelligent” allocation order can be
encoded in the program, i.e. can theoretically be done by the compiler.
Consequently, it might be worth doing at runtime level only because actually
doing it in the compiler can be very hard and/or result in extra overhead from
many chains.

### Proposal

We propose to start with the second approach - allocating in-order, potentially
outside of `cuda.mem.allocate` kernel. This will make the system simpler and
easier to debug. Later on, we might find that some use cases require the runtime
to reorder allocations. One way to let compilers cleanly specify “I want you to
satisfy these `N` allocation requests in any order” is by adding a variadic
`cuda.mem.allocate` that takes parameters for all `N` requests at the same time
and satisfies them in some order. The BEF’s `AsyncValue` abstraction allows
return values (buffers in this case) to become ready in any order.

The important design decision at this point is not to specify when allocation
actually takes place but encourage compiler engineers to do the usual allocation
optimizations e.g. allocating right before usage and coalescing allocations when
beneficial. This will allow CRT to optimize and adopt allocation strategies as
code and use cases evolve.

## Kernel Execution

### Buffers or Buffer Futures

A distinctive feature of TFRT compared to TF is that kernels are discouraged
from allocating memory inside themselves[^12]. Consequently, kernels take all of
their input/output/temporary buffers as arguments. In this section, we make this
assumption and describe how we can handle the memory allocations inside the
kernel later.

To launch a CUDA kernel, all of its input buffers must be allocated, i.e. we
must know the actual virtual address. As we mentioned above, satisfying an
allocation can require waiting for launched kernels to complete. In other words,
some work must be performed asynchronously, after a CRT kernel function returns.
There are a couple of options for where and how asynchrony to put. They are
effectively decided by what exactly is returned from `cuda.mem.allocate`.

#### `cuda.mem.allocate` returns AsyncValue&lt;Buffer>

In this design, the buffer AsyncValue returned by `cuda.mem.allocate` will
complete only when the allocation was actually satisfied. This design makes it
possible for kernels consuming `AsyncValue<Buffer>` to be synchronous. For
example, a `crt.offset <buf> <offset> -> <child_buf>` kernel can simply create
the sub-buffer at the given offset of the parent buffer synchronously inside
itself without needing to schedule any async work. Another, more important,
example is that `cuda.launch` can be synchronous. Because BEFExecutor will call
it only when all the input buffers/tensors are fully ready, this kernel can
simply launch the launchable.

This design should work well with the recently extension of
[SyncKernel/SyncValue](https://github.com/tensorflow/runtime/commit/e52f863cb52bb02c3ad19f8bfbd007ba15752265)
to TFRT. If we can assume that all kernel launches and operations on buffers are
synchronous, we can invoke them synchronously and avoid AsyncValue overheads.

This design choice has a few potential issues:

*   CRT won’t see any kernels that depend on some buffer before that buffer has
    been allocated. This can prevent some optimizations. We will cover some in
    the next [sub-section](#cuda-mem-allocate-returns-asyncvalue<bufferfuture>)
    below.
*   If allocations are not sequenced (see the diagram in allocation order
    [section](#allocate-in-order-potentially-outside-of-cuda-mem-allocate-call)
    above) and they succeed, a large number of kernels can become runnable.
    Launching all of them without deferring can delay processing of out-of-band
    memcpy requests. Users/compilers can potentially ameliorate this issue by
    inserting CUDA events and waiting for them before dispatching more kernels.
    In other words, because BEFExecutor kernels are much more expressive than a
    GraphDef, various scheduling policies can be encoded in the program itself
    rather than in the runtime.
*   In the current BEFExecutor, `cuda.launch` can be invoked on any thread. In a
    multi-GPU scenario, this translates to multiple host threads launching
    kernels on multiple GPUs in a random fashion. This has a fairly large
    overhead due to internal CUDA lock contention[^13] as observed by the GPU
    Perf team in the current TF. This is part of the reason why private GPU
    thread pool was introduced to TF.

*   Another issue, and the other reason for a private GPU thread pool, is that
    once allocation succeeds CRT relies on the BEFExecutor to run all dependent
    `cuda.launch` kernels. In the current BEFExecutor, these `cuda.launch`
    kernels can be stuck behind compute heavy CPU task. This issue could be
    addressed by the introduction of
    [`AddBlockingTask`](https://github.com/tensorflow/runtime/blob/68a1543ecc1e5863ae64280575e8f0fd8b4fb25f/include/tfrt/host_context/concurrent_work_queue.h#L95)
    API that should be used not just for blocking but also for CPU heavy
    computation. Dispatch work should be added using the `AddTask` API.

#### `cuda.mem.allocate` returns AsyncValue&lt;BufferFuture>

In this design, `cuda.mem.allocate` would be synchronous, i.e. the return value
of `cuda.mem.allocate` would be complete when it returns. Instead of returning
`AsyncValue<Buffer>`, `cuda.mem.allocate` would return something like
`AsyncValue<BufferFuture>`. The difference with the section above is whether the
actual allocation took place when `cuda.mem.allocate` completes its return value
or not. In the previous option, the actual address is available in
`cuda.mem.allocate`'s return value. In this option, the return value is merely a
future that will contain the address at some point later. The `BufferFuture`
would be available immediately and BEFExecutor could use it to call `cuda.launch
<launchable> <buffer_futures>` immediately. However, because the actual address
is not yet available, `cuda.launch` would need to defer actually launching the
CUDA kernel. Similarly, all other CRT kernels taking in `BufferFutures` would
need to be async.

This design is more complex because CRT needs to deal with BufferFuture and
defer CUDA kernel launches, but it has some advantages. Advantages include:

*   The runtime can see the sequence of kernels to be launched before allocating
    the buffer. This can enable some runtime optimizations. For example, CRT can
    look ahead to find a stream that has a device-to-host transfer on top. It is
    likely that the work on this stream is on the critical path because that
    transfer will likely unlock some CPU computation (e.g. a conditional) and
    future work.
*   By deferring CUDA kernel launches, CRT can easily limit the number of
    outstanding launched kernels similarly to recent additions to TF.
*   By having a queue of kernels to launch, CRT can launch them using the same
    thread and shave off some overhead from CUDA context setting and internal
    CUDA lock contention.

#### Proposal

First, note that the designs above are not completely mutually exclusive. One
could imagine a world where `cuda.mem.allocate` returns a BufferFuture, there is
a `cuda.future_to_buf <buffer_future> -> <buffer>` kernel to convert a future
into a buffer, and a synchronous `cuda.launch` kernel that takes buffers.

Furthermore, returning `AsyncValue<Buffer>` does not necessarily preclude CRT
from seeing kernels using the buffer before satisfying the allocation. We can
define a non-strict version of `cuda.launch` that can be called by BEFExecutor
before `AsyncValue<Buffer>` is ready.

Weighing the trade-offs discussed in previous sections and the observations
above, we propose to start off with the first, simpler approach and consider
adding more elements of the second one if/when actual use cases arise.

### What is Inside the `launchable`?

#### Serializability

Early on in TFRT discussions, we discussed the question of whether a program in
the runtime dialect should be serializable or not. In particular, can
`launchable` contain a function pointer or not? The answer to this question has
only local influence to the design of CRT - it influences only the format of
`launchable`. In particular, if function pointers are allowed, we can implement
calling cuDNN just by including a function pointer in `launchable`. Otherwise,
we would need to come up with some serializable format to represent whatever we
want to call from `launchable`.

Supporting function pointers simplifies the implementation and is needed to
support XLA’s CustomCall HLO instruction. At the same time, there are no
immediate use cases requiring serializability[^14]. We will start off with
allowing function pointers in `launchable`.

#### Restrictions

Allowing `launchable` to contain a function pointer is quite powerful but can
also be dangerous if the function does something bad, e.g. call cudaDeviceReset.
The exact restrictions on what the function can do will evolve, but here are
some basics:

*   `launchable` should not destroy/reset anything global, e.g. contexts,
    streams, devices, etc.
*   `launchable` should launch its work on the given stream in the currently
    active device context. Note that while abstractions like TensorRT or CUDA
    Graphs can run multiple kernels in parallel internally, they still
    [take](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1g1accfe1da0c605a577c22d9751a09597)
    a stream argument for external synchronization.
*   `launchable` should not block since it will be called directly on the
    BEFExecutor thread.
*   `launchable` should not allocate its own memory if possible, but take device
    buffers as arguments.

### Execution Order

The allocation blocking mechanism in TF actually does some kernel “scheduling”.
Multiple GPU kernels can be blocked in `GPUBFCAllocator::Allocate` method. For
simplicity, assume one requires a very large allocation and one requires a small
allocation. As memory becomes available, the allocate call asking for a small
allocation returns first and unblocks the corresponding kernel.

In CRT, if the large allocation came first, CRT will wait until it can be
satisfied. It will not try satisfying a smaller allocation first. Also, because
CRT launches kernels synchronously, it can’t reorder kernels like TF does.

### CUDA Kernel Tracking

CUDA Kernel Tracking refers to the mechanism to track the buffers used by the
kernel and when the kernel completes in order to decrease necessary stream
synchronizations, and achieving higher GPU utilization.

This optimization is particularly useful in TF because without it, the only way
to use a newly allocated buffer on a non-compute stream is to wait until all
launched kernels actually complete on the compute stream.

With the stream-aware allocator approach used by PyTorch, we can request a
buffer that is safe to use on any stream in a JUQ manner. Kernel tracking can be
useful to further optimize a stream-aware allocator, especially in cases where
memory cannot be statically partitioned across streams in steady state. For
example, PyTorch allocator synchronizes to the top of all non-primary streams on
which the buffer was ever used before actually marking the buffer as available.
With kernel tracking, one can synchronize to the correct (or close to correct)
point in each non-primary stream. We don’t plan to utilize kernel tracking in
the short term in CRT.

### Temporary Memory Allocation

Shape inference can return output shapes given the operation and input shapes.
Knowing input and output shapes allows the compiler to allocate memory outside
of the operation and feed in ready buffers. Unfortunately, input and output is
not the only memory that can be used by a kernel. Some kernels also require a
buffer for storing intermediary computation results. Moreover, the necessary
temporary buffer size is **kernel** dependent. In other words, the same
operation can require different amounts of temporary storage depending on which
kernel has been chosen to implement the operation.

Ideally, shape inference mechanisms should be extendable to cover “temporary
buffer size inference” use cases. Another alternative is to let each kernel
implement a C++ “temporary buffer size inference” function that CRT can call.
The worst case would be to allow kernels to allocate memory internally. Such
kernels would need to be launched on a separate thread (because allocations can
block) and incur the overhead or raise an OOM error from allocation request
instead of blocking.

## Buffers and Tensors

Regular CUDA APIs operate on raw addresses - `CUdeviceptr`. In CUDA TFRT, we
have two main types to represent memory.

**Buffer** is the type that is produced by the `cuda.mem.allocate`. Besides the
raw pointer, it also includes the size and the allocator pointer. Neither of
these probably needs to be exposed through CRT kernels. We track the size
because passing the size to `Allocator::Deallocate` enables significantly more
efficient algorithms. Buffer includes `Allocator*` so that the destructor can
deallocate the memory. One might wonder why we are not calling some global
`GetCUDAAllocator()` function to get the allocator. First, mapping the pointer
to the device and obtaining the corresponding context and allocator is overhead
and requires us to maintain these mappings. Second, having an allocator pointer
gives flexibility. For example, we represent a buffer not owning the memory with
a null allocator. We can also imagine supporting sub-allocators or arena
allocators in the future. Finally, the extra fields should cause minimal
overhead because Buffers will be contained in heap-allocated
`AsyncValue<Buffer>` and everyone just passes pointers to these AsyncValues
around.

Buffers returned from `cuda.mem.allocate` are the sole owner of the memory. View
buffers, buffers not owning their memory, can be created from solo owner
buffers. We don’t support shared memory ownership at this time, but can add it
if/when a need arises. Here is an example of how a buffer class can support all
ownership types.

**Tensors** are Buffers plus dtype, shape, and layout (only dense raw-major
layout is currently supported). CUDA kernels that are not specialized to
particular input shapes and/or dtypes (especially hand-written ones) want to
dynamically dispatch based on these attributes. These kernels need to take
Tensors as arguments. Tensors are created from buffers. When a tensor is created
it takes over the buffer - no operations can be performed on the buffer after it
has been consumed by the tensor. If the buffer was owning its memory, the
resulting tensor takes over the memory ownership. Otherwise, the resulting
tensor does not own its memory and it is the user’s responsibility to make sure
the owning buffer outlives the tensor.

### Memory Ownership and Types

As described above in the current implementation, there is a single Buffer and a
single DenseTensor class to represent objects that own memory and objects that
don’t. In other words, our Buffer can be a `void*` as well as an
`std::unique_ptr<void*>`. This is convenient since all current use cases don’t
care about memory lifetime, i.e. as long as the memory survives until the end of
the function, they are fine. Having single C++ types also allows us to have a
single MLIR type and not template the kernels. We can revisit this design
if/when more complex use cases arise.

## Memory Safety

As discussed above, CRT will support the JUQ memory lifetype model for all
streams. In other words, the user can deallocate the buffer as soon as all the
kernels using it have been launched. Unlike TF, this usage model is valid for
all streams, i.e. independently of which stream the buffer was associated with
when it was allocated.

Ensuring standard memory safety when the same buffer is used on multiple streams
is the user's responsibility. In other words, if the user writes a buffer on
stream 0 and reads it on stream 1, the user must call the appropriate CRT
kernels to ensure that the read happens after the write. CRT will not
automatically insert such synchronizations.

Importantly, unlike regular multi-threaded programming, CRT requires that the
user inserts stream synchronization not only for read-after-write andself.lines
write-after-read, but also for parallel **first** reads or parallel **first**
writes of a buffer on non-primary streams. There should be no good reasons for
such programs to exist, but programmers would likely expect that a program
containing unsynchronized first reads or writes will simply read some garbage
twice or write something twice. Unfortunately, in a multi-stream CUDA
environment, such programs can actually cause undefined behavior. AFAICT,
PyTorch does not automatically deal with these issues or document them.

Finally, the aforementioned synchronizations don’t deal with safety around
buffer reuse, i.e. returning a buffer from allocate that was recently used on
multiple streams and freed. CRT will automatically ensure safety in these cases
by inserting proper stream synchronizations. To start with we can adopt the
PyTorch strategy (synchronize all non-primary streams before marking a freed
buffer as available for reuse) and improve upon it with more advanced allocation
algorithms and kernel tracking if/when necessary.

## Misc

### Returning Errors from CUDA Kernels

Some GPU kernels today can encounter an error on the GPU. The famous example of
this is [tf.gather](https://www.tensorflow.org/api_docs/python/tf/gather). Its
documentation states: “On GPU, if an out of bound index is found, a 0 is stored
in the corresponding output value”. This “feature” has most likely wasted days
or weeks of debugging time. It would be nice if CRT can provide a way for GPU
kernels to return errors.

Here are some options:

*   Can write some error code to pinned memory. Each future GPU kernel can check
    if the error code set, and if so, do not execute (although for cuDNN
    kernels, etc, we cannot have the check). CPU can check error code
    periodically (e.g. when a tensor transferred to CPU), raise sensible error
    message

*   Instead of having each GPU kernel check, can have kernel with error just
    return dummy data as it is done today

*   Can `assert(false)` inside the kernel since latest CUDA versions support
    this. But this causes every single host-side CUDA function to return an
    error, perhaps with an error code that doesn't make it clear a failed
    assertion occurs. Would be difficult to raise sensible error message

    *   It’s unclear that the data written on a pinned memory followed by a
        assert(false) is visible on the host. So it may be impossible to raise a
        sensible error message with this approach.

*   Can always keep input of such tfrt kernels on the host. Either we can
    require such kernels are always executed on host, or only have a host-only
    kernel that checks for an error, before running the normal op on the GPU

Having a reserved global buffer for an error message seems like the best option.
The buffer’s format can be something like &lt;is_filled_boolean>&lt;human
readable error message>. Kernels that can produce an error can take this buffer
pointer as an argument. If the error is already filled, the kernel would not
override it, but maybe append to it. We can also add space for an opaque
token/void* to the error message. This can allow more effective symbolization in
case of errors by matching up to something persisted higher in the stack (e.g. a
stack trace).

Always copying the error out might have an overhead. Instead, we can introduce a
debugging mode for CRT (and TFRT overall). In this mode, we can execute all
kernels synchronously and copy the error string to the CPU after each kernel
that takes it.

<!-- Footnotes themselves at the bottom. -->

## Notes

[^1]: The C++ code implementing the dialect is referred to as just CRT.
[^2]: When users train ML models, if extra memory is available, they just
    increase the batch size.
[^3]: CUDA has stream priorities and `tf.data` could have used a low priority
    stream to make sure that prefetching transfers don’t block transfers of
    the current step. Unfortunately, CUDA docs say that "stream priorities
    have no effect on host-to-device and device-to-host memory operations."
[^4]: Of course, there are models where a single stream is too restrictive.
[^5]: TF actually has options called `allow_growth` and
    `TF_ENABLE_GPU_GARBAGE_COLLECTION`. If both of these are turned on, TF
    essentially behaves similarly to PyTorch.
[^6]: JUQ - Just-Until-Queued.
[^7]: There are other trade-offs like on-demand allocation can play better for
    multi-tenant use cases. Though it has not been a huge problem because
    users can just limit the memory used by TF or set `allow_growth=true`
    option.
[^8]: A recent extreme example of this is b/138121009. TF OOMs while trying to
    allocate 1.2 GB even when the GPU has 10 GB available, due to
    fragmentation.
[^9]: Variables are created in the very beginning and live for the whole
    duration of the program. All other tensors are short-lived. GPU kernels
    are small compared to fully unrolled TPU programs and don’t require very
    large contiguous memory blocks for the “stack”.
[^10]: GC'ed languages tend to have a "GC stress mode" to help flush out bugs
    like this (in modern terminology we'd call this a sanitizer). In this
    mode the GC relocates all pointers continuously and this continuous
    relocation flushes out these "raw pointer" bugs.
[^11]: By “satisfying an allocation” we mean obtaining the actual address into
    GPU virtual memory.
[^12]: This might be unavoidable in some cases, e.g. legacy kernels, but we hope
    to keep those kernels to a minimum. For example, for kernels like
    `tf.where`, where the output size data dependent, we can split them up
    into two kernels will a dynamic `cuda.mem.allocate` in between.
[^13]: Each GPU has an associated cuContext that holds all the CUDA state for
    this GPU. It seems like cuContext has a mutex that is locked on most
    accesses. When multiple threads call into one cuContext they contend on
    this lock.
[^14]: Serializability would be very nice for runtime debugging. If all BEF
    functions were serializable, a person hitting a runtime issue could
    simply send us the BEF file to reproduce the issue.
