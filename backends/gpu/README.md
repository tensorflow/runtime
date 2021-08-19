# TFRT GPU backend

This document gives a brief overview of the CUDA and ROCm backend of the new
Runtime for TensorFlow, intended as a starting point for code contributors.
Concrete APIs and implementation details are documented in the code itself.

## Code organiziation

The C++ code is split into two main directories: `include/` for public headers,
and `lib/` for implementation and private headers. The `tools/` directory
contains the source of binaries used for building and testing. The tests are
split across `cpp_test/` using GoogleTest, and `mlir_tests/` using LLVM's lit
test infrastructure.

The TFRT GPU backend consists of multiple layers. The following sections give a
brief overview, roughly from lower to higher level.

## Low Level GPU runtime wrappers

The TFRT GPU backend supports both NVIDIA GPUs (through CUDA) and AMD GPUs
(through ROCm) in the same binary.

All vendor libraries (cuDNN, MIOpen, etc.) are dynamically loaded at runtime.
The dynamic loading code is checked in (for AMD libraries) or generated at build
time (for NVIDIA libraries) from the C API headers using the
`tools/stub_codegen` python tool.

The vendor API types are wrapped in disciminated unions
(`wrapper::Resource<...>`), which are further wrapped in `std::unique_ptr` for
RAII safety.

Free functions are used to add better error handling (`llvm::Error/Expected`)
and C++ ergonomics to the vendor API functions (see e.g.
`wrapper::CuCtxCreate()`). In case where both vendors provide an identical API,
we provide a function that forwards to the appropriate API based on the platform
of the arguments (e.g. `wrapper::CtxCreate()`). We don't gloss over API
differences here, but expect higher level code to be mostly vendor agnostic
interleaved with a few vendor-specific sections.

## BEF-level kernels and types

The TFRT kernels (not to be confused with GPU device kernels) are functions that
are linked into the `bef_executor` and referenced from the BEF file by their
name. The kernels functions consume arguments and produce results of
`AsyncValueRef<T>`s, which are ref-counted futures. The kernels can be
implemented using conrete types, and kernel registration will take care of
translating them from and to async values. The concrete TFRT GPU types wrap the
low level RAII types (e.g. the `GpuContext` is an aggregate of
`wrapper::Context`). Some types acquire a ref-count to other underlying
resources (e.g. a `GpuStream` holds a ref-count to `GpuContext`). The
`GpuContext` contains resource pools and caches for types that are expensive to
create (e.g. `GpuModule`).

## ODS ops and rewrite patterns

The BEF kernels and types have a corresponding
[ODS](http://mlir.llvm.org/docs/OpDefinitions) definition (see `gpu_ops.td`).
The `mlir-tblgen` tool generates C++ classes from these definitions which the
compiler uses in the AST representation of a program.

The `tfrt_gpu_translate` tool parses an assembly program and produces the
corresponding BEF file.

TFRT GPU provides two
[rewrite patterns](https://mlir.llvm.org/docs/PatternRewriter/). They are used
by XLA:GPU to prepare a program for MLIR's `gpu-async-region` pass, and lower
the program further to use TFRT streams and events for asynchronous execution.
The `tfrt_gpu_opt` tool provides passes to test the rewrite patterns.

## TensorFlow eager execution

TFRT GPU contains kernels for TensorFlow eager ops. They are backed by (e.g.
cuDNN) library calls and handwritten CUDA device code. Note that the rest of
TFRT GPU (including BEF kernels) do not require a device compiler.

The `GpuDevice` contains a context, allocator and various usage-specific streams
(compute, h2d, d2h, d2d etc). It is referenced by the `GpuOpHandler`, which is
created to execute CoreRT kernels. Conversion functions support data transfer by
calling the memcpy functions from the TFRT device APIs.
