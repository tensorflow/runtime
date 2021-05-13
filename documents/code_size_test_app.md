# Code Size Test App

<!--* freshness: {
  owner: 'chuanhao'
  reviewed: '2021-05-13'
} *-->

<!-- TOC -->

The goal of this test app is to track the binary size of a potential app built
on top of TFRT. Such apps need to link the core components of TFRT libraries
including `:hostcontext`, `:support`, `:befexecutor` and various kernel
libraries.

This document describes the test app setup and explains various techniques we
used to reduce the binary size, and also documents tools that we have found
useful for analyzing binary size.

## Test App Setup

This test app computes the Fibonacci sequence. Today, the application logic
needs to be written in MLIR, and converted to BEF format using `tfrt_translate
-mlir-to-bef` tool, before getting run by a simplified `BEF executor` We also
need to link `basic_kernels` library for the kernel implementations to compute
Fibonacci sequence.

With the TFRT Host Runtime APIs, a potential application can be built on top of
C++ API directly without going through MLIR to BEF conversion.

To run the test:

```shell
$ bazel test //mlir_tests/code_size_test_app:fib.mlir.test
```

What really happens can be inferred from the first line of
[`fib.mlir`](https://github.com/tensorflow/runtime/blob/master/mlir_tests/code_size_test_app/fib.mlir):

```c++
// RUN: tfrt_translate -mlir-to-bef %s | code_size_test_driver | FileCheck %s
```

-   `tfrt_translate -mlir-to-bef` converts `fib.mlir` to a BEF file.
-   `code_size_test_driver` reads the BEF file from `stdin` and runs it.
-   the output of the test is `FileCheck`-ed against expected results annotated
    in `fib.mlir`.

## Techniques to Reduce Binary Size

As of March 31 2020, the `code_size_test_driver` minimal binary size is 78KB. We
statically link core TFRT libraries, the required kernel libraries and
`llvm::support` library, while dynamically link the C++ runtime library and
system libraries. If we also statically link the C++ runtime, as done by default
in Bazel, we get 405KB.

```shell
$ bazel build --config=code_size_test
    third_party/tf_runtime:code_size_test_driver
```

The above configuration applies a few well-known techniques in mobile
app/embedded world to reduce the binary size:

### Disable RTTI and exceptions

We use a `rtti_and_exceptions` Skylark flag to disable RTTI and exceptions
through the corresponding copts `-fno-rtti` and `-fno-exceptions`. As a side
effect, we also need to disable header modules.

### Strip symbols

We pass linker options `--linkopt=-Wl,--strip-all` to strip all symbols.

### Ask compiler to optimize for binary size.

We ask compiler to optimize for size by specifying `--copt=-Os`, where `-Os`
enables all `-O2` optimizations except those that often increase code size.

### Safe ICF

We ask compiler to merge identical functions into a single copy whenever safe to
do so with `--linkopt=-Wl,--icf=safe`.

## Tools for Analyzing Binary Size

*   `llvm-nm`: Dump all the symbols.

```shell
$ llvm-nm --demangle --print-size --radix=d --size-sort --reverse-sort nonstripped_binary | head -n 20

0000000000006208 0000000000002011 T main
0000000000005728 0000000000000308 T ReadFromStdInToBuffer()
0000000000006048 0000000000000146 W std::__u::vector<unsigned char, std::__u::allocator<unsigned char> >::reserve(unsigned long)
0000000000008288 0000000000000104 W llvm::SmallVectorImpl<tfrt::AsyncValue*>::assign(unsigned long, tfrt::AsyncValue* const&)
0000000000008416 0000000000000103 T __libc_csu_init
0000000000005376 0000000000000042 T _start
0000000000004808 0000000000000032 r std::__u::__function::__policy const* std::__u::__function::__policy::__choose_policy<std::__u::__function::__default_alloc_func<main::$_0, void (tfrt::DecodedDiagnostic const&)> >(std::__u::integral_constant<bool, true>)::__policy_
0000000000004840 0000000000000032 r std::__u::__function::__policy const* std::__u::__function::__policy::__choose_policy<std::__u::__function::__default_alloc_func<main::$_0, void (tfrt::DecodedDiagnostic)> >(std::__u::integral_constant<bool, true>)::__policy_
0000000000016672 0000000000000008 B stdout
0000000000016664 0000000000000008 b dtor_idx.6689
0000000000016656 0000000000000001 b completed.6687
0000000000008528 0000000000000001 T __libc_csu_fini
0000000000008224 0000000000000001 t void std::__u::__function::__policy_invoker<void (tfrt::DecodedDiagnostic const&)>::__call_impl<std::__u::__function::__default_alloc_func<main::$_0, void (tfrt::DecodedDiagnostic const&)> >(std::__u::__function::__policy_storage const*, tfrt::DecodedDiagnostic const&)
0000000000008256 0000000000000001 t void std::__u::__function::__policy_invoker<void (tfrt::DecodedDiagnostic)>::__call_impl<std::__u::__function::__default_alloc_func<main::$_0, void (tfrt::DecodedDiagnostic)> >(std::__u::__function::__policy_storage const*, tfrt::DecodedDiagnostic&&)
                                  U strlen
0000000000005488 0000000000000000 t register_tm_clones
                                  U puts
                                  U putchar
                                  U printf
                                  U memcpy
```
