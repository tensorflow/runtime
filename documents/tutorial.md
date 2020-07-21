# TFRT Tutorial

<!--* freshness: {
  owner: 'lauj'
  reviewed: '2020-04-28'
} *-->

<!-- TOC -->

This document shows how to run some simple example code with TFRT's BEFExecutor.

This document assumes you've installed TFRT and its prerequisites as described
in
[the README](https://github.com/tensorflow/runtime/blob/master/README.md#getting-started).

## Hello World

Create a file called `hello.mlir` with the following content:

```c++
func @hello() {
  %chain = tfrt.new.chain

  // Create a string containing "hello world" and store it in %hello.
  %hello = "tfrt_test.get_string"() { string_attr = "hello world" } : () -> !tfrt.string

  // Print the string in %hello.
  "tfrt_test.print_string"(%hello, %chain) : (!tfrt.string, !tfrt.chain) -> !tfrt.chain

  tfrt.return
}
```

The `@hello` function above shows how to create and print a string. The text
after each `:` specifies the types involved:

-   `() -> !tfrt.string` means that `tfrt_test.get_string` takes no arguments
    and returns a `!tfrt.string`. `tfrt` is a
    [MLIR dialect](https://mlir.llvm.org/docs/LangRef/#dialects) prefix (or
    namespace) for TFRT.
-   `(!tfrt.string, !tfrt.chain) -> !tfrt.chain` means that
    `tfrt_test.print_string` takes two arguments (`!tfrt.string` and
    `!tfrt.chain`) and returns a `!tfrt.chain`. `chain` is a TFRT abstraction to
    manage dependencies. For detailed explanation, see the
    [Explicit Dependency Management in TFRT documentation](explicit_dependency.md).

`tfrt_test.get_string`'s `string_attr` is an *attribute*, not an *argument*.
Attributes are compile-time constants, while arguments are only available at
runtime upon kernel/function invocation. In the above example, the `string_attr`
attribute has the value `hello world`.

`tfrt.return` is a special form that specifies the function's return values,
similar to a C++ `return` statement. In the above case, the function `@hello`
does not have a return value. For detailed explanation and more examples, refer
to the
[TFRT Host Runtime Design documentation](tfrt_host_runtime_design.md#tfrt_return).

This example code ignores the `!tfrt.chain` returned by
`tfrt_test.print_string`.

Translate `hello.mlir` to [BEF](binary_executable_format.md) by running
`tfrt_translate --mlir_to_bef`:

```shell
$ bazel-bin/tools/tfrt_translate --mlir-to-bef hello.mlir > hello.bef
```

You can dump the encoded BEF file, and see that it contains the `hello world`
string attribute:

```shell
$ hexdump -C hello.bef
```

Run `hello.bef` with `bef_executor` to see it print `hello world`:

```shell
$ bazel-bin/tools/bef_executor hello.bef
Choosing memory leak check allocator.
Choosing single-threaded work queue.
--- Running 'hello':
string = hello world
```

The first two `Choosing` lines are `bef_executor` explaining which
implementations of
[HostAllocator](https://github.com/tensorflow/runtime/blob/master/include/tfrt/host_context/host_allocator.h)
and
[ConcurrentWorkQueue](https://github.com/tensorflow/runtime/blob/master/include/tfrt/host_context/concurrent_work_queue.h)
it's using. The third `--- Running 'hello':` line is printed by `bef_executor`
to show which MLIR function is currently executing (`@hello` in this case). The
fourth `string = hello world` line is printed by `tfrt_test.print_string`, as
requested by `hello.mlir`.

## Hello Integers

`bef_executor` runs all functions defined in the `.mlir` file that accept no
arguments. We can add another function to `hello.mlir` by appending the
following to `hello.mlir`:

```c++
func @hello_integers() {
  %chain = tfrt.new.chain

  // Create an integer containing 42.
  %forty_two = tfrt.constant.i32 42

  // Print 42.
  tfrt.print.i32 %forty_two, %chain

  tfrt.return
}
```

`@hello_integers` shows how to create and print integers. This example does not
have the verbose type information we saw in `@hello` because we've defined
custom parsers for the `tfrt.constant.i32` and `tfrt.print.32` kernels in
[basic_kernels.td](https://github.com/tensorflow/runtime/blob/master/include/tfrt/basic_kernels/opdefs/basic_kernels.td).
See MLIR's
[Operation Definition Specification (ODS)](https://mlir.llvm.org/docs/OpDefinitions/)
for more information on how this works.

If we run `tfrt_translate` and `bef_executor` over `hello.mlir` again, we see
that the executor calls our second function in addition to the first:

```shell
$ bazel-bin/tools/tfrt_translate --mlir-to-bef hello.mlir > hello.bef
$ bazel-bin/tools/bef_executor hello.bef
Choosing memory leak check allocator.
Choosing single-threaded work queue.
--- Running 'hello':
string = hello world
--- Running hello_integers:
int32 = 42
```

## Defining Kernels

Let's define some custom kernels that manipulate *(x, y)* coordinate pairs.
Create `lib/test_kernels/my_kernels.cc` containing the following:

```c++
#include <cstdio>

#include "tfrt/host_context/chain.h"
#include "tfrt/host_context/kernel_registry.h"
#include "tfrt/host_context/kernel_utils.h"

namespace tfrt {
namespace {

struct Coordinate {
  int32_t x = 0;
  int32_t y = 0;
};

static Coordinate CreateCoordinate(int32_t x, int32_t y) {
  return Coordinate{x, y};
}

static Chain PrintCoordinate(Coordinate coordinate) {
  printf("(%d, %d)\n", coordinate.x, coordinate.y);
  return Chain();
}
}  // namespace

void RegisterMyKernels(KernelRegistry* registry) {
  registry->AddKernel("my.create_coordinate",
                      TFRT_KERNEL(CreateCoordinate));
  registry->AddKernel("my.print_coordinate",
                      TFRT_KERNEL(PrintCoordinate));
}

}  // namespace tfrt
```

Edit `include/tfrt/test_kernels.h` to forward declare `RegisterMyKernels`:

```c++
// Lots of existing forward declarations here...
void RegisterMyKernels(KernelRegistry* registry);  // <-- ADD THIS LINE
```

Also edit `lib/test_kernels/static_registration.cc`, updating
`RegisterExampleKernels` to call `RegisterMyKernels`:

```c++
static void RegisterExampleKernels(KernelRegistry* registry) {
  // Lots of existing registrations here...
  RegisterMyKernels(registry);  // <-- ADD THIS LINE
}
```

Finally, edit the definition of `test_kernels` in the top level `BUILD` file, to
add `lib/test_kernels/my_kernels.cc` to `srcs`:

```python
tfrt_cc_library(
    name = "test_kernels",
    srcs = [
        # Lots of existing srcs here ...
        "lib/test_kernels/my_kernels.cc",  # <-- ADD THIS LINE
    ],
```

Now we can rebuild `bef_executor` to compile and link with our new kernels:

```shell
$ bazel build -c opt //tools:bef_executor
```

With that done, we can write a `coordinate.mlir` program that calls our new
kernels:

```c++
func @print_coordinate() {
  %chain = tfrt.new.chain

  %two = tfrt.constant.i32 2
  %four = tfrt.constant.i32 4

  %coordinate = "my.create_coordinate"(%two, %four) : (i32, i32) -> !my.coordinate

  "my.print_coordinate"(%coordinate, %chain) : (!my.coordinate, !tfrt.chain) -> !tfrt.chain

  tfrt.return
}
```

MLIR types that begin with `!` are user-defined types like `!my.coordinate`,
compared to built-in types like `i32`. User-defined types do not need to be
registered with TFRT, so we do not need to rebuild `tfrt_translate`:
`tfrt_translate --mlir_to_bef` is a generic compiler transformation.

So now we can compile and run `coordinate.mlir`:

```shell
$ bazel-bin/tools/tfrt_translate --mlir-to-bef coordinate.mlir > coordinate.bef
$ bazel-bin/tools/bef_executor coordinate.bef
Choosing memory leak check allocator.
Choosing single-threaded work queue.
--- Running 'print_coordinate':
(2, 4)
```

`coordinate.mlir` shows several TFRT features:

-   Kernels are just C++ functions with a name in MLIR: `my.print_coordinate` is
    the MLIR name for the C++ `PrintCoordinate` function.
-   Kernels may pass arbitrary user-defined types: `my.create_coordinate` passes
    a custom `Coordinate` struct to `my.print_coordinate`.

## Under Construction!

This tutorial is a work in progress. We hope to add more tutorials for topics
like:

-   Asynchronous execution
-   Control flow
-   Non-strict execution

## What's Next

Note in order to use TFRT, we do not expect TensorFlow end users to hand-write
the MLIR programs as shown above. Instead, we are building a graph compiler that
will generate such MLIR programs from TensorFlow functions created from
TensorFlow model code.

Next, see [TFRT Host Runtime Design](tfrt_host_runtime_design.md) for detailed
explanation on TFRT concepts including `AsyncValue`, `Kernel`, and `Graph
Execution` etc. Also, see
[TFRT Op-by-op Execution Design](tfrt_op_by_op_execution_design.md) on how TFRT
will support eagerly executing TensorFlow ops.
