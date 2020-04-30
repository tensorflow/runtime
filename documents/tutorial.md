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
  %chain = hex.new.chain

  // Create a string containing "hello world" and store it in %hello.
  %hello = "tfrt_test.get_string"() { value = "hello world" } : () -> !hex.string

  // Print the string in %hello.
  "tfrt_test.print_string"(%hello, %chain) : (!hex.string, !hex.chain) -> !hex.chain

  hex.return
}
```

The `@hello` function above shows how to create and print a string. The text
after each `:` specifies the types involved:

-   `() -> !hex.string` means that `tfrt_test.get_string` takes no arguments and
    returns a `!hex.string`. `hex` stands for "host executor", a name which we
    might revisit in future design.
-   `(!hex.string, !hex.chain) -> !hex.chain` means that
    `tfrt_test.print_string` takes two arguments (`!hex.string` and
    `!hex.chain`) and returns a `!hex.chain`. `chain` is a TFRT abstraction to
    manage dependencies; see [explicit_dependency.md](explicit_dependency.md).

`tfrt_test.get_string`'s `value` is an *attribute*, not an argument. Attributes
are compile-time constants, while arguments are only available at runtime upon
kernel/function invocation.

This example code ignores the `!hex.chain` returned by `tfrt_test.print_string`.

Translate `hello.mlir` to [BEF](binary_executable_format.md) by running
`tfrt_translate --mlir_to_bef`:

```shell
$ bazel-bin/tools/tfrt_translate --mlir-to-bef hello.mlir > hello.bef
```

You can dump the encoded BEF file, and see that it contains the `hello world`
string attribute:

```shell
$ hexdump -C hello.bef
00000000  0b ef 00 02 00 01 16 68  65 6c 6c 6f 2e 6d 6c 69  |.......hello.mli|
00000010  72 00 02 18 00 02 0c 00  05 0c 00 08 03 00 01 01  |r...............|
00000020  03 81 3a 21 68 65 78 2e  63 68 61 69 6e 00 21 68  |..:!hex.chain.!h|
00000030  65 78 2e 73 74 72 69 6e  67 00 68 65 6c 6c 6f 00  |ex.string.hello.|
00000040  68 65 78 2e 6e 65 77 2e  63 68 61 69 6e 00 74 66  |hex.new.chain.tf|
00000050  72 74 5f 74 65 73 74 2e  67 65 74 5f 73 74 72 69  |rt_test.get_stri|
00000060  6e 67 00 74 66 72 74 5f  74 65 73 74 2e 70 72 69  |ng.tfrt_test.pri|
00000070  6e 74 5f 73 74 72 69 6e  67 00 76 61 6c 75 65 00  |nt_string.value.|
00000080  04 18 0b 68 65 6c 6c 6f  20 77 6f 72 6c 64 05 08  |...hello world..|
...
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
  %chain = hex.new.chain

  // Create an integer containing 42.
  %forty_two = hex.constant.i32 42

  // Print 42.
  hex.print.i32 %forty_two, %chain

  hex.return
}
```

`@hello_integers` shows how to create and print integers. This example does not
have the verbose type information we saw in `@hello` because we've defined
custom parsers for the `hex.constant.i32` and `hex.print.32` kernels in
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
  %chain = hex.new.chain

  %two = hex.constant.i32 2
  %four = hex.constant.i32 4

  %coordinate = "my.create_coordinate"(%two, %four) : (i32, i32) -> !my.coordinate

  "my.print_coordinate"(%coordinate, %chain) : (!my.coordinate, !hex.chain) -> !hex.chain

  hex.return
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

Also, see [TFRT Op-by-op Execution Design](tfrt_op_by_op_execution_design.md) on
how TFRT will support eagerly executing TensorFlow ops.
