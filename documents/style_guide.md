# TFRT C++ style guide

<!--* freshness: {
  owner: 'pgavin'
  owner: 'zhangqiaorjc'
  reviewed: '2019-12-12'
} *-->

[TOC]

TFRT follows the
[Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
unless an exception is listed below.

## Relative include paths

Unlike the Google style guide, which recommends include path relative to project
source directory, we require include path relative to an `include/` directory
explicitly specified in the BUILD system (see `tfrt_cc_library` Bazel
definition).

For example:

[`include/tfrt/bef_executor/bef_file.h`](https://cs.opensource.google/tensorflow/tensorflow/+/master:include/tfrt/bef_executor/bef_file.h)

should be included as:

`#include "tfrt/bef_executor/bef_file.h"`

See context:
https://google.github.io/styleguide/cppguide.html#Names_and_Order_of_Includes
https://docs.bazel.build/versions/master/bazel-and-cpp.html#include-paths.

## Forward Declarations

Unlike the Google style guide, our style prefers forward declarations to
`#includes` where possible. This can reduce compile times and result in fewer
files needing recompilation when a header changes.

You can and should use forward declarations for most types passed or returned by
reference, pointer, or types stored as pointer members or in most STL
containers. However, if it would otherwise make sense to use a type as a member
by-value, don't convert it to a pointer just to be able to forward-declare the
type.

## Includes

You should include all the headers that define the symbols you rely upon, except
in the case of forward declaration. If you rely on symbols from `bar.h`, don't
count on the fact that you included `foo.h` which (currently) includes `bar.h`:
include `bar.h` yourself, *unless `foo.h` explicitly demonstrates its intent to
provide you the symbols of `bar.h`*.

*Example*

*   Even though `async_value.h` includes `<string>`, since `<string>` is not
    inherently part of the AsyncValue concept, the clients that include
    `async_value.h` should include `<string>` if they use `std::string`.

*   Since `async_value_ref.h` includes `async_value.h` and `async_value.h` is
    inherently part of the AsyncValueRef concept, the clients that include
    `async_value_ref.h` should not need to include `async_value.h`.

## File headers

All source files start with a file header. The first and the last line of the
file header must have exactly 80 characters. That header should look like this:

```c++
//===- HostAllocator.h - Host Memory Allocator Abstraction ------*- C++ -*-===//
//
// This file declares the generic interface for host memory allocators.
//
//===----------------------------------------------------------------------===//
```

## Capitalization for acronyms

You should capitialize the first letter for acronyms. In particular, a
`tfrt::OpHandler` name should be in PascalCase and any acronyms inside should
have only its first letter capitalized, e.g.:

``` {.good}
CpuOpHandler
GpuOpHandler
```

instead of:

``` {.bad}
CPUOpHandler
GPUOpHandler
```

TODO(pgavin): incorporate contents from pgavin@ Google doc.

## Tags for the future work

Use `TODO` to tag code that is temporary or a short-term solution. Tasks marked
as `TODO` are considered tech debt and should be fixed sooner than later.

Use `IDEA` to tag code that is not required to address in the near future. This
includes ideas for further performance optimization or features to support more
use-cases.

See below for example tag usages.

```c++
// TODO: Deprecate the "Last visitors" feature.
// IDEA: We can optimize performance by reducing memory copy here.
```

## CL Descriptions

### Start with WHY

CL description should not just describe the WHAT and HOW, when the motivation of
the change is not yet clear. In that case, please always start with WHY, while
still keeping the first line short. Some good examples include:

``` {.good}
Reduced async op launch overhead by 10% by removing a data copy.
```

No need to motivate this if the change is simple. If it incurs a lot more code
complexity, then please do justify why the performance improvement is worth it.

``` {.good}
Extended GPU runtime with async execution support.
```

Okay to not describe why this matters, if we believe relevant readers already
know it or can easily find it out.

``` {.good}
To achieve <describe your goal>, extended threadpool with ...
```

If the goal is well established, a single sentence like this will do. If more
context is needed, please elaborate on it later in the CL description, or link
to a doc.

A bad example looks like:

``` {.bad}
Changed the default value of FOO from 1 to 2.
```

It is not clear why 2 is better than 1, or if there's an even better option.
