# TFRT C++ style guide

<!--* freshness: {
  owner: 'pgavin'
  owner: 'zhangqiaorjc'
  reviewed: '2020-07-21'
  review_interval: '1 year'
} *-->

<!-- TOC -->

TFRT follows the
[Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
unless an exception is listed below.

## Relative include paths

Unlike the Google style guide, which recommends include path relative to project
source directory, we require include path relative to an `include/` directory
explicitly specified in the BUILD system (see `tfrt_cc_library` Bazel
definition).

For example:

[`include/tfrt/bef_executor/bef_file.h`](https://github.com/tensorflow/runtime/blob/master/include/tfrt/bef_executor/bef_file.h)

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

## Pass efficiently movable types by value

When a function needs to sink a parameter (e.g. storing the parameter in an
object, as opposed to just using the parameter), we prefer using pass by value
over pass by rvalue reference for the parameter, if the parameter is an
efficiently movable type.

``` {.good}
class Foo {
 public:
  // The parameter `bar` is stored internally in the constructor and is
  // efficiently movable, so we pass `bar` by value.
  explicit Foo(std::string bar) : bar_{std::move(bar)} {}

 private:
  std::string bar_;
};
```

instead of,

``` {.bad}
class Foo {
 public:
  // Bad exmample. We prefer not to pass `bar` using rvalue reference.
  explicit Foo(std::string&& bar) : bar_{std::move(bar)} {}

 private:
  std::string bar_;
};

class Bar {
 public:
  // Bad exmample. std::array<int, 100000> is a movable type, but it is
  // inefficient to move, so we should not pass it by value. We should pass it
  // by const reference instead.
  explicit Bar(std::array<int, 100000> arr) : arr_{arr} {}

 private:
  std::array<int, 100000> arr_;
};
```

Pass-by-value is preferred because:

*   The code is less cluttered.
*   For copyable types, it handles both lvalue and rvalue as the argument value.

``` {.bad}
class Foo {
 public:
  explicit Foo(std::string&& bar) : bar_{std::move(bar)} {}

 private:
  std::string bar_;
};

void ClientCode(const std::string& s) {
  // This will not work as the constructor for foo only takes an
  // rvalue reference.
  Foo foo(s);
  // ...
}
```

Note that using the rvalue reference sometimes gives better performance, but as
always, make sure you have evidence that it helps before writing more
complicated code for the sake of performance.

Please see
[Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html#Rvalue_references)
for a detailed discussion on using pass-by-value vs rvalue reference for
function parameters.

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

## Write small CLs

Note: CLs refer to changelists, and are also known as \"patches\" or \"pull
requests\".

Small CLs (measured by both # of lines and files changed) usually lead to
productive and high quality code reviews. The amount of review work tends to
increase "quadratically" with the size of the CL, because a larger CL will
likely involve more rounds of CL reviews, where each round takes longer.

Small CLs are also less likely to cause merge conflicts, and are simpler to roll
back when needed. They also make code archeology study (code reading after
submission) more productive.

Fundamentally, engineer's attention is a scarce resource, and smaller CLs tend
to leverage this resource more effectively.

Tip: To write smaller CLs, consider focusing on one change in each CL.

For example, separate refactoring from new feature development. The general
implementation strategy of sending NFCs (No Functionality Change) followed by
separate changes to add new features tend to make code writing, testing and code
review more productive. For example, NFCs in general do not require adding new
tests (unless test coverage is inadequate, in which case testing should be
enhanced before refactoring work). For more context on NFCs,
[see](https://twitter.com/clattner_llvm/status/1045715652537868289)
[these](https://twitter.com/clattner_llvm/status/1045548372134846464)
[tweets](https://twitter.com/clattner_llvm/status/964206793885798400) by Chris
Lattner.

Question: What to do when a newly introduced function has a large number of
call-sites?

Consider the following possibilities:

1.  Only introduce the new function, and then integrate it into call-sites in a
    follow-up CL.

1.  In some cases people may want to assess the API design of the new feature in
    the context of call-sites. One option is to integrate with a few call-sites
    first and have it reviewed. If doing so does not make the codebase compile,
    consider still sending a preliminary/prototype CL (that might not compile)
    for early feedback. Once the key design points/risks are discussed and
    addressed, the author can then "mass migrate" other call-sites with less
    scrutiny required.

More generally, when a CL contains say 100 chunks of changes, where 10 out of
them are worth scrutiny, and the rest more mundane, it's in everyone's best
interest to trying to separate these 2 classes of changes, so that different
amounts of attention can be applied to reviewing them.

Please use your judgement. For example, lumping a few unrelated but tiny changes
into a single CL can be fine and productive.

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
