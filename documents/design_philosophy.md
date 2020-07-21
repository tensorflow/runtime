# TFRT Design Philosophy

<!--* freshness: {
  owner: 'hongm'
  owner: 'jingdong'
  reviewed: '2020-03-31'
} *-->

<!-- TOC -->

This is a living document that aims to capture the overall design of the `tfrt`
project, provides a place to capture the rationale for various decisions, and
aims to be a good reference for new contributors as well as users of the
libraries.

## What is `TFRT`?

This is a low level runtime for TensorFlow, with many goals:

-   Provide extremely high performance, efficient memory use, and focus on
    low-level efficiency.
-   Efficiently use modern CPU hardware, including multicore, NUMA, etc.
-   Strong support for heterogenous hardware (e.g. mobile phones).
-   Make it easy to plug in support for new accelerators into TensorFlow.
-   Provide extremely modular libraries, enabling products to be built out of
    them that use different slices of functionality (e.g. for mobile, or an
    inference server).
-   Be easily extensible, easily hackable, easy to develop, easy to understand
    and work with.
-   Follow modern open source development methodology.
-   Design for scale - we expect this to grow very large over time.
-   Enable progressive migration of functionality from the existing TensorFlow
    stack into a new design.

What do we mean by "low level" runtime? This library contains a significant
amount of compute and accelerator functionality, but does not include Python
bindings or other higher level APIs. Those will remain in the core TensorFlow
repository.

At the time of this writing, this runtime is quite small, but we expect it to
grow over time, including support for new hardware, new kernels, new APIs, and
new applications.

This runtime is co-designed with the MLIR compiler infrastructure project.
Several of its design points enable specific aspects of this design, and thus
the overall TFRT pipeline benefits from and depends heavily on MLIR and its
infrastructure. The design philosophies of TFRT are very similar to that
followed by the LLVM and MLIR communities.

## Guiding Philosophies

Here we describe a few of the high-level design principles, and unpack what that
means for how we build things and how things are structured. These principles
are generally independent of any individual library or tool in the project, but
we use a few examples to illustrate the points.

### Build the right thing

Build the right thing is more important than building something fast. We value
iteration, evaluation, and experimentation, but eschew writing large amounts of
throw-away code that "we'll come back to later". It is better to build things
deliberately "bottom up", so we can make sure the lower level abstractions are
right, and the higher level pieces can build on top of well considered
underlying layers.

We aim to have subsystems that work together and have clearly defined roles and
responsibilities. We do not want to see lots of internally defensive code, or
code that works around deficiencies in other subsystems. This has implications:
we need to spend a significant amount of time debating and iterating on design
approaches, and document the results.

### Design from first principles, and keep it simple

We take the time to learn the requirements/history/lessons from the current TF
stack and other relevant systems, and avoid gratuitous deviation and reinventing
wheels.

At the same time, we try and not get constrained by the existing design and code
complexity. For example, we will not automatically reimplement all existing
[grappler passes](https://www.tensorflow.org/guide/graph_optimization) for
TFRT's graph compiler. Instead, we figure out the problems the existing features
tackle, validate that they are still worth solving, and then find the best ways
to solve them.

In short, we derive designs and solutions from requirements, and not directly
from existing solutions.

We acknowledge the human tendency of taking the "path of least resistance". For
example, the temptation to figure out a "local solution" that works for one
accelerator device, without consideration / coordination on the "global design"
with other device types. Even though this could be expedient in the short term,
we caution that this type of local thinking and approach will tend to add
extraneous system complexity over time, increasing the risks of causing the
system to collapse under its weight.

At the same time, we strike a balance in making continued incremental progress
and not biting off more than we can chew, to avoid analysis paralysis. We will
also continue to build out and refine the long term project vision, to help make
sure the incremental progress does move us closer to the vision.

### Modularity

Modularity is a strong goal for TFRT - not to split TensorFlow into a few
libraries - but we believe the end state will have dozens or perhaps over a
hundred different libraries which can be composed together in different ways for
different products.

This is important because our goal is to build an extremely flexible and
unlimited platform - and one of the limits we want to break through is mobile,
IoT and other limited deployment scenarios. Because of this, TFRT is not
designed as a monolithic layer - it is defined as a "runtime infrastructure"
with many different libraries - each with specifically defined and curated
dependencies.

**Structure of a TFRT Library**

Each TFRT library may have:

*   a public API defined by one or more header files,
*   an implementation consisting of a directory of code,
*   a set of specifically curated dependencies,
*   correctness and performance tests, and
*   documentation.

In order to assist with scaling the project, all of the libraries are organized
in a consistent style, which aims to be as simple as possible.

#### Public API for a TFRT Library

TFRT is designed as an infrastructure that can be used and remixed in different
ways by different clients. As such, it is important to distinguish between a
library's internal implementation and its public API - the public API is the
intended interface to the library.

This has several implications:

-   We explicitly split the headers for the public API out to a directory under
    `include/tfrt`, keeping them disjoint from the implementation files. No
    library is allowed to include files from the implementation details of
    another library.

-   These header files should be kept as minimal as reasonably possible: they
    should make use of forward declarations of other types and should use the
    [pIMPL idiom](https://en.cppreference.com/w/cpp/language/pimpl).

-   The API for a library should either directly map to a directory of header
    files (e.g.
    [`include/tfrt/support`](https://github.com/tensorflow/runtime/blob/master/include/tfrt/support/))
    or be exactly one header file for very simple libraries (e.g.
    [`include/tfrt/bef_converter/mlir_to_bef.h`](https://github.com/tensorflow/runtime/blob/master/include/tfrt/bef_converter/mlir_to_bef.h)).

-   If a library contains no public API (e.g. because its functionality is
    provided by a static constructor registration system) then it may have no
    public header.

Note: The seemingly-unnecessary `tfrt` in `include/tfrt` comes from the way that
the `-I` flag on C++ compilers work: this allows us to use relative `#include`
directories like `#include "tfrt/support/ref_count.h"`, which makes TFRT coexist
well with other infrastructure libraries.

#### Implementation Code for a TFRT Library

Implementation in a library. Optional for header-only libraries.

#### Dependencies for a TFRT Library

-   Specific ops can depend on Eigen, but the core should not.
-   Mobile and small footprint

#### Correctness and Performance Tests

-   Run at scale
    -   Point to testing section below.
-   Linking lots of libraries in a unit test target is not a great thing
    (**N^2** disk space size, lots of link time, etc)
-   xref MLIR testing guide.

#### Documentation

### "Open Source Friendly" Development

We would like the TFRT project to be friendly for open source development.

This imposes many specific goals which shape what we do:

-   It must be possible to build the runtime components on a machine with modest
    resources.

-   We want it to be easy to work on a portion of the runtime (say a new CPU
    kernel, optimizations for a specific hardware target, or the implementation
    of a runtime for a new kind of hardware) with fast build/test/edit cycles.
    This means that we should group tests into test suites, so the user can
    choose which suites to run, rebuilding a very small amount of code. A more
    thorough test run can be done when making a pull request.

-   We want public CI to have many tests, and to make it easy for hardware
    vendors to plug their support into the project's centralized CI system.

Fortunately, many of these goals directly align with our modularity goals.

### Testing

Every behavior changing commit gets a testcase - new functionality and
regression tests.

We take care in making and keeping tests fast. We prefer to write LLVM
[FileCheck](https://llvm.org/docs/CommandGuide/FileCheck.html) based tests,
which treat test cases as data instead of code, reducing the compilation and
execution overhead of the tests.

Mocking of ops and kernels is really easy, and can be used to simulate the
runtime for a piece of hardware by providing an alternate implementation of its
host executor ops/kernels. This is good for testing without the hardware, makes
it easier to exercise failure cases, and speeds iteration time. We'll still need
real hardware tests of course.

See
[MLIR Testing doc](https://github.com/tensorflow/mlir/blob/master/documents/TestingGuide.md)
for relevant discussion.

### Performance

High performance is one of the defining goals of this project -- it is not an
after-thought. As such, we invest heavily in performance engineering work.

### High Quality Implementation in Error Handling

Non-fatal errors should be made recoverable, with no memory leaking or crashing.

Error handling needs to be considered from the beginning, both from the runtime
(reliable recovery from error conditions, machine failures, etc) as well as from
the compiler stack. Error messages from both the compiler and runtime should
relay problems back to the user’s source code by default.

### Debuggability / Introspection

Large scale distributed systems are complicated, and we take care in making the
system debuggable.
